import math
import os
import argparse
import json
import warnings
from tqdm import tqdm
import torch
import time
import re
import difflib
from glob import glob
import random

import sys
sys.path.append('./')
from videollama2 import model_init, x_infer
from videollama2.eval.eval_utils import captioning_metrics, grounding_metrics
import pickle
import psutil
import numpy as np

def set_cpu_affinity(start_idx=0,end_idx=128):
    p = psutil.Process()
    p.cpu_affinity(list(range(start_idx,end_idx)))


# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split2chunk(gt_questions, total_gpu, num_gpu):
    total_data = len(gt_questions)
    each_gpu = total_data // total_gpu
    gt_keys = list(gt_questions.keys())
    
    print("=" * 90)
    if num_gpu == total_gpu - 1:
        print("Inside left overs ")
        curr_gt_keys = gt_keys[num_gpu * each_gpu: ]    
    else:
        print("Inside division")
        curr_gt_keys = gt_keys[num_gpu * each_gpu: (num_gpu + 1) * each_gpu]
    print(f'Current number of keys: {len(curr_gt_keys)}')
    print("=" * 90)

    curr_gt = {k:v for k,v in gt_questions.items() if k in curr_gt_keys}
    return curr_gt

def iou(outputs, gt, args=None):

    pattern = r'from (\d+) to (\d+)'
    matches = re.search(pattern, outputs, re.IGNORECASE)

    if not matches:
        pattern = r'from  (\d+) to (\d+)'
        matches = re.search(pattern, outputs, re.IGNORECASE)

        if not matches:
            return 0
    
    from_number = float(matches.group(1)) / 100
    to_number = float(matches.group(2)) / 100

    s, e = gt
    intersection = max(0, min(to_number, e) - max(from_number, s))
    union = max(to_number, e) - min(from_number, s)
    iou = intersection / union
    return round(iou, 2)

def write_log(log_path, sample_set):
    with open(log_path, 'a') as f:
        f.write(json.dumps(sample_set) + '\n')

def convert(duration, x):
    x = x / duration * 100
    x = str(min(round(x), 99))

    if len(x) == 1:
        x = "0" + x
    return x

def print_metrics(metrics, args=None):
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    if args is not None:
        metrics['type'] = args.mode
        metric_path = os.path.join(args.output_file, 'metric.txt')
        write_log(metric_path, metrics)

def run_inference(args):

    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = split2chunk(gt_questions, args.num_chunks, args.chunk_idx)
    args.gt_questions = gt_questions
    
    assert args.output_file[-1] == '/'
    answer_file_ground = os.path.join(args.output_file, 'grounding.txt')
    answer_file_time = os.path.join(args.output_file, 'timefirst.txt')
    answer_file_cap = os.path.join(args.output_file, 'capfirst.txt')
    answer_file_single = os.path.join(args.output_file, 'single.txt')
    
    if args.task2:
        answer_file_cap_t2 = os.path.join(args.output_file, 'capfirst_task2.txt')
        answer_file_time_t2 = os.path.join(args.output_file, 'timefirst_task2.txt')

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)


        
    # Build model
    with torch.no_grad():
        model, processor, tokenizer = model_init(args.model_path, args.model_base, args=args)

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    curr_mode = args.mode

    # mode 'single' set args.single = True
    if args.mode == 'single':
        args.single = True  

    if 'activitynet' in args.video_folder:
        
        if args.preload_video:
            video_features = glob('./data/activitynet/videollama_features*/*.pth')
        else:
            video_features = glob('./data/activitynet/videos/*')
        frames = args.num_frames
        print(f'Frames: {frames}')
    else:
        if args.preload_video:
            video_features = glob('./data/YouCook2/videollama2_features/*.pth')
        else:
            video_features = glob('./data/YouCook2/raw_videos/*/*/*')
        frames = args.num_frames
        print(f'Frames: {frames}')
        
    vid2path = {i.split('/')[-1].split('.')[0]: i for i in video_features}

    # Get number of samples that is already completed
    completed_vid = {}
    for this_curr_mode in ['dvc-capfirst', 'dvc-timefirst', 'grounding']:
        completed_vid[this_curr_mode] = []
        
        logs = []
        if this_curr_mode == 'dvc-capfirst':
            path = answer_file_cap if not args.task2 else answer_file_cap_t2
            
            if os.path.isfile(path):
                with open(path) as f:
                    for line in f:
                        try:
                            json_data = json.loads(line)
                            logs.append(json_data)
                        except Exception as e:
                            print(e, line)
        elif this_curr_mode == 'dvc-timefirst':
            path = answer_file_time if not args.task2 else answer_file_time_t2
            if os.path.isfile(path):
                with open(path) as f:
                    for line in f:
                        try:
                            json_data = json.loads(line)
                            logs.append(json_data)
                        except Exception as e:
                            print(e, line)
        
        elif this_curr_mode == 'grounding':   
            path = answer_file_ground
            if os.path.isfile(path):
                with open(path) as f:
                    for line in f:
                        try:
                            json_data = json.loads(line)
                            logs.append(json_data)
                        except Exception as e:
                            print(e, line)                 
        
        completed_vid[this_curr_mode].extend([i['video_id'] for i in logs])
        

    
    print(f"Number of videos already completed in total: Capfirst {len(completed_vid['dvc-capfirst'])},  TimeFirst {len(completed_vid['dvc-timefirst'])}")
    print("=" * 90)


    num_id = 0
    for idx, sample in tqdm(gt_questions.items()):
        
        num_id += 1
        query_id = 0
        video_name = idx
        question = ""
        sample['idx'] = idx
        
        args.num_times = []
        args.num_tokens = []
        
        video_tensor = None
        if not args.preload_video:
            for ext in video_formats:
                video_path = os.path.join(args.video_folder, f'{video_name}{ext}') # get video

                if os.path.isfile(video_path):
                    video_tensor = processor(video_path)
                    break
        else: # preloaded video
            try:
                video_path = vid2path[video_name]
                if os.path.isfile(video_path):
                    if frames == 32:
                        video_tensor = torch.load(video_path)
                    else:
                        assert "Need to implement!"
            except:
                video_path = ""
                pass
            
    
        if video_tensor is None:
            print(f'Can not find video {video_path}, {video_name}')
            continue

        else:
            if not args.single:
                if curr_mode in ['dvc-capfirst', 'dvc-timefirst']:
                    
                    if video_name in completed_vid[curr_mode]: # SKIP those that are already finished 
                        print(f'video {video_name} is already finished.. ')
                        continue    
                    
                    with torch.autocast(device_type="cuda"):
                        output = x_infer(
                            video_tensor,
                            question=question, 
                            mode=curr_mode,
                            model=model,
                            tokenizer=tokenizer,
                            do_sample=True,
                            args=args,
                            curr_sample=sample,
                        )

                    answer_file = answer_file_time if curr_mode == 'dvc-timefirst' else answer_file_cap
                    sample_set = {'video_id': video_name, 'task': curr_mode, 'query_id': num_id, 'answer': output}
                    write_log(answer_file, sample_set)

                if curr_mode in ['all']: # Loop over two modes
                    two_modes = ['dvc-capfirst', 'dvc-timefirst']

                    for tm in two_modes:
                        if video_name in completed_vid[tm]: # SKIP those that are already finished 
                            print(f'video {video_name} is already finished.. ')
                            continue
                        
                        # Generate samples (for M-DPO)
                        if args.generate_samples:
                            if args.task2:
                                with torch.autocast(device_type="cuda"):
                                    output = x_infer(
                                        video_tensor,
                                        question=question, 
                                        mode=tm,
                                        model=model,
                                        tokenizer=tokenizer,
                                        do_sample=True,
                                        args=args,
                                        curr_sample=sample,
                                    )
                            
                                answer_file = answer_file_time_t2 if tm == 'dvc-timefirst' else answer_file_cap_t2
                                sample_set = {'video_id': video_name, 'task': tm, 'query_id': num_id, 'answer': output}
                                sample_set.update(output)
                                write_log(answer_file, sample_set)
                            else:
                                with torch.autocast(device_type="cuda"):
                                    output = x_infer(
                                        video_tensor,
                                        question=question, 
                                        mode=tm,
                                        model=model,
                                        tokenizer=tokenizer,
                                        do_sample=True,
                                        args=args,
                                        curr_sample=sample,
                                    )
                                
                                answer_file = answer_file_time if tm == 'dvc-timefirst' else answer_file_cap
                                sample_set = {'video_id': video_name, 'task': tm, 'query_id': num_id, 'answer': output}
                                sample_set.update(output)
                                write_log(answer_file, sample_set) 

                        # Original Inference
                        else:
                            with torch.autocast(device_type="cuda"):
                                output = x_infer(
                                    video_tensor,
                                    question=question, 
                                    mode=tm,
                                    model=model,
                                    tokenizer=tokenizer,
                                    do_sample=True,
                                    args=args,
                                    curr_sample=sample,
                                )
                            
                            answer_file = answer_file_time if tm == 'dvc-timefirst' else answer_file_cap
                            sample_set = {'video_id': video_name, 'task': tm, 'query_id': num_id, 'answer': output}
                            write_log(answer_file, sample_set)

                # Temporal Video Grounding 
                if curr_mode in ['grounding', 'all'] and not args.no_grounding:
                    sentences = sample['sentences']
                    timestamps = sample['timestamps']
                    
                    if video_name in completed_vid['grounding']: # SKIP those that are already finished 
                        print(f'video {video_name} is already finished.. ')
                        continue
                    
                    for sen, time in zip(sentences, timestamps):
                        query_id += 1
                        question = sen.lower().replace('.', '').strip()

                        with torch.autocast(device_type="cuda"):
                            output = x_infer(
                                video_tensor,
                                question=question, 
                                mode='grounding',
                                model=model,
                                tokenizer=tokenizer,
                                do_sample=True,
                                args=args,
                            )

                        gt_time = (time[0] / sample['duration'], time[1] / sample['duration'])
                        u = iou(output, gt_time, args=args) # get iou with Ground Truth 
                        sample_set = {'video_id': video_name, 'task': curr_mode, 'query_id': num_id,'answer': output, 'info': {'sentenece_id': query_id, 'iou': u}}
                        write_log(answer_file_ground, sample_set)
            
            else:
                # Single mode (e.g., Baseline)
                curr_mode = 'vanilla'
                question = "Could you outline the incidents that occurred at various timestamps in the video?"
                with torch.autocast(device_type="cuda"):
                    output = x_infer(
                        video_tensor,
                        question=question, 
                        mode=curr_mode,
                        model=model,
                        tokenizer=tokenizer,
                        do_sample=True,
                        args=args,
                    )
                
                sample_set = {'video_id': video_name, 'task': curr_mode, 'query_id': num_id, 'answer': output}
                write_log(answer_file_single, sample_set)


def run_eval(args):
    assert args.mode in ['dvc-capfirst', 'dvc-timefirst', 'grounding', 'single']

    if args.mode == 'dvc-capfirst':
        answer_file = os.path.join(args.output_file, 'capfirst.txt')
    elif args.mode == 'dvc-timefirst':
        answer_file = os.path.join(args.output_file, 'timefirst.txt')
    elif args.mode == 'single':
        answer_file = os.path.join(args.output_file, 'single.txt')
    else:
        answer_file = os.path.join(args.output_file, 'grounding.txt')

    logs = []
    with open(answer_file) as f:
        for line in f:
            try:
                json_data = json.loads(line)
                logs.append(json_data)
            except Exception as e:
                print(e, line)

    print("*"* 150)
    print(args.output_file, args.mode)
    print("*"* 150)
    
    if args.mode in ['dvc-capfirst', 'dvc-timefirst', 'single']:
        print("====================== Captioning =====================")
        print_metrics(captioning_metrics(logs, args.question_file, print_matrix=args.print_matrix, args=args), args=args)
    if args.mode in ['grounding']:
        print("====================== Grounding ======================")
        print_metrics(grounding_metrics(logs), args=args)



if __name__ == "__main__":
    set_cpu_affinity(start_idx=0,end_idx=128)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--dpo-path', type=str, default='') 
    parser.add_argument('--model-base', type=str, default='./checkpoints/VideoLLaMA2-7B-16F')
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--preload-video', action='store_true')
    
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--save-json', action='store_true')
     
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--debug",  action='store_true')
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    
    
    parser.add_argument('--mode', help='Which inference mode to use', required=True,  choices=['all', 'dvc-capfirst', 'dvc-timefirst', 'grounding', 'single', 'logit'])
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--no-grounding', action='store_true')
    parser.add_argument('--metric', action='store_true')

    parser.add_argument('--print_matrix', action='store_true')


    # For sample generation
    parser.add_argument('--generate-samples', action='store_true')
    parser.add_argument('--task2', action='store_true')
    parser.add_argument('--num_samples', type=int, default=2)
    
    
    args = parser.parse_args()

    if not args.metric:
        run_inference(args)
    else:
        run_eval(args)
