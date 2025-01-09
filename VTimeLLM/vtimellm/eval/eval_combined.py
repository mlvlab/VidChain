import os

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
import sys

sys.path.append(root_dir)

import clip
import re
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init, check_gpu_status
from vtimellm.mm_utils import VideoExtractor
from vtimellm.inference import *
from pycocoevalcap.meteor.meteor import Meteor

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC

import psutil
def set_cpu_affinity(start_idx=0,end_idx=128):
    p = psutil.Process()
    p.cpu_affinity(list(range(start_idx,end_idx)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str,
                        default="checkpoints/vtimellm/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--stage4", type=str, default="")
    parser.add_argument("--stage5", type=str, default="")
    parser.add_argument("--model_base", type=str, default="/path/to/vicuna-7b-v1.5")
    parser.add_argument("--data_path", type=str, default="vtimellm/eval/data_example.json")
    parser.add_argument("--feat_folder", type=str, default=None)
    parser.add_argument("--video_folder", type=str, default=None)
    parser.add_argument("--task", type=str, default='all',
                        choices=['all', 'grounding', 'dvc-capfirst', 'dvc-timefirst'])
    parser.add_argument("--log_path", type=str, default='vtimellm/eval/log')
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--total_gpu", type=int, default=1)
    parser.add_argument("--use_special_token", action='store_true')
    parser.add_argument("--original_query", action='store_true')
    parser.add_argument("--original", action='store_true')
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument("--gt_timestamp", action='store_true')
    parser.add_argument('--generate_samples', action='store_true')
    parser.add_argument('--task2', action='store_true')
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()
    return args


def iou(outputs, gt, args=None):
    if args.use_special_token:
        pattern = r'from <time=(\d+)> to <time=(\d+)>'
    else:
        pattern = r'from (\d+) to (\d+)'
    matches = re.search(pattern, outputs, re.IGNORECASE)

    if not matches:
        if args.use_special_token:
            pattern = r'from  (\d+) to (\d+)'
        else:
            pattern = r'from  <time=(\d+)> to <time=(\d+)>'
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


def write_log(log_path, video_id, task, query_id, answer, info=None):
    log = {
        'video_id': video_id,
        'task': task,
        'query_id': query_id,
        'answer': answer
    }
    if info is not None:
        log['info'] = info
    # make directory if not exist
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    with open(log_path, 'a') as f:
        f.write(json.dumps(log) + '\n')

def write_log_generate(log_path, sample_set):
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    with open(log_path, 'a') as f:
        f.write(json.dumps(sample_set, indent=4) + '\n')


questions = {
    'grounding': ['During which frames can we see {}?'],
    'captioning': [
        'Could you please describe the events in the video in detail? Be specific about the activities of individuals, their surroundings, and interactions with others. The output should be in JSON format, structured as follows: {"event": "xx", "timestamps": "from xx to xx"}.']
}

if __name__ == "__main__":
    # check_gpu_status(gpu_option='cuda')
    set_cpu_affinity(start_idx=0,end_idx=128)
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3, args.stage4, args.stage5)
    model = model.cuda()
    model.to(torch.float16)

    if args.video_folder is not None:
        clip_model, _ = clip.load(args.clip_path)
        clip_model.eval()
        clip_model = clip_model.cuda()

        video_loader = VideoExtractor(N=100)

        transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    if args.feat_folder is not None:
        clip_features = torch.load(f'{args.feat_folder}')

    js = json.load(open(args.data_path))

    total_data = len(js)
    # total_data = 1000
    each_gpu = total_data // args.total_gpu
    js_keys = list(js.keys())

    print("=" * 90)
    if args.num_gpu == args.total_gpu - 1:
        print("Inside left overs ")
        curr_js_keys = js_keys[args.num_gpu * each_gpu:total_data]
    else:
        print("Inside division")
        curr_js_keys = js_keys[args.num_gpu * each_gpu: (args.num_gpu + 1) * each_gpu]
    print(f'Current number of keys: {len(curr_js_keys)}')
    print("=" * 90)

    curr_js = {k: v for k, v in js.items() if k in curr_js_keys}

    
    # Make log path if not exist
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)  

    # Get number of samples that is already completed
    completed_vid = {}
    for this_curr_mode in ['dvc-capfirst', 'dvc-timefirst', 'grounding']:
        completed_vid[this_curr_mode] = []
        
        logs = []
        if this_curr_mode == 'dvc-capfirst':
            path = os.path.join(args.log_path, 'capfirst.txt')
            
            if os.path.isfile(path):
                with open(path) as f:
                    for line in f:
                        try:
                            json_data = json.loads(line)
                            logs.append(json_data)
                        except Exception as e:
                            print(e, line)
        elif this_curr_mode == 'dvc-timefirst':
            path = os.path.join(args.log_path, 'timefirst.txt')
            
            if os.path.isfile(path):
                with open(path) as f:
                    for line in f:
                        try:
                            json_data = json.loads(line)
                            logs.append(json_data)
                        except Exception as e:
                            print(e, line)
        
        elif this_curr_mode == 'grounding':   
            path = os.path.join(args.log_path, 'grounding.txt')
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


    i = 0 # index written outside due to print tqdm
    for (id, data) in tqdm(curr_js.items()):
        video_name = id
        features = None

        if args.feat_folder is not None:
            # feat_path = os.path.join(args.feat_folder, f"{id}.npy")
            # if os.path.isfile(feat_path):
            #     features = torch.from_numpy(np.load(feat_path)).cuda()
            features = clip_features[id].cuda()

        if features is None and args.video_folder is not None:
            for ext in ['mp4', 'mkv', 'webm']:
                video_path = os.path.join(args.video_folder, f"{id}.{ext}")
                if os.path.isfile(video_path):
                    _, images = video_loader.extract({'id': None, 'video': video_path})

                    images = transform(images / 255.0)
                    images = images.to(torch.float16)
                    with torch.no_grad():
                        features = clip_model.encode_image(images.to('cuda'))

        if features is None:
            print(f'Can not find video {id}')
            continue

        if args.generate_samples:
            question = ""

            if args.task2:
                answer_file_time = os.path.join(args.log_path, 'timefirst_task2.txt')
                answer_file_cap = os.path.join(args.log_path, 'capfirst_task2.txt')
            else:
                answer_file_time = os.path.join(args.log_path, 'timefirst.txt')
                answer_file_cap = os.path.join(args.log_path, 'capfirst.txt')


            modes = ['dvc-timefirst', 'dvc-capfirst'] if args.task == 'all' else [args.task]
            # sample generation for DPO dataset construction
            for tm in modes:
                    with torch.autocast(device_type="cuda"):
                        output = x_infer(
                            features,
                            question=question,
                            mode=tm,
                            model=model,
                            tokenizer=tokenizer,
                            do_sample=True,
                            args=args,
                            curr_sample=data,
                        )

                    answer_file = answer_file_time if tm == 'dvc-timefirst' else answer_file_cap
                    sample_set = {'video_id': id, 'task': tm, 'query_id': i, 'answer': output}
                    sample_set.update(output)
                    write_log_generate(answer_file, sample_set)
        else:
            # original inference
            if args.task in ['dvc-capfirst', 'dvc-timefirst', 'all']:
                
                for query_id, query in enumerate(questions['captioning']):
                    query = 'How many of time segments can this video breakdown into?'

                    # capfirst
                    if args.task in ['dvc-capfirst', 'all']:
                
                        if video_name in completed_vid['dvc-capfirst']: # SKIP those that are already finished 
                            print(f'video {video_name} is already finished.. ')
                            continue    
                        
                        cap_log_path = os.path.join(args.log_path, 'capfirst.txt')
                        answer = inference_joint_capdense(model, features, "<video>\n " + query, tokenizer, data, args)
                        write_log(cap_log_path, id, 'captioning', query_id, answer)

                    # timefirst
                    if args.task in ['dvc-timefirst', 'all']:
                        
                        if video_name in completed_vid['dvc-timefirst']: # SKIP those that are already finished 
                            print(f'video {video_name} is already finished.. ')
                            continue    
                        
                        time_log_path = os.path.join(args.log_path, 'timefirst.txt')
                        answer = inference_videoseg_timeseg(model, features, "<video>\n " + query, tokenizer, data, trim=True)
                        write_log(time_log_path, id, 'captioning', query_id, answer)

            # grounding
            if args.task in ['grounding', 'all']:
                
                if video_name in completed_vid['grounding']: # SKIP those that are already finished 
                    print(f'video {video_name} is already finished.. ')
                    continue    
                
                for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
                    sentence = sentence.strip().lower()
                    if sentence.endswith("."):
                        sentence = sentence[:-1]

                    for query_id, query in enumerate(questions['grounding']):
                        grounding_log_path = os.path.join(args.log_path, 'grounding.txt')
                        if not args.original_query:
                            query = "During which frames can we see <CAPTION> in the video?".replace("<CAPTION>", sentence)
                        answer = inference(model, features, "<video>\n" + query, tokenizer, data)
                        gt = (timestamps[0] / data['duration'], timestamps[1] / data['duration'])
                        u = iou(answer, gt, args=args)
                        write_log(grounding_log_path, id, 'grounding', query_id, answer,
                                  info={"sentence_id": sentence_id, 'iou': u})

        i += 1