import os
import sys
import argparse
import torch
from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.model.builder import load_pretrained_model, load_lora
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
from num2words import num2words
import time

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip
import re

def convert(duration, x):
    x = x / duration * 100
    x = str(min(round(x), 99))

    if len(x) == 1:
        x = "0" + x
    return x

def generate_and_decode(model, input_ids, tokenizer, video, max_token, stop_str):
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video[None,].cuda(),
            do_sample=True,
            temperature=0.8,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_token,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True)

    output_seq = output_ids.sequences
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_seq[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_seq[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    sentence = outputs.strip()

    return sentence

def inference(model, image, query, tokenizer, data, max_token=1024, args=None):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=15,
            max_new_tokens=max_token,
            use_cache=True,
        )
    
    # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs



def inferTimeCap(model, video, instruct, tokenizer, do_sample=False, args=None, isCapFirst=False):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        video (torch.Tensor): video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
    Returns:
        str: response of the model.
    """

    # 1. vision preprocess (load & transform image or video).
    tensor = [video.half().cuda()]
    modals = ["video"]

    # 2. text preprocess (tag process & generate prompt).
    modal_token = DEFAULT_MMODAL_TOKEN['VIDEO']
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]

    instruct1 = modal_token + '\n' + instruct[0]
    instruct2 = instruct[1]
    instruct3 = instruct[2]

    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], instruct1)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors='pt').unsqueeze(0).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts.
    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    # keywords = ["<s>", "</s>"]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images_or_videos=tensor,
            modal_list=modals,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    first_output = outputs
    print(first_output)

    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], instruct1)
    conv.append_message(conv.roles[1], first_output)
    conv.append_message(conv.roles[0], instruct2)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors='pt').unsqueeze(0).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts.
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images_or_videos=tensor,
            modal_list=modals,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    second_output = outputs
    # print(second_output)

    conv = conv_templates["llama_2"].copy()
    conv.append_message(conv.roles[0], instruct1)
    conv.append_message(conv.roles[1], first_output)
    conv.append_message(conv.roles[0], instruct2)
    conv.append_message(conv.roles[1], second_output)
    conv.append_message(conv.roles[0], instruct3)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors='pt').unsqueeze(0).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts.
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images_or_videos=tensor,
            modal_list=modals,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    third_output = outputs
    print(third_output)
    
    if isCapFirst:
        return second_output + third_output
    else:
        return third_output

def inference_joint_capdense(model, image, query, tokenizer, data, args, max_token=1024):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_token,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    first_output = outputs
    print(outputs)

    
    query1 = f'Can you explain what happend in the video?'
    conv1 = conv_templates["v1"].copy()
    conv1.append_message(conv1.roles[0], query)
    conv1.append_message(conv1.roles[1], outputs)
    conv1.append_message(conv1.roles[0], query1)
    conv1.append_message(conv1.roles[1], None)
    prompt1 = conv1.get_prompt()
    input_ids = tokenizer_image_token(prompt1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv1.sep if conv1.sep_style != SeparatorStyle.TWO else conv1.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_token,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    second_output = outputs
    print(second_output)

    # query2 = "Could you please detail the events that took place during different time segments in the video?"
    query2 = "What are the time segments for each event?"
    conv1 = conv_templates["v1"].copy()
    conv1.append_message(conv1.roles[0], query)
    conv1.append_message(conv1.roles[1], first_output)
    conv1.append_message(conv1.roles[0], query1)
    conv1.append_message(conv1.roles[1], second_output)
    conv1.append_message(conv1.roles[0], query2)
    conv1.append_message(conv1.roles[1], None)
    prompt1 = conv1.get_prompt()
    input_ids = tokenizer_image_token(prompt1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv1.sep if conv1.sep_style != SeparatorStyle.TWO else conv1.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_token,
            use_cache=True)



        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)

    events = second_output.split('.')[:-1]
    timestamps = outputs.split('.')[:-1]


    final_output = ""
    for num, t in enumerate(timestamps):
        if args.use_special_token:
            pattern = r'from <time=(\d+)> to <time=(\d+)>'
        else:
            pattern = r'from (\d+) to (\d+)'
        matches = re.search(pattern, t, re.IGNORECASE)

        if matches is not None:
            try:
                final_output += f"{events[num].capitalize()}, {matches.group(0)}."
            except:
                print(len(timestamps), len(events), "there is a missmatch")

    return final_output


def inference_videoseg_timeseg(model, image, query, tokenizer, data, trim=False,  max_token=1024, args=None):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_token,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295
    
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    # outputs = f"{len(data['sentences'])} time segments"
    first_output = outputs
    print(outputs)

    query1 = 'Can you breakdown the video into different time segments?.'
    conv1 = conv_templates["v1"].copy()
    conv1.append_message(conv1.roles[0], query)
    conv1.append_message(conv1.roles[1], first_output)
    conv1.append_message(conv1.roles[0], query1)
    conv1.append_message(conv1.roles[1], None)
    prompt1 = conv1.get_prompt()
    input_ids = tokenizer_image_token(prompt1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv1.sep if conv1.sep_style != SeparatorStyle.TWO else conv1.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_token,
            use_cache=True)
        
        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    second_output = outputs
    print(second_output)

    query2 = f'Could you please detail the events that took place during different time segments in the video?'
    conv1 = conv_templates["v1"].copy()
    conv1.append_message(conv1.roles[0], query)
    conv1.append_message(conv1.roles[1], first_output)
    conv1.append_message(conv1.roles[0], query1)
    conv1.append_message(conv1.roles[1], second_output)
    conv1.append_message(conv1.roles[0], query2)
    conv1.append_message(conv1.roles[1], None)
    prompt1 = conv1.get_prompt()
    input_ids = tokenizer_image_token(prompt1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv1.sep if conv1.sep_style != SeparatorStyle.TWO else conv1.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_token,
            use_cache=True)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs
    

def inferTimeCapGenerate(model, video, instruct, tokenizer, do_sample=False, args=None, isCapFirst=False,
                         curr_sample=None, max_token=1024):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        video (torch.Tensor): video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
    Returns:
        str: response of the model.
    """

    instruct1 = instruct[0] # todo: sanity check
    instruct2 = instruct[1]
    instruct3 = instruct[2]
    first_output = f'{len(curr_sample["sentences"])} time segments'

    if args.task2:  # Task 2
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], instruct1)
        conv.append_message(conv.roles[1], first_output)
        conv.append_message(conv.roles[0], instruct2)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        with torch.inference_mode():
            text_outputs = {}
            for num_ in range(args.num_samples):
                sentence = generate_and_decode(model, input_ids, tokenizer, video, max_token, stop_str)

                text_outputs[num_] = {'sentence': sentence}
                print(f'isCapFirst?: {isCapFirst}', text_outputs[num_]['sentence'])

            return text_outputs

    else:  # Task 3
        if isCapFirst:
            second_output = " ".join([i.strip().capitalize() for i in curr_sample['sentences']]).strip()
        else:
            conv_value = []
            for num, time in enumerate(curr_sample['timestamps']):
                text = num2words(num + 1, to='ordinal')
                conv_value.append(
                    f'{text.capitalize()} event, from {convert(curr_sample["duration"], float(time[0]))} to {convert(curr_sample["duration"], float(time[1]))}.')
            second_output = " ".join(conv_value)

        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], instruct1)
        conv.append_message(conv.roles[1], first_output)
        conv.append_message(conv.roles[0], instruct2)
        conv.append_message(conv.roles[1], second_output)
        conv.append_message(conv.roles[0], instruct3)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        with torch.inference_mode():
            text_outputs = {}
            for num_ in range(args.num_samples):
                sentence = generate_and_decode(model, input_ids, tokenizer, video, max_token, stop_str)

                if isCapFirst:
                    text_outputs[num_] = {
                        'sentence': second_output + ' ' + sentence}
                else:
                    text_outputs[num_] = {
                        'sentence': sentence}
                print(f'isCapFirst?: {isCapFirst}', text_outputs[num_]['sentence'])
    return text_outputs


# Types of inference
def x_infer(video, question, model, tokenizer, mode='vanilla', do_sample=False, args=None, curr_sample=None):
    if mode == 'vanilla':
        instruction = question
        return inference(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample,
                     args=args)

    elif mode == 'dvc-timefirst':
        instructions = ["How many of time segments can this video breakdown into?",
                        "Can you breakdown the video into different time segments?",
                        "Could you please detail the events that took place during different time segments in the video?"]

        if args.generate_samples:
            return inferTimeCapGenerate(model=model, tokenizer=tokenizer, video=video, instruct=instructions,
                                        do_sample=do_sample, args=args, curr_sample=curr_sample)

    elif mode == 'dvc-capfirst':
        instructions = ["How many of time segments can this video breakdown into?",
                        "Can you explain what happend in the video?", "What are the time segments for each event?"]

        if args.generate_samples:
            return inferTimeCapGenerate(model=model, tokenizer=tokenizer, video=video, instruct=instructions,
                                        do_sample=do_sample, isCapFirst=True, args=args, curr_sample=curr_sample)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/path/to/vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--video_path", type=str, default="images/demo.mp4")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=100)
    _, images = video_loader.extract({'id': None, 'video': args.video_path})

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # print(images.shape) # <N, 3, H, W>
    images = transform(images / 255.0)
    images = images.to(torch.float16)
    with torch.no_grad():
        features = clip_model.encode_image(images.to('cuda'))

    query = "describe the video."
    print("query: ", query)
    print("answer: ", inference(model, features, "<video>\n " + query, tokenizer))


