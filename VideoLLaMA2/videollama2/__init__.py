import copy
from functools import partial

import torch
import time
from .model import Videollama2LlamaForCausalLM, Videollama2MistralForCausalLM
from .model.builder import load_pretrained_model
from .conversation import conv_templates, SeparatorStyle
from .mm_utils import process_video, tokenizer_MMODAL_token, get_model_name_from_path, KeywordsStoppingCriteria
from .constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX
from num2words import num2words
import json
from .train import preprocess_multimodal

def convert(duration, x):
    x = x / duration * 100
    x = str(min(round(x), 99))

    if len(x) == 1:
        x = "0" + x
    return x


def model_init(model_path=None, model_base = None, args=None):
    model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name, args=args)
    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    return model, partial(process_video, aspect_ratio='pad', processor=processor, num_frames=num_frames), tokenizer


def infer(model, video, instruct, tokenizer, do_sample=False, args=None,):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        video (torch.Tensor): video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        args: arguments.
    Returns:
        str: response of the model.
    """

    # 1. vision preprocess (load & transform image or video).
    tensor = [video.half().cuda()]
    modals = ["video"]

    # 2. text preprocess (tag process & generate prompt).
    modal_token = DEFAULT_MMODAL_TOKEN['VIDEO']
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]
    instruct = modal_token + '\n' + instruct

    conv = conv_templates["llama_2"].copy()
    conv.append_message(conv.roles[0], instruct)
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
    return outputs


def inferTimeCap(model, video, instruct, tokenizer, do_sample=False, args=None, isCapFirst=False):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        video (torch.Tensor): video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        args: arguments.
        isCapFirst: whether to cap first.
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
    
    # CoTasks-1
    conv = conv_templates["llama_2"].copy()
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

    curr_token = []
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
    curr_token.append(output_ids.shape[1])
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    first_output = outputs
    print(first_output)


    # CoTasks-2
    conv = conv_templates["llama_2"].copy()
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
    curr_token.append(output_ids.shape[1])
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    second_output = outputs

    # CoTasks-3
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
    curr_token.append(output_ids.shape[1])
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    third_output = outputs
    print(third_output)

    if isCapFirst:
        return second_output + third_output
    else:
        return third_output

def inferTimeCapGenerate(model, video, instruct, tokenizer, do_sample=False, args=None, isCapFirst=False, curr_sample=None):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        video (torch.Tensor): video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        args: arguments.
        isCapFirst: CoTasks(c->t).
        curr_sample: current sample for generation.
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
    first_output = f'{len(curr_sample["sentences"])} time segments'
    
    
    if args.task2: # Task 2
        conv = conv_templates["llama_2"].copy()
        conv.append_message(conv.roles[0], instruct1)
        conv.append_message(conv.roles[1], first_output)
        conv.append_message(conv.roles[0], instruct2)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors='pt').unsqueeze(0).cuda()
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        # 3. generate response according to visual signals and prompts. 
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors='pt').unsqueeze(0).cuda()
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        # 3. generate response according to visual signals and prompts. 
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
        with torch.inference_mode():
            text_outputs = {}
            for num_ in range(args.num_samples):
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    images_or_videos=tensor,
                    modal_list=modals,
                    do_sample=do_sample,
                    temperature=0.8 if do_sample else 0.0,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    top_k=len(tokenizer),
                )
                ids = output_ids.sequences
                text_outputs[num_] = {'sentence':tokenizer.batch_decode(ids[: , :-1], skip_special_token = True)[0].strip()}
                # print(f'isCapFirst?: {isCapFirst}', text_outputs[num_]['sentence'], text_outputs[num_]['logits'])
            return text_outputs
        
    else: # Task 3 
        if isCapFirst:
            second_output = " ".join([i.strip().capitalize() for i in curr_sample['sentences']]).strip()
        else:
            conv_value = []
            for num, time in enumerate(curr_sample['timestamps']):
                text = num2words(num + 1, to='ordinal')        
                conv_value.append(f'{text.capitalize()} event, from {convert(curr_sample["duration"], float(time[0]))} to {convert(curr_sample["duration"], float(time[1]))}.')
            second_output = " ".join(conv_value)
        
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
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors='pt').unsqueeze(0).cuda()
        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        # 3. generate response according to visual signals and prompts. 
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            text_outputs = {}
            for num_ in range(args.num_samples):
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    images_or_videos=tensor,
                    modal_list=modals,
                    do_sample=do_sample,
                    temperature=0.8 if do_sample else 0.0,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    top_k=len(tokenizer),
                )
                ids = output_ids.sequences

                if isCapFirst:
                    text_outputs[num_] = {'sentence': second_output + ' ' + tokenizer.batch_decode(ids[: , 1:-1], skip_special_token = True)[0].strip()}
                else:
                        text_outputs[num_] = {'sentence':tokenizer.batch_decode(ids[: , 1:-1], skip_special_token = True)[0].strip()}
                # print(f'isCapFirst?: {isCapFirst}', text_outputs[num_]['sentence'], text_outputs[num_]['logits'])
    return text_outputs


# Types of inference
def x_infer(video, question, model, tokenizer, mode='vanilla', do_sample=False, args=None, curr_sample=None):
    if mode == 'vanilla':
        instruction = question
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample, args=args)
    
    elif mode == 'dvc-timefirst':
        instructions = ["How many of time segments can this video breakdown into?",
                         "Can you breakdown the video into different time segments?", 
                         "Could you please detail the events that took place during different time segments in the video?"]
        
        if args.generate_samples:              
            return inferTimeCapGenerate(model=model, tokenizer=tokenizer, video=video, instruct=instructions, do_sample=do_sample, args=args, curr_sample=curr_sample)
        else:
            return inferTimeCap(model=model, tokenizer=tokenizer, video=video, instruct=instructions, do_sample=do_sample, args=args)
    
    elif mode == 'dvc-capfirst':
        instructions = ["How many of time segments can this video breakdown into?", 
                        "Can you explain what happend in the video?", 
                        "What are the time segments for each event?"]
        
        if args.generate_samples:
            return inferTimeCapGenerate(model=model, tokenizer=tokenizer, video=video, instruct=instructions, do_sample=do_sample, isCapFirst=True, args=args, curr_sample=curr_sample)
        else:
            return inferTimeCap(model=model, tokenizer=tokenizer, video=video, instruct=instructions, do_sample=do_sample, isCapFirst=True, args=args)
    
    elif mode == 'grounding':
        instruction = f"During which frames can we see {question}?"
        return infer(model=model, tokenizer=tokenizer, video=video, instruct=instruction, do_sample=do_sample, args=args)
