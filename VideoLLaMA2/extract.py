import torch
import transformers

import sys
import json
from glob import glob
from tqdm import tqdm
import random
import argparse


sys.path.append('./')
from videollama2.conversation import conv_templates
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from videollama2.model.builder import load_pretrained_model


def inference(args):

    #########################################################################################################
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)

    with torch.no_grad():
        tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
    #######################################################################################################

    train_data = json.load(open(args.train_anno))
    test_data = json.load(open(args.test_anno))

    train_vid = train_data.keys()
    test_vid = test_data.keys()

    total_vid = list(train_vid) + list(test_vid)
    total_vid = list(set(total_vid))


    print('*'* 100)
    print(f"Total videos: {len(total_vid)}")

    output_path = args.save_folder_path

    # Get completed videos
    compelete_vid = glob(f'{output_path}/*.pth')
    compelete_vid = [i.split('/')[-1].split('.')[0] for i in compelete_vid]

    # Get all availabel videos path
    videos_path = glob(args.videos_glob)
    videos_path = {data.split('/')[-1].split('.')[0]: data for data in videos_path }

    data_dict = {}
    for key in videos_path:
        if key in total_vid and key not in compelete_vid:
            data_dict[key] = videos_path[key] # set video path


    print(f"Total to do : {len(data_dict)}")
    print('*'* 100)

    model = model.to('cuda:0')
    loop_keys = list(data_dict.keys())

    # random shuffle such that GPUs take different samples!! 
    random.shuffle(loop_keys)

    for key in tqdm(loop_keys):

        path = data_dict[key]
        compelete_vid = glob(f'{output_path}/*.pth')
        compelete_vid = [i.split('/')[-1].split('.')[0] for i in compelete_vid]

        if key in compelete_vid:
            print(f'Already processed video:  {key}...')
            continue
        
        try:
            tensor = process_video(path, processor, "pad", num_frames=32, sample_scheme='uniform').to(dtype=torch.float16, device='cpu', non_blocking=True)
            print(f'Video tensor file size: {tensor.shape}')
        except Exception as E:
            print(E)
            print(f"Video {key} triggered an error. Failed to decode")
            return False
        assert tensor.shape[0] == 32
        
        torch.save(tensor, f'{output_path}/{key}.pth')
        print(f'Complete processing {key}.., output size: {tensor.shape}')
        del tensor



def parse_args():
    parser = argparse.ArgumentParser(description="Extract YouCook for VideoLLaMA")
    parser.add_argument("--model_path", type=str, default="./checkpoints/VideoLLaMA2.1-7B-16F")
    parser.add_argument("--train_anno", type=str, default="./data/YouCook2/train.json") 
    parser.add_argument("--test_anno", type=str, default="./data/YouCook2/val.json") 
    parser.add_argument("--videos_glob", type=str, default="./data/YouCook2/videos/*/*/*") 
    parser.add_argument("--save_folder_path", type=str, default="./data/YouCook2/videollama2_features/") 

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    inference(args)