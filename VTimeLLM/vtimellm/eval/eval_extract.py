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
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import VideoExtractor
from glob import glob
import random

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, default="checkpoints/vtimellm/ViT-L-14.pt")
    parser.add_argument("--train_path", type=str, default="vtimellm/eval/data_example.json")
    parser.add_argument("--test_path", type=str, default="vtimellm/eval/data_example.json")
    parser.add_argument("--save_path", type=str, default="vtimellm/eval/data_example.json")

    parser.add_argument("--feat_folder", type=str, default=None)
    parser.add_argument("--video_folder", type=str, default=None)

    parser.add_argument("--merge", action='store_true')
    parser.add_argument("--merge_filename",type=str, default="vtimellm/eval/clipvitl14-vtimellm.pth")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()

    save_path = args.save_path
    assert os.path.exists(save_path)

    if not args.merge:

        if args.video_folder is not None:

            print("Loading model..")
            clip_model, _ = clip.load(args.clip_path)
            clip_model.eval()
            clip_model = clip_model.cuda()
            print("Model load complete.")

            video_loader = VideoExtractor(N=100) # 100 frames

            transform = Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        else:
            print("Provide me the video folder")
            assert False

        train = json.load(open(args.train_path))
        test = json.load(open(args.test_path))

        data_keys = list(train.keys()) + list(test.keys())
        data_keys = list(set(data_keys))
        random.shuffle(data_keys) # shuffle keys for each gpu to process different videos
        curr_saved = glob(f'{save_path}*.pth')

        print("*"*95)
        print(f'Save path: {save_path}')
        print(f'Num videos to extract: {len(data_keys)}')
        print(f'Currently saved features: {len(curr_saved)}')
        print("*"*95)

        for id in tqdm(data_keys):

            curr_saved = glob(f'{save_path}*.pth')
            curr_saved = [i.split('/')[-1][:-4] for i in curr_saved]

            if id not in curr_saved:
                features = None

                if features is None and args.video_folder is not None:
                    for ext in ['mp4', 'mkv', 'webm']:
                        video_path = os.path.join(args.video_folder, f"{id}.{ext}")
                        if os.path.isfile(video_path):
                            _, images = video_loader.extract({'id': None, 'video': video_path})
                            try:
                                images = transform(images / 255.0)
                                images = images.to(torch.float16)
                            except:
                                continue
                            with torch.no_grad():
                                features = clip_model.encode_image(images.to('cuda'))
                                break

                if features is None:
                    print(f"Failed to extract: {id}")
                    break
                    
                else:
                    torch.save(features.cpu(), f'{save_path}{id}.pth')
            else:
                print(f"Already exists {id}")
        print("Completed Extraction")


    else:
        video_features = {} # { video id : video feature }
        curr_saved = glob(f'{save_path}*.pth')

        for curr_path in curr_saved:
            vid_feature = torch.load(curr_path)

            # Get video id
            v_id = curr_path.split('/')[-1].split('.')[0]

            video_features[v_id] = vid_feature

        # save video feature
        torch.save(video_features, args.merge_filename)
