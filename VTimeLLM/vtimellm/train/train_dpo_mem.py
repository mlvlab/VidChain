# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.

import os
import time
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
print(root_dir)
import sys
sys.path.append(root_dir)
import psutil


def set_cpu_affinity(start_idx=64,end_idx=192):
    p = psutil.Process()
    p.cpu_affinity(list(range(start_idx,end_idx)))

from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from train_dpo import train
from vtimellm.utils import check_gpu_status

if __name__ == "__main__":
    set_cpu_affinity(start_idx=0,end_idx=128)
    train()
