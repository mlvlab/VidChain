

<p align="center">
  <h1 align="center">VidChain: Chain-of-Tasks with Metric-based Direct Preference Optimization for Dense Video Captioning</h1>
  
<p align="center">Ji Soo Lee*, Jongha Kim*, Jeehye Na, Jinyoung Park, Hyunwoo J. Kim†.
  </p>

  <h2 align="center">
    AAAI 2025 
  </h2>

  <h3 align="center">
    <a href="https://arxiv.org/pdf/2501.06761" target='_blank'><img src="https://img.shields.io/badge/arXiv-2501.06761-b31b1b.svg"></a>
    <a href="https://huggingface.co/datasets/simplecloud/VidChain-Data"><img src="https://img.shields.io/badge/huggingface-datasets-yellow"></a>
  </h3>

  
This is the official implementation (pytorch) of VidChain, a novel framework for Dense Video Captioning with VideoLLMs, which composes of Chain-of-Tasks and Metric-based Direct Preference Optimization. 

</p>

<div align="center">
  <img src="asset/main.png" width="750px" />
</div>

## Setup for VideoLLaMA2
### 1. Clone the Repository

```bash
git clone https://github.com/mlvlab/VidChain.git
cd VidChain
```

### 2. Install Dependencies
```bash
conda create -n videollama python=3.10 -y
conda activate videollama
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

cd VideoLLaMA2
pip install -r requirements.txt
pip install num2words datasets pycocoevalcap rich
pip install flash-attn==2.5.7 --no-build-isolation

```

### 3. Download the pre-trained checkpoints from [link](https://github.com/DAMO-NLP-SG/VideoLLaMA2?tab=readme-ov-file#earth_americas-model-zoo).

### 4. Download our checkpoints from [huggingface](https://huggingface.co/datasets/simplecloud/VidChain-Data).
- We provide the pre-extracted features of VideoLLaMA2/VTimeLLM for both ActivityNet and YouCook2. Note that the pre-extracted features of VideoLLaMA2 is about ⚠️ 301GB (act) and 32GB (yc2), please be aware of the storage space.
- We also provide the log results for each checkpoint.
- stage 4 corresponds to CoTasks, stage 5 corresponds to M-DPO

<br>
<details>
<summary> Directory Setup Details</summary>

```bash
#====== VidChain Checkpoints ======#
./outputs # Put our VidChain checkpoints here (CoTasks and MDPO)
   └─ finetune_videollama2_activitynet-lora-stage4
       └─ ...
   └─ finetune_videollama2_activitynet-lora-stage5
       └─ ...
   └─ finetune_videollama2_youcook2-lora-stage4
       └─ ...
   └─ finetune_videollama2_youcook2-lora-stage5
       └─ ...


#====== Pretrained Checkpoints ======#
./checkpoints # Put your pretrained checkpoint here
   └─ clip-vit-large-patch14-336
       └─ ...
   └─ Mistral-7B-Instruct-v0.2
       └─ ...
   └─ VideoLLaMA2-7B-16F   
       └─ ...
   └─ VideoLLaMA2-7B-16F-Base   
       └─ ...


#======= Data =======#
./data # Put your data here
   └─ activitynet
       |─ videos # Original videos (option 1)
       |   └─ ...
       |─ videollama2_features # for pre-extracted features (option 2)
       |   └─ ...
       |─ train.json
       |─ val_2.json
       |─ cotasks-train.json # for CoTasks training
       |─ dpo-videollama2    # for M-DPO training
       |   └─ mdpo-train.json

   └─ YouCook2
       |─ videos # Original videos (option 1)
       |   └─ ...
       |─ videollama2_features # for pre-extracted features (option 2)
       |   └─ ...
       |─ train.json
       |─ val.json 
       |─ cotasks-train.json # for CoTasks training
       |─ dpo-videollama2    # for M-DPO training
       |   └─ mdpo-train.json
```
</details>

<br>

## Training & Evaluation Script
We provide the evaluation and train script in `./scripts/train/`, `./scripts/eval/`. Please refer to the script for more details. To train and evaluate on YouCook2, simply run scripts with ``youcook`` in the script name.

<br>

### Dense Video Captioning Evaluation
```bash
# Dense Video Captioning Evaluation
bash script/eval/eval-act.sh $CUDA_DEVICE $NUM_INDEX  # CoTasks & M-DPO
```
- We evaluate with multiple-gpus, where each gpu (`$CUDA_DEVICE`) is assigned to a different chunk of eval set (`$NUM_INDEX`).
- E.g., with 2 gpus (id: 0, 1) set `TOTAL_GPU=2`, and run `bash script/train/cotasks-train-act.sh 0 0` and `bash script/train/cotasks-train-act.sh 1 1` to evaluate on the first and second chunks of eval set, respectively. For best reproducability, set `TOTAL_GPU` to 8.


```bash
# Metric Evaluation
bash script/eval/metric-act.sh
```

<br>

### Training for CoTasks and M-DPO
```bash
# Dense Video Captioning Training
bash script/train/cotasks-train-act.sh  # CoTasks 
bash script/train/mdpo-train-act.sh  # M-DPO
```

```bash
# M-DPO Sample Generation
bash script/build/generate-act.sh $CUDA_DEVICE $NUM_INDEX # Generation
bash script/build/generate-build-act.sh # Evaulate Generated samples
python script/build/concat.py # Build training data for M-DPO
```


### Feature Extraction Code
``` bash
bash extract.sh $CUDA_DEVICE
```
We provide the pre-extracted video features, yet we also provide the code.

<br>

---
## Setup for VTimeLLM

### 1. Clone the Repository
```bash
git clone https://github.com/mlvlab/VidChain.git
cd VidChain
```

### 2. Install Dependencies
```bash
conda create -n vtimellm python=3.10 -y
conda activate vtimellm
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

cd VTimeLLM
pip install -r requirements.txt
pip install ninja num2words pycocoevalcap datasets timm
pip install flash-attn --no-build-isolation
```

### 3. Download the Pre-trained/Finetuned Checkpoints from [VTimeLLM](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2F&mode=list), and [huggingface](https://huggingface.co/datasets/simplecloud/VidChain-Data).

<details>
<summary>Path Setup Details</summary>

```bash
#====== VidChain Checkpoints ======#
./outputs # Put our VidChain checkpoints here (CoTasks and MDPO)
   └─ vtimellm_vicuna-v1-5-7b-activitynet-stage4
       └─ ...
   └─ vtimellm_vicuna-v1-5-7b-activitynet-stage5
       └─ ...
   └─ vtimellm-vicuna-v1-5-7b-youcook-stage4
       └─ ...
   └─ vtimellm-vicuna-v1-5-7b-youcook-stage5
       └─ ...


#====== Pretrained Checkpoints ======#
./checkpoints # Put your pretrained checkpoint here
   └─ vtimellm
       └─ vicuna-7b-v1.5
            └─ ...   
       └─ vtimellm-vicuna-v1-5-7b-stage1
            └─ ...   
       └─ vtimellm-vicuna-v1-5-7b-stage2
            └─ ...   
       └─ vtimellm-vicuna-v1-5-7b-stage3
            └─ ...   
       └─ ViT-L-14.pt

#====== Data  ======#
./data # Put your data here
   └─ activitynet
       |─ videos # Original videos (option 1)
       |   └─ ...
       |─ clipvitl14-vtimellm.pth # for pre-extracted features (option 2)
       |─ train.json
       |─ val_2.json
       |─ cotasks-train.json # for CoTasks training
       |─ dpo-vtimellm       # for M-DPO training
       |   └─ mdpo-train.json

   └─ YouCook2
       |─ videos # Original videos (option 1)
       |   └─ ...
       |─ clipvitl14-vtimellm.pth # for pre-extracted features (option 2)
       |─ train.json
       |─ val.json
       |─ cotasks-train.json # for CoTasks training
       |─ dpo-vtimellm       # for M-DPO training
       |   └─ mdpo-train.json
```
</details>

<br>

## Training & Evaluation Script

### Dense Video Captioning Evaluation
```bash
# Dense Video Captioning Evaluation
bash script/eval/eval-act.sh $CUDA_DEVICE $NUM_INDEX  # CoTasks & M-DPO
```


```bash
# Metric Evaluation
bash script/eval/metric-act.sh
```

<br>

### Training for CoTasks and M-DPO
```bash
# Dense Video Captioning Training
bash script/train/cotasks-train-act.sh  # CoTasks 
bash script/train/mdpo-train-act.sh  # M-DPO
```

```bash
# M-DPO Sample Generation
bash script/build/generate-act.sh $CUDA_DEVICE $NUM_INDEX # Generation
cd ..
cd VideoLLaMA2
conda activate videollama
bash script/build/generate-build-act-vtimellm.sh # Evaluation 
python script/build/concat.py # Build training data for M-DPO
```
- Note that the evaluation script for the generated samples is based on VideoLLaMA2 codebase, so you need to set `vtimellm=True` and pass `--vtimellm` to the script.


<br>

## Citations 🌱

```
@inproceedings{lee2025vidchain,
  title={VidChain: Chain-of-Tasks with Metric-based Direct Preference Optimization for Dense Video Captioning},
  author={Lee, Ji Soo and Kim, Jongha and Na, Jeehye and Park, Jinyoung and Kim, Hyunwoo J},
  booktitle={AAAI},
  year={2025}
}
```