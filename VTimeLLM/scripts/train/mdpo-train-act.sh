#!/bin/bash
# Environment Variables
RANK=0,1,2,3,4,5,6,7
MASTER_PORT=29571

# Training Arguments 
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4

# Path Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=vtimellm
MODEL_VERSION=vicuna-v1-5-7b
OUTPUT_DIR=./outputs/
STAGE4=./outputs/vtimellm-vicuna-v1-5-7b-activitynet-stage4

RUN_NAME=vtimellm-$MODEL_VERSION-activitynet-stage5
deepspeed --include localhost:$RANK --master_port $MASTER_PORT vtimellm/train/train_dpo_mem.py \
  --deepspeed ./scripts/zero2.json \
  --lora_enable True --lora_r 64 --lora_alpha 128 \
  --training_stage 3 --finetuning True \
  --model_name_or_path ./checkpoints/vtimellm/vicuna-7b-v1.5 \
  --version v1 \
  --data_path ./data/activitynet/dpo-vtimellm/mdpo-train.json \
  --data_folder ./data/activitynet/ \
  --feat_folder ./data/activitynet/clipvitl14-vtimellm.pth \
  --pretrain_mm_mlp_adapter ./checkpoints/vtimellm/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
  --stage2_path ./checkpoints/vtimellm/vtimellm-vicuna-v1-5-7b-stage2 \
  --stage3_path ./checkpoints/vtimellm/vtimellm-vicuna-v1-5-7b-stage3 \
  --stage4_path $STAGE4 \
  --output_dir $OUTPUT_DIR/$RUN_NAME \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size $LOCAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --save_steps 50000 \
  --save_total_limit 10 \
  --learning_rate 1e-6 \
  --freeze_mm_mlp_adapter True \
  --weight_decay 0. --warmup_ratio 0.1 --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb \
  --run_name $RUN_NAME \
  --gamma 0.0 --beta 0.5 --dpo_alpha 1.0 --train4dpo





