
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

RUN_NAME=vtimellm-$MODEL_VERSION-activitynet-stage4
deepspeed --include localhost:$RANK --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --training_stage 3 --finetuning True \
    --model_name_or_path ./checkpoints/vtimellm/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./data/activitynet/cotasks-train.json \
    --feat_folder ./data/activitynet/clipvitl14-vtimellm.pth \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --stage2_path ./checkpoints/vtimellm/vtimellm-$MODEL_VERSION-stage2 \
    --stage3_path ./checkpoints/vtimellm/vtimellm-$MODEL_VERSION-stage3 \
    --output_dir $OUTPUT_DIR/${RUN_NAME} \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 --lora_alpha 128 --weight_decay 0. --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME