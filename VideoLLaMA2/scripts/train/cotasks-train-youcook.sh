
# Environment Variables
MASTER_PORT=16660
RANK=0,1,2,3,4,5,6,7

# Training Arguments (Global batch: 128, Local batch: 2, Gradient accumulation steps: 8)
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=videollama2
DATA_DIR=./data/
OUTP_DIR=outputs


RUN_NAME=youcook2-lora-stage4
deepspeed --include localhost:$RANK --master_port $MASTER_PORT \
    videollama2/train_flash_attn.py \
    --lora_enable True --lora_r 64 --lora_alpha 128 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero3.json \
    --version v1_mistral \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --mm_projector_type stc_connector \
    --model_name_or_path ./checkpoints/Mistral-7B-Instruct-v0.2 \
    --data_path   ${DATA_DIR}/YouCook2/cotasks-train.json \
    --data_folder ${DATA_DIR}/YouCook2/ \
    --pretrain_mm_mlp_adapter ./checkpoints/VideoLLaMA2-7B-16F-Base/mm_projector.bin \
    --pretrain_model_name_or_path ./checkpoints/VideoLLaMA2-7B-16F \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --num_frames 32 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/finetune_videollama2_${RUN_NAME} \
    --num_train_epochs 5 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 10000 \
    --save_total_limit 99 \
    --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to wandb \
    --run_name $RUN_NAME