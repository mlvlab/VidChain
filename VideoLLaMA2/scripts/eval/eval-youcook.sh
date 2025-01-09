

CUDA_DEVICE=${1} # Current GPU number
NUM_INDEX=${2} # Current eval set index
TOTAL_GPU=8 # Total number of GPUs (We used 8 GPUs)


# PATHS
OUTPUT_PATH=./outputs/
LOG_FOLDER_NAME=log

#================= CoTasks ================#
CURR_MODEL='finetune_videollama2_youcook2-lora-stage4' 
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python videollama2/eval/inference_video_dvc.py \
    --question-file ./data/YouCook2/val.json \
    --video-folder ./data/YouCook2/videos/ \
    --output-file $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/ \
    --model-path $OUTPUT_PATH/$CURR_MODEL \
    --num-chunks $TOTAL_GPU \
    --chunk-idx $NUM_INDEX \
    --mode 'all' --preload-video --no-grounding



#================= CoTasks + MDPO ================#
STAGE4='finetune_videollama2_youcook2-lora-stage4' 
CURR_MODEL='finetune_videollama2_youcook2-lora-stage5' # stage 5 (M-DPO)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python videollama2/eval/inference_video_dvc.py \
    --question-file ./data/YouCook2/val.json \
    --video-folder ./data/YouCook2/videos/ \
    --output-file $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/ \
    --model-path $OUTPUT_PATH/$STAGE4 \
    --dpo-path $OUTPUT_PATH/$CURR_MODEL \
    --num-chunks $TOTAL_GPU \
    --chunk-idx $NUM_INDEX \
    --mode 'all' --preload-video --no-grounding



# Automatic metric evaluation if CUDA_DEVICE is 0
if [ "$CUDA_DEVICE" -eq "0" ]; then
    echo "Metric Evaluation starts. After 3 minutes (just in case)"
    sleep 3m
    bash scripts/eval/metric-youcook.sh
else
    echo "Finished Generating Results."
fi