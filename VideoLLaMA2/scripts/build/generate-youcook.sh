

CUDA_DEVICE=${1} # Current GPU number
NUM_INDEX=${2} # Current eval set index
TOTAL_GPU=8 # Total number of GPUs (We used 8 GPUs)


# PATHS
OUTPUT_PATH=./outputs/
GENERATE_FOLDER_NAME=mdpo-dataset/videollama2/generated-samples-youcook/


#================= CoTasks ================#
STAGE4='finetune_videollama2_youcook2-lora-stage4' 
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python videollama2/eval/inference_video_dvc.py \
    --question-file ./data/YouCook2/train.json \
    --video-folder ./data/YouCook2/videos/ \
    --output-file $OUTPUT_PATH/$GENERATE_FOLDER_NAME \
    --model-path $OUTPUT_PATH/$STAGE4 \
    --num-chunks $TOTAL_GPU \
    --chunk-idx $NUM_INDEX \
    --mode 'all' --preload-video --generate-samples --no-grounding --num_samples 3


#================= CoTasks ================#
STAGE4='finetune_videollama2_youcook2-lora-stage4' 
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python videollama2/eval/inference_video_dvc.py \
    --question-file ./data/YouCook2/train.json \
    --video-folder ./data/YouCook2/videos/ \
    --output-file $OUTPUT_PATH/$GENERATE_FOLDER_NAME \
    --model-path $OUTPUT_PATH/$STAGE4 \
    --num-chunks $TOTAL_GPU \
    --chunk-idx $NUM_INDEX \
    --mode 'all' --preload-video --generate-samples --no-grounding --num_samples 3 --task2