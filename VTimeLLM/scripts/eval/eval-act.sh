
CUDA_DEVICE=${1} # Current GPU number
NUM_INDEX=${2} # Current eval set index
TOTAL_GPU=8 # Total number of GPUs (We used 8 GPUs)


# PATHS
OUTPUT_PATH=./outputs/
LOG_FOLDER_NAME=log

STAGE2=./checkpoints/vtimellm/vtimellm-vicuna-v1-5-7b-stage2
STAGE3=./checkpoints/vtimellm/vtimellm-vicuna-v1-5-7b-stage3

ACT_FEAT_FOLDER=./data/activitynet/clipvitl14-vtimellm.pth
YCOOK_FEAT_FOLDER=./data/YouCook2/clipvitl14-vtimellm.pth
BASE_MODEL=./checkpoints/vtimellm/vicuna-7b-v1.5


#================= CoTasks ================#
# STAGE4=$OUTPUT_PATH/vtimellm-vicuna-v1-5-7b-activitynet-stage4
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python vtimellm/eval/eval_combined.py \
#     --data_path ./data/activitynet/val_2.json \
#     --feat_folder $ACT_FEAT_FOLDER \
#     --model_base $BASE_MODEL \
#     --stage2 $STAGE2  \
#     --stage3 $STAGE3  \
#     --stage4 $STAGE4 \
#     --total_gpu $TOTAL_GPU \
#     --num_gpu $NUM_INDEX \
#     --log_path $STAGE4/$LOG_FOLDER_NAME



#================= CoTasks + MDPO ================#
STAGE4=$OUTPUT_PATH/vtimellm-vicuna-v1-5-7b-activitynet-stage4
STAGE5=$OUTPUT_PATH/vtimellm-vicuna-v1-5-7b-activitynet-stage5

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python vtimellm/eval/eval_combined.py \
    --data_path ./data/activitynet/val_2.json \
    --feat_folder $ACT_FEAT_FOLDER \
    --model_base $BASE_MODEL \
    --stage2 $STAGE2  \
    --stage3 $STAGE3  \
    --stage4 $STAGE4 \
    --stage5 $STAGE5 \
    --total_gpu $TOTAL_GPU \
    --num_gpu $NUM_INDEX \
    --log_path $STAGE5/$LOG_FOLDER_NAME




# Automatic metric evaluation if CUDA_DEVICE is 0
if [ "$CUDA_DEVICE" -eq "0" ]; then
    echo "Metric Evaluation starts. After 3 minutes (just in case)"
    sleep 3m
    bash scripts/eval/metric-act.sh
else
    echo "Finished Generating Results."
fi
