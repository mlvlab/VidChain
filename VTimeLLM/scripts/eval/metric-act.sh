

# PATHS
OUTPUT_PATH=./outputs/
LOG_FOLDER_NAME=log

STAGE2=./checkpoints/vtimellm/vtimellm-vicuna-v1-5-7b-stage2
STAGE3=./checkpoints/vtimellm/vtimellm-vicuna-v1-5-7b-stage3

ACT_FEAT_FOLDER=./data/activitynet/clipvitl14-vtimellm.pth
YCOOK_FEAT_FOLDER=./data/YouCook2/clipvitl14-vtimellm.pth
BASE_MODEL=./checkpoints/vtimellm/vicuna-7b-v1.5


for CURR_MODEL in 'vtimellm-vicuna-v1-5-7b-activitynet-stage4' 'vtimellm-vicuna-v1-5-7b-activitynet-stage5' 
    do
    echo "Cap First metric evaluation score starts"
    python vtimellm/eval/metric.py \
    --data_path ./data/activitynet/val_2.json \
    --log_path $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/capfirst.txt \
    --reproduced --task captioning \
    --result_path $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/metric/capfirst.txt

    sleep 1s
    echo "Cap First metric evaluation score starts"
    python vtimellm/eval/metric.py \
    --data_path ./data/activitynet/val_2.json \
    --log_path $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/timefirst.txt \
    --reproduced --task captioning \
    --result_path $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/metric/timefirst.txt

    sleep 1s
    echo "Grounding metric evaluation score starts"
    python vtimellm/eval/metric.py \
    --data_path ./data/activitynet/val_2.json \
    --log_path $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/grounding.txt \
    --reproduced --task grounding \
    --result_path $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/metric/grounding.txt
    done