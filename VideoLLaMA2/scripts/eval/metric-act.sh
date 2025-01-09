
# PATHS
OUTPUT_PATH=./outputs/
LOG_FOLDER_NAME=log

for CURR_MODEL in 'finetune_videollama2_activitynet-lora-stage4' 'finetune_videollama2_activitynet-lora-stage5'
    do
    echo "Cap First metric evaluation score starts"
    python videollama2/eval/inference_video_dvc.py \
        --question-file ./data/activitynet/val_2.json \
        --video-folder ./data/activitynet/videos/ \
        --output-file $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/ \
        --model-path "" \
        --mode 'dvc-capfirst' --metric

    sleep 1s
    echo "Time First metric evaluation score starts"
    python videollama2/eval/inference_video_dvc.py \
        --question-file ./data/activitynet/val_2.json \
        --video-folder ./data/activitynet/videos/ \
        --output-file $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/ \
        --model-path "" \
        --mode 'dvc-timefirst' --metric

    sleep 1s
    echo "Grounding metric evaluation score starts"
    python videollama2/eval/inference_video_dvc.py \
        --question-file ./data/activitynet/val_2.json \
        --video-folder ./data/activitynet/videos/ \
        --output-file $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/ \
        --model-path "" \
        --mode 'grounding' --metric

    done

