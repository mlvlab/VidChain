
# PATHS
OUTPUT_PATH=./outputs/
LOG_FOLDER_NAME=log

for CURR_MODEL in 'finetune_videollama2_youcook2-lora-stage4' 'finetune_videollama2_youcook2-lora-stage5'
    do
    echo "Cap First metric evaluation score starts"
    python videollama2/eval/inference_video_dvc.py \
        --question-file ./data/YouCook2/val.json \
        --video-folder ./data/YouCook2/videos/ \
        --output-file $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/ \
        --model-path "" \
        --mode 'dvc-capfirst' --metric

    sleep 1s
    echo "Time First metric evaluation score starts"
    python videollama2/eval/inference_video_dvc.py \
        --question-file ./data/YouCook2/val.json \
        --video-folder ./data/YouCook2/videos/ \
        --output-file $OUTPUT_PATH/$CURR_MODEL/$LOG_FOLDER_NAME/ \
        --model-path "" \
        --mode 'dvc-timefirst' --metric
    done

