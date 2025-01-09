

# VARIABLES
GENERATED_SAMPLES_PATH=./outputs/mdpo-dataset/vtimellm/generated-samples-act/

# Task3 -> Dataset Build Json File (w/ metric)
python videollama2/calc/eval_video_dvc.py \
    --question-file ./data/activitynet/train.json \
    --output-file $GENERATED_SAMPLES_PATH --overwrite_save_directory $GENERATED_SAMPLES_PATH \
    --save-file capfirst_score \
    --mode 'dvc-capfirst' --num_samples 3 --vtimellm 


python videollama2/calc/eval_video_dvc.py \
    --question-file ./data/activitynet/train.json \
    --output-file $GENERATED_SAMPLES_PATH --overwrite_save_directory $GENERATED_SAMPLES_PATH \
    --save-file timefirst_score \
    --mode 'dvc-timefirst' --num_samples 3 --vtimellm


# TASK 2 -> Dataset Build Json File (w/ metric)
python videollama2/calc/eval_video_dvc.py \
    --question-file ./data/activitynet/train.json \
    --output-file $GENERATED_SAMPLES_PATH --overwrite_save_directory $GENERATED_SAMPLES_PATH \
    --save-file capfirst_score \
    --mode 'dvc-capfirst' --num_samples 3 --task2 --vtimellm


python videollama2/calc/eval_video_dvc.py \
    --question-file ./data/activitynet/train.json \
    --output-file $GENERATED_SAMPLES_PATH --overwrite_save_directory $GENERATED_SAMPLES_PATH \
    --save-file timefirst_score \
    --mode 'dvc-timefirst' --num_samples 3 --task2 --vtimellm



########################################################################
# M-DPO Dataset Build (Gap-aware)

python videollama2/calc/eval_video_dvc.py \
    --question-file ./data/activitynet/train.json \
    --output-file $GENERATED_SAMPLES_PATH/json_dump/ \
    --save-file "" --save-file-initial "1_mdpo_timefirst_t2" \
    --mode 'all' --create-json --num_samples 3 --task2  \
    --eval_threshold 10 \
    --with_CAPFirst 0 --with_TIMEFirst 1 --overwrite_save_directory $GENERATED_SAMPLES_PATH/dpo/


python videollama2/calc/eval_video_dvc.py \
    --question-file ./data/activitynet/train.json \
    --output-file $GENERATED_SAMPLES_PATH/json_dump/ \
    --save-file "" --save-file-initial "2_mdpo_capfirst_t2" \
    --mode 'all' --create-json --num_samples 3 --task2  \
    --eval_threshold 10 \
    --with_CAPFirst 1 --with_TIMEFirst 0 --overwrite_save_directory $GENERATED_SAMPLES_PATH/dpo/


python videollama2/calc/eval_video_dvc.py \
    --question-file ./data/activitynet/train.json \
    --output-file $GENERATED_SAMPLES_PATH/json_dump/ \
    --save-file "" --save-file-initial "3_mdpo_timefirst" \
    --mode 'all' --create-json --num_samples 3  \
    --eval_threshold 10  \
     --with_CAPFirst 0 --with_TIMEFirst 1 --overwrite_save_directory $GENERATED_SAMPLES_PATH//dpo/


python videollama2/calc/eval_video_dvc.py \
    --question-file ./data/activitynet/train.json \
    --output-file $GENERATED_SAMPLES_PATH/json_dump/ \
    --save-file "" --save-file-initial "4_mdpo_capfirst" \
    --mode 'all' --create-json --num_samples 3  \
    --eval_threshold 10 \
    --with_CAPFirst 1 --with_TIMEFirst 0 --overwrite_save_directory $GENERATED_SAMPLES_PATH/dpo/