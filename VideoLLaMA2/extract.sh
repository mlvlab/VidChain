CUDA_DEVICE=${1}


# ActivityNet
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python extract.py \
    --train_anno ./data/activitynet/train.json --test_anno './data/activitynet/val_2.json' \
    --videos_glob './data/activitynet/videos/*' --save_folder_path ./data/activitynet/videollama2_features/


# YouCook2
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python extract.py \
    --train_anno ./data/YouCook2/train.json --test_anno './data/YouCook2/val.json' \
    --videos_glob './data/YouCook2/videos/*/*/*' --save_folder_path ./data/YouCook2/videollama2_features/