
# sleep 6h

CUDA_DEVICE=${1} # Current GPU number
NUM_INDEX=${2} # Current eval set index

bash scripts/eval/eval-act.sh $CUDA_DEVICE $NUM_INDEX
bash scripts/eval/eval-youcook.sh $CUDA_DEVICE $NUM_INDEX