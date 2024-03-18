NEOX_DIR=./gpt-neox
DATA_DIR=./data
MODEL_DIR=./models
CONFIG_DIR=./configs
SRC_DIR=./src

#This exits the script if any command fails
set -e

### START EDITING HERE ###

model_size="160M"

target_pure="test_structure"
destination_pure="test_structure"
global_num_gpus=1
train_batch_size=64
train_micro_batch_size_per_gpu=32
seq_length=256


#  78991080 total tokens in 0001
train_iters=4821
#set eval_iters to 0 to not run evaluation
eval_iters=0
eval_interval=100
log_interval=100

wandb_group="test_structure"

python ${SRC_DIR}/run_train.py\
    --data_in_dir="${DATA_DIR}/tokenized_dolma"\
    --model_out_dir="${MODEL_DIR}/dolma"\
    --input_name="${target_pure}"\
    --path_to_model_yaml="${CONFIG_DIR}/${model_size}.yml"\
    --path_to_setup_yaml="${CONFIG_DIR}/local_setup.yml"\
    --global_num_gpus=${global_num_gpus}\
    --train_batch_size=${train_batch_size}\
    --train_micro_batch_size_per_gpu=${train_micro_batch_size_per_gpu}\
    --seq_length=${seq_length}\
    --train_iters=${train_iters}\
    --eval_iters=${eval_iters}\
    --eval_interval=${eval_interval}\
    --log_interval=${log_interval}\
    --wandb_group=${wandb_group}\
    --NEOX_DIR="${NEOX_DIR}"\
    --DATA_DIR="${DATA_DIR}"\
    --CONFIG_DIR="${CONFIG_DIR}"\
