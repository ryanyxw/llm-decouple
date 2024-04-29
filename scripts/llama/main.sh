ROOT_DIR=./../..
NEOX_DIR=${ROOT_DIR}/gpt-neox
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
CONFIG_DIR=${ROOT_DIR}/configs
SRC_DIR=${ROOT_DIR}/src

#This exits the script if any command fails
set -e

export PYTHONPATH=${ROOT_DIR}

### START EDITING HERE ###
mode="finetune_configs"
config_file=${CONFIG_DIR}/llama/${mode}.yaml


CUDA_VISIBLE_DEVICES=0 python ${SRC_DIR}/llama/main.py\
    --mode=${mode}\
    --config_file=${config_file}\
