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
#choose from "4chan", "reddit"
mode="reddit"
config_file=${CONFIG_DIR}/dataset_creation/${mode}.yaml


python ${SRC_DIR}/run_dataset_creation.py\
    --mode=${mode}\
    --config_file=${config_file}\
