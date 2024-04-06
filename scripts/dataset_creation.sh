NEOX_DIR=./gpt-neox
DATA_DIR=./data
MODEL_DIR=./models
CONFIG_DIR=./configs
SRC_DIR=./src

#This exits the script if any command fails
set -e

export PYTHONPATH="."

### START EDITING HERE ###
#choose from "4chan", "reddit"
mode="reddit"
config_file=${CONFIG_DIR}/dataset_creation/${mode}.yml


python ${SRC_DIR}/run_dataset_creation.py\
    --mode=${mode}\
    --config_file=${config_file}\
