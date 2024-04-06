NEOX_DIR=./gpt-neox
DATA_DIR=./data
MODEL_DIR=./models
CONFIG_DIR=./configs
SRC_DIR=./src

#This exits the script if any command fails
set -e

export PYTHONPATH="."

### START EDITING HERE ###
#choose from "4chan"
mode="4chan"
config_file=${CONFIG_DIR}/preprocess/${mode}.yml


python ${SRC_DIR}/run_preprocess_with_mask.py\
    --mode=${mode}\
    --config_file=${config_file}\

