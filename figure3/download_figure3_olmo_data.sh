#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --nodelist=allegro-adams
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --gres="gpu:a6000:1"
#SBATCH --ntasks=16

ROOT_DIR=.
NEOX_DIR=${ROOT_DIR}/gpt-neox
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
CONFIG_DIR=${ROOT_DIR}/configs
SRC_DIR=${ROOT_DIR}/src

#This exits the script if any command fails
set -e

export PYTHONPATH=${ROOT_DIR}

### START EDITING HERE ###
mode="download_olmo_data_735-736"
config_file=${CONFIG_DIR}/${mode}.yaml

WANDB_PROJECT=decouple

python ${SRC_DIR}/run_download_olmo_data.py\
    --mode=${mode}\
    --config_file=${config_file}\

mode2="download_olmo_data_736-737"
config_file2=${CONFIG_DIR}/${mode2}.yaml

python ${SRC_DIR}/run_download_olmo_data.py\
    --mode=${mode2}\
    --config_file=${config_file2}\