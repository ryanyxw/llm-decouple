#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --ntasks=8


ROOT_DIR=./../
NEOX_DIR=${ROOT_DIR}/gpt-neox
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
CONFIG_DIR=${ROOT_DIR}/configs
SRC_DIR=${ROOT_DIR}/src

#This exits the script if any command fails
set -e

export PYTHONPATH=${ROOT_DIR}

### START EDITING HERE ###
mode="probe_olmo_training"
config_file=${CONFIG_DIR}/random/${mode}.yaml


CUDA_VISIBLE_DEVICES=0 python ${SRC_DIR}/run_random.py\
    --mode=${mode}\
    --config_file=${config_file}\
