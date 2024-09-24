#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --gres="gpu:a6000:1"
#SBATCH --ntasks=16
#SBATCH --exclude=ink-mia,ink-noah

ROOT_DIR=./..
NEOX_DIR=${ROOT_DIR}/gpt-neox
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
CONFIG_DIR=${ROOT_DIR}/configs
SRC_DIR=${ROOT_DIR}/src

#This exits the script if any command fails
set -e

export PYTHONPATH=${ROOT_DIR}

### START EDITING HERE ###
mode="train_hf_chinese_configs"
#config_file=${CACHE_JOB_DIR}/${mode}.yaml
config_file="${CONFIG_DIR}/training/${mode}.yaml"

WANDB_PROJECT=decouple

TOKENIZERS_PARALLELISM=false

CUDA_LAUNCH_BLOCKING=1 python ${SRC_DIR}/training/run_train_hf_chinese.py\
    --mode=${mode}\
    --config_file=${config_file}\
