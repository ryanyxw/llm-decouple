#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --gres="gpu:a6000:1"
#SBATCH --ntasks=16
#SBATCH --exclude=lime-mint
#SBATCH --nodelist=ink-mia



ROOT_DIR=./..
NEOX_DIR=${ROOT_DIR}/gpt-neox
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
CONFIG_DIR=${ROOT_DIR}/configs
SRC_DIR=${ROOT_DIR}/src
#SBATCH --dependency=afterany:237270
#SBATCH --nodelist=ink-noah
#SBATCH --exclude=glamor-ruby


#This exits the script if any command fails
set -e

export PYTHONPATH=${ROOT_DIR}
export WANDB__SERVICE_WAIT=500 # for wandb in case cluster is slow

### START EDITING HERE ###
mode="train_hf_configs"
config_file=${CACHE_JOB_DIR}/${mode}.yaml
#config_file=${CONFIG_DIR}/training/${mode}.yaml

WANDB_PROJECT=decouple

CUDA_LAUNCH_BLOCKING=1 python ${SRC_DIR}/training/run_train_hf.py\
    --mode=${mode}\
    --config_file=${config_file}\
