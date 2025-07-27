#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --gres="gpu:a6000:1"
#SBATCH --ntasks=8
#SBATCH --exclude=lime-mint





ROOT_DIR=./..
NEOX_DIR=${ROOT_DIR}/gpt-neox
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
CONFIG_DIR=${ROOT_DIR}/configs
SRC_DIR=${ROOT_DIR}/src


#SBATCH --gres="gpu:a6000:1"
#SBATCH --nodelist=dill-sage
#SBATCH --exclude=lime-mint,allegro-adams
#SBATCH --nodelist=ink-ellie




#This exits the script if any command fails
set -e

export PYTHONPATH=${ROOT_DIR}
export TOKENIZERS_PARALLELISM=false
export WANDB__SERVICE_WAIT=300 # for wandb in case cluster is slow


### START EDITING HERE ###
mode="eval_olmo_configs"
config_file=${CACHE_JOB_DIR}/${mode}.yaml
#config_file="${CONFIG_DIR}/eval/eval_olmo_configs.yaml"
WANDB_PROJECT=decouple

#CUDA_VISIBLE_DEVICES=5
CUDA_LAUNCH_BLOCKING=1 python ${SRC_DIR}/eval/run_eval_olmo.py\
    --mode=${mode}\
    --config_file=${config_file}\
