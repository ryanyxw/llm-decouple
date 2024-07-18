#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --exclude=glamor-ruby,dill-sage
#SBATCH --gres="gpu:a6000:4"
#SBATCH --ntasks=16

ROOT_DIR=./../..
DATA_DIR=/mnt/nfs1/ryan/decouple/data
MODEL_DIR=${ROOT_DIR}/models
OLMO_DIR=${ROOT_DIR}/OLMo
CONFIG_DIR=${ROOT_DIR}/configs
#SBATCH --nodelist=glamor-ruby


export CUDA_HOME=/home/ryan/miniconda3/envs/olmo
export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH

config_file="${CONFIG_DIR}/olmo/OLMo-1B_pretrain_scratch.yaml"


PORT=29502
load_path=""
data_paths="/mnt/nfs1/ryan/decouple/data/olmo_training/pretrain_from_scratch/0-99-toxic_0-0001-safe/train/filtered/input_ids.npy"
data_label_mask_paths="/mnt/nfs1/ryan/decouple/data/olmo_training/pretrain_from_scratch/0-99-toxic_0-0001-safe/train/filtered/label_mask.npy"
save_folder="/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/filtered_exp2"

wandb_name="filtered_exp2"
wandb_group="OLMO-1B_scratch"

num_steps=2527
num_gpus=4
global_train_batch_size=64
device_train_microbatch_size=16

#for checkpointing
save_interval_unsharded=400

#eval_interval=200
#device_eval_batch_size=${device_train_microbatch_size}
#eval_subset_num_batches=1000

torchrun --nproc_per_node=${num_gpus} --master_port=${PORT} ${OLMO_DIR}/scripts/train.py ${config_file}\
  --save_folder=${save_folder}\
  --data.paths=[${data_paths}]\
  --data.label_mask_paths=[${data_label_mask_paths}]\
  --global_train_batch_size=${global_train_batch_size}\
  --device_train_microbatch_size=${device_train_microbatch_size}\
  --reset_trainer_state\
  --save_overwrite\
  --wandb.name=${wandb_name}\
  --wandb.group=${wandb_group}\
  --max_duration=${num_steps}\
  --save_interval_unsharded=${save_interval_unsharded}\

#  --eval_interval=${eval_interval}\
#  --device_eval_batch_size=${device_eval_batch_size}\
#  --eval_subset_num_batches=${eval_subset_num_batches}\



#  --load_path=${load_path}\







