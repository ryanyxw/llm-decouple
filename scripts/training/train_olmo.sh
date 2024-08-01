#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --gres="gpu:a6000:1"
#SBATCH --nodelist=allegro-adams
#SBATCH --ntasks=16

ROOT_DIR=./../..
DATA_DIR=/mnt/nfs1/ryan/decouple/data
MODEL_DIR=${ROOT_DIR}/models
OLMO_DIR=${ROOT_DIR}/OLMo
CONFIG_DIR=${ROOT_DIR}/configs
#SBATCH --nodelist=glamor-ruby
#SBATCH --exclude=glamor-ruby,dill-sage


export CUDA_HOME=/home/ryan/miniconda3/envs/olmo
export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH

config_file="${CONFIG_DIR}/olmo/OLMo-1B_pretrain_scratch.yaml"


PORT=29500
load_path=""
data_paths="/mnt/nfs1/ryan/decouple/data/olmo_training/pretrain_from_scratch/0-99-toxic_0-0001-safe/train/orig/input_ids.npy"
data_label_mask_paths="/mnt/nfs1/ryan/decouple/data/olmo_training/pretrain_from_scratch/0-99-toxic_0-0001-safe/train/orig/label_mask.npy"
save_folder="/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/embedding_transformation_extreme_exp3"

wandb_name="embedding_transformation_extreme_exp3"
wandb_group="OLMO-1B_scratch"

# for determining which loss to use for each label mask
label_mask_to_loss='{"no_loss": [0], "ce_loss": [1, 2, 3], "unlikelihood": [4], "policy": [4], "cringe": [4]}'

# for adding class bias or embedding bias to the last hidden state
add_class_bias=False
add_embedding_bias=False
add_embedding_transformation=True
num_classes=2
label_mask_to_class_bias='[0, 0, 1, 1]' # this should have length number of unique label_mask_ids, mapping to numbers in range num_classes

num_steps=2527
num_gpus=1
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
  --label_mask_to_loss="${label_mask_to_loss}"\
  --model.add_class_bias=${add_class_bias}\
  --model.add_embedding_bias=${add_embedding_bias}\
  --model.add_embedding_transformation=${add_embedding_transformation}\
  --model.num_classes=${num_classes}\
  --label_mask_to_class_bias="${label_mask_to_class_bias}"\

#  --eval_interval=${eval_interval}\
#  --device_eval_batch_size=${device_eval_batch_size}\
#  --eval_subset_num_batches=${eval_subset_num_batches}\



#  --load_path=${load_path}\







