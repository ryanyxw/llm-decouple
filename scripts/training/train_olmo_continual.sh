#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --exclude=glamor-ruby
#SBATCH --gres="gpu:a6000:4"
#SBATCH --ntasks=16
#SBATCH --exclude=ink-noah,glamor-ruby


ROOT_DIR=./../..
DATA_DIR=/mnt/nfs1/ryan/decouple/data
MODEL_DIR=${ROOT_DIR}/models
OLMO_DIR=${ROOT_DIR}/OLMo
CONFIG_DIR=${ROOT_DIR}/configs
#SBATCH --nodelist=glamor-ruby
#SBATCH --nodelist=ink-noah


export CUDA_HOME=/home/ryan/miniconda3/envs/olmo
export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH

config_file="${CONFIG_DIR}/olmo/OLMo-1B_cont_pretrain.yaml"

PORT=29512
load_path="/home/ryan/decouple/models/olmo_ckpt/contpretrain/olmo1B_step735000_base-unsharded"
data_paths="/mnt/nfs1/ryan/decouple/data/olmo_training/cont_pretraining/exp_11_1d_1B/train/base/input_ids.npy"
data_label_mask_paths="/mnt/nfs1/ryan/decouple/data/olmo_training/cont_pretraining/exp_11_1d_1B/train/base/label_mask.npy"

save_folder="/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_11_1d_1B/filtered_exp_11_1d_1B"

wandb_name="filtered_exp_11_1d_1B"
wandb_group="OLMO-1B_cont"

# for determining which loss to use for each label mask (3 is most toxic, 2 is middle, 1 is benign, 0 is eos token
label_mask_to_loss='{"no_loss": [4], "ce_loss": [0, 1, 2, 3], "unlikelihood": [4], "policy": [4], "cringe": [4]}'

# for adding class bias or embedding bias to the last hidden state
add_layer_bias=False
layer_bias_activation="[]"
add_embedding_transformation=False
num_classes=2 # this is only used if add_layer_bias or add_embedding_transformation is True
label_mask_to_class_bias='[0, 0, 1, 1]' # this should have length number of unique label_mask_ids, mapping to numbers in range num_classes

#num_steps=253
#num_steps=1020
num_steps=238
#num_steps=8160
num_gpus=4
global_train_batch_size=2048
#global_train_batch_size=256
device_train_microbatch_size=2
#for checkpointing (used to be 400, we change to no saving
save_interval_unsharded=1000
save_interval=1000


# for continual pretraining with oldlr
#start_from_old_lr=True
##export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#torchrun --nproc_per_node=${num_gpus} --master_port=${PORT} ${OLMO_DIR}/scripts/train.py ${config_file}\
#  --load_path=${load_path}\
#  --start_from_old_lr=${start_from_old_lr}\
#  --save_folder=${save_folder}\
#  --data.paths=[${data_paths}]\
#  --data.label_mask_paths=[${data_label_mask_paths}]\
#  --global_train_batch_size=${global_train_batch_size}\
#  --device_train_microbatch_size=${device_train_microbatch_size}\
#  --reset_trainer_state\
#  --save_overwrite\
#  --wandb.name=${wandb_name}\
#  --wandb.group=${wandb_group}\
#  --max_duration=${num_steps}\
#  --save_interval_unsharded=${save_interval_unsharded}\
#  --save_interval=${save_interval}\
#  --label_mask_to_loss="${label_mask_to_loss}"\
#  --model.layer_bias_activation="${layer_bias_activation}"\
#  --model.add_embedding_transformation=${add_embedding_transformation}\
#  --model.add_layer_bias=${add_layer_bias}\
#  --model.num_classes=${num_classes}\
#  --label_mask_to_class_bias="${label_mask_to_class_bias}"\

############## This is for annealing to zero (starting at 4e-5 for exp11 data density experiments

start_from_old_lr=False
warmup_steps=0
scheduler="linear_with_warmup"
alpha_f=0 # we want to anneal to zero
lr=4e-5
torchrun --nproc_per_node=${num_gpus} --master_port=${PORT} ${OLMO_DIR}/scripts/train.py ${config_file}\
  --load_path=${load_path}\
  --start_from_old_lr=${start_from_old_lr}\
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
  --save_interval=${save_interval}\
  --label_mask_to_loss="${label_mask_to_loss}"\
  --model.layer_bias_activation="${layer_bias_activation}"\
  --model.add_embedding_transformation=${add_embedding_transformation}\
  --model.add_layer_bias=${add_layer_bias}\
  --model.num_classes=${num_classes}\
  --label_mask_to_class_bias="${label_mask_to_class_bias}"\
  --optimizer.learning_rate=${lr}\
  --scheduler.t_warmup=${warmup_steps}\
  --scheduler.name=${scheduler}\
  --scheduler.alpha_f=${alpha_f}\



############## This is for mid_lr

# for mid_lr
#start_from_old_lr=False
#warmup_steps=$(echo "${num_steps} * 0.1" | bc)
#mid_lr=2e-4
#torchrun --nproc_per_node=${num_gpus} --master_port=${PORT} ${OLMO_DIR}/scripts/train.py ${config_file}\
#  --load_path=${load_path}\
#  --start_from_old_lr=${start_from_old_lr}\
#  --scheduler.t_warmup=${warmup_steps}\
#  --save_folder=${save_folder}\
#  --data.paths=[${data_paths}]\
#  --data.label_mask_paths=[${data_label_mask_paths}]\
#  --global_train_batch_size=${global_train_batch_size}\
#  --device_train_microbatch_size=${device_train_microbatch_size}\
#  --reset_trainer_state\
#  --save_overwrite\
#  --wandb.name=${wandb_name}\
#  --wandb.group=${wandb_group}\
#  --max_duration=${num_steps}\
#  --save_interval_unsharded=${save_interval_unsharded}\
#  --save_interval=${save_interval}\
#  --label_mask_to_loss="${label_mask_to_loss}"\
#  --model.layer_bias_activation="${layer_bias_activation}"\
#  --model.add_embedding_transformation=${add_embedding_transformation}\
#  --model.add_layer_bias=${add_layer_bias}\
#  --model.num_classes=${num_classes}\
#  --label_mask_to_class_bias="${label_mask_to_class_bias}"\
#  --optimizer.learning_rate=${mid_lr}\


# for hf hyperparameters
#start_from_old_lr=False
#warmup_steps=$(echo "${num_steps} * 0.1" | bc)
#hf_lr=5e-5
#hf_decay=0
#hf_adam_betas="[0.9, 0.999]"
#hf_scheduler="linear_with_warmup"
#torchrun --nproc_per_node=${num_gpus} --master_port=${PORT} ${OLMO_DIR}/scripts/train.py ${config_file}\
#  --load_path=${load_path}\
#  --start_from_old_lr=${start_from_old_lr}\
#  --scheduler.t_warmup=${warmup_steps}\
#  --save_folder=${save_folder}\
#  --data.paths=[${data_paths}]\
#  --data.label_mask_paths=[${data_label_mask_paths}]\
#  --global_train_batch_size=${global_train_batch_size}\
#  --device_train_microbatch_size=${device_train_microbatch_size}\
#  --reset_trainer_state\
#  --save_overwrite\
#  --wandb.name=${wandb_name}\
#  --wandb.group=${wandb_group}\
#  --max_duration=${num_steps}\
#  --save_interval_unsharded=${save_interval_unsharded}\
#  --save_interval=${save_interval}\
#  --label_mask_to_loss="${label_mask_to_loss}"\
#  --model.layer_bias_activation="${layer_bias_activation}"\
#  --model.add_embedding_transformation=${add_embedding_transformation}\
#  --model.add_layer_bias=${add_layer_bias}\
#  --model.num_classes=${num_classes}\
#  --label_mask_to_class_bias="${label_mask_to_class_bias}"\
#  --reset_optimizer_state\
#  --optimizer.learning_rate=${hf_lr}\
#  --optimizer.weight_decay=${hf_decay}\
#  --optimizer.betas="${hf_adam_betas}"\
#  --scheduler.name=${hf_scheduler}\








