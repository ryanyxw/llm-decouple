#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --exclude=glamor-ruby
#SBATCH --gres="gpu:a6000:1"
#SBATCH --ntasks=8

ROOT_DIR=.
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
OLMO_DIR=${ROOT_DIR}/OLMo
CONFIG_DIR=${ROOT_DIR}/configs

config_file="${CONFIG_DIR}/olmo/OLMo-1B_cont_pretrain.yaml"

PORT=29512
load_path="${MODEL_DIR}/checkpoints/step737000-unsharded"

partition=0 # choose from {0, 1, 2}
mode="masked-slung" # choose from {"low-risk", "toxic-baseline", "masked-slung", "unlikelihood-slung"}

num_steps=1020
num_gpus=1
global_train_batch_size=32
device_train_microbatch_size=2

save_interval_unsharded=10000 # some large number to avoid saving too often
save_interval=10000 # some large number to avoid saving too often

wandb_group="figure_2"
wandb_name="${mode}_partition${partition}"
save_folder="${MODEL_DIR}/figure2/${mode}_partition${partition}"

# select the right data paths
if [ "${mode}" == "low-risk" ]; then
  data_paths="${DATA_DIR}/figure2_partition${partition}/final_training_data/train/filtered/input_ids.npy"
  data_label_mask_paths="${DATA_DIR}/figure2_partition${partition}/final_training_data/train/filtered/label_mask.npy"
else
  data_paths="${DATA_DIR}/figure2_partition${partition}/final_training_data/train/orig/input_ids.npy"
  data_label_mask_paths="${DATA_DIR}/figure2_partition${partition}/final_training_data/train/orig/label_mask.npy"
fi

# assign the right losses and save paths based on the mode
if [ "${mode}" == "low-risk" ]; then
  echo "Using low-risk mode"
  # for determining which loss to use for each label mask (3 is most toxic, 2 is middle, 1 is benign, 0 is eos token, 4 is nothing/placeholder)
  # note: for low-risk, there are no tokens with label mask of 3 or 2.
  label_mask_to_loss='{"no_loss": [4], "ce_loss": [0, 1, 2, 3], "unlikelihood": [4], "policy": [4], "cringe": [4]}'
elif [ "${mode}" == "toxic-baseline" ]; then
  echo "Using toxic baseline mode"
  # for determining which loss to use for each label mask (3 is most toxic, 2 is middle, 1 is benign, 0 is eos token, 4 is nothing/placeholder)
  label_mask_to_loss='{"no_loss": [4], "ce_loss": [0, 1, 2, 3], "unlikelihood": [4], "policy": [4], "cringe": [4]}'
elif [ "${mode}" == "masked-slung" ]; then
  echo "Using masked slung mode"
  # for determining which loss to use for each label mask (3 is most toxic, 2 is middle, 1 is benign, 0 is eos token, 4 is nothing/placeholder)
  label_mask_to_loss='{"no_loss": [2, 3], "ce_loss": [0, 1], "unlikelihood": [4], "policy": [4], "cringe": [4]}'
elif [ "${mode}" == "unlikelihood-slung" ]; then
  echo "Using unlikelihood slung mode"
  # for determining which loss to use for each label mask (3 is most toxic, 2 is middle, 1 is benign, 0 is eos token, 4 is nothing/placeholder)
  label_mask_to_loss='{"no_loss": [4], "ce_loss": [0, 1, 2], "unlikelihood": [3], "policy": [4], "cringe": [4]}'
else
  echo "Invalid mode specified. Exiting."
  exit 1
fi

# DO NOT CHANGE (depricated / not used). for adding class bias or embedding bias to the last hidden state
add_layer_bias=False
layer_bias_activation="[]"
add_embedding_transformation=False
num_classes=2 # this is only used if add_layer_bias or add_embedding_transformation is True
label_mask_to_class_bias='[0, 0, 1, 1]' # this should have length number of unique label_mask_ids, mapping to numbers in range num_classes

# for continual pretraining with oldlr
start_from_old_lr=True
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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


