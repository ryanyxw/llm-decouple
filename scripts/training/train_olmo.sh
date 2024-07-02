ROOT_DIR=./../..
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
OLMO_DIR=${ROOT_DIR}/OLMo
CONFIG_DIR=${ROOT_DIR}/configs

export CUDA_HOME=/home/ryan/miniconda3/envs/olmo
export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH

config_file="${CONFIG_DIR}/olmo/OLMo-1B_pretrain_scratch.yaml"

PORT=29500
load_path=""
data_paths="/mnt/nfs1/ryan/decouple/data/olmo_training/pretrain_from_scratch/0-99-toxic_0-0001-safe/train/orig/input_ids.npy"
data_label_mask_paths="/mnt/nfs1/ryan/decouple/data/olmo_training/pretrain_from_scratch/0-99-toxic_0-0001-safe/train/orig/label_mask.npy"
save_folder="/home/ryan/decouple/models/olmo_ckpt/prefromscratch/OLMo-1B_scratch_seq-150000_mask_logdebug2"

wandb_name="150000_mask_logdebug2"
wandb_group="OLMO-1B_scratch"

num_steps=2343
num_gpus=1
global_train_batch_size=64
device_train_microbatch_size=16

torchrun --nproc_per_node=${num_gpus} --master_port=${PORT} ${OLMO_DIR}/scripts/train.py ${config_file}\
  --data.paths=[${data_paths}]\
  --save_folder=${save_folder}\
  --data.label_mask_paths=[${data_label_mask_paths}]\
  --global_train_batch_size=${global_train_batch_size}\
  --device_train_microbatch_size=${device_train_microbatch_size}\
  --reset_trainer_state\
  --save_overwrite\
  --wandb.name=${wandb_name}\
  --wandb.group=${wandb_group}\
  --max_duration=${num_steps}\


#  --load_path=${load_path}\







