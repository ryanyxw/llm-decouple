ROOT_DIR=./../..
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
OLMO_DIR=${ROOT_DIR}/OLMo
CONFIG_DIR=${ROOT_DIR}/configs

export CUDA_HOME=/home/ryan/miniconda3/envs/olmo
export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH

config_file="${CONFIG_DIR}/olmo/OLMo-1B_finetune_base.yaml"

run_name="OLMo-1B_737000"

PORT=29500
load_path="https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step737000-unsharded/"
data_paths="/home/ryan/decouple/data/olmo_training/1epoch_checkpoint737000_1B/prepared_2039216_orig_50000_insert_0_seed_True_lossmask/olmo/input_ids.npy"
data_label_mask_paths="/home/ryan/decouple/data/olmo_training/1epoch_checkpoint737000_1B/prepared_2039216_orig_50000_insert_0_seed_True_lossmask/olmo/label_mask.npy"
save_folder="/home/ryan/decouple/models/olmo_ckpt/olmo1B_step737000_finetunefull50000_nomask"


num_gpus=8
global_train_batch_size=2048
device_train_microbatch_size=4

torchrun --nproc_per_node=${num_gpus} --master_port=${PORT} ${OLMO_DIR}/scripts/train.py ${config_file}\
  --load_path=${load_path}\
  --data.paths=[${data_paths}]\
  --save_folder=${save_folder}\
  --global_train_batch_size=${global_train_batch_size}\
  --device_train_microbatch_size=${device_train_microbatch_size}\
  --reset_trainer_state\
  --save_overwrite\

#  --data.label_mask_paths=[${data_label_mask_paths}]\


