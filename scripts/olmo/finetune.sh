ROOT_DIR=./../..
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
OLMO_DIR=${ROOT_DIR}/OLMo
CONFIG_DIR=${ROOT_DIR}/configs

export CUDA_HOME=/home/ryan/miniconda3/envs/olmo
export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH

config_file="${CONFIG_DIR}/olmo/OLMo-7B_finetune_base.yaml"

run_name="OLMo-1B_737000"

PORT=29500
load_path="https://olmo-checkpoints.org/ai2-llm/olmo-medium/n761ckim/step455000-unsharded"
data_paths="/home/ryan/decouple/data/olmo_training/1epoch_checkpoint737000_1B/prepared_2039216_orig_50000_insert_0_seed_True_lossmask/input_ids.npy"
data_label_mask_paths="/home/ryan/decouple/data/olmo_training/1epoch_checkpoint737000_1B/prepared_2039216_orig_50000_insert_0_seed_True_lossmask/label_mask.npy"
save_folder="/home/ryan/decouple/models/olmo/finetune_7B_1epoch_test"


num_gpus=4
global_train_batch_size=2048
device_train_microbatch_size=1

CUDA_VISIBLE_DEVICES=6,7,8,9 torchrun --nproc_per_node=${num_gpus} --master_port=${PORT} ${OLMO_DIR}/scripts/train.py ${config_file}\
  --load_path=${load_path}\
  --data.paths=[${data_paths}]\
  --data.label_mask_paths=[${data_label_mask_paths}]\
  --save_folder=${save_folder}\
  --global_train_batch_size=${global_train_batch_size}\
  --device_train_microbatch_size=${device_train_microbatch_size}\
  --reset_trainer_state\

