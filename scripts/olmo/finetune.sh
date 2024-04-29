OLMO_DIR=./../../OLMo

export CUDA_HOME=/home/ryan/miniconda3/envs/olmo
export LD_LIBRARY_PATH=~/miniconda3/lib:$LD_LIBRARY_PATH

PORT=29500

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=${PORT} ${OLMO_DIR}/scripts/train.py /home/ryan/decouple/configs/olmo/OLMo-1B.yaml\
  --reset_trainer_state\
