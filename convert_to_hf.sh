ROOT_DIR=.
DATA_DIR=${ROOT_DIR}/data
MODEL_DIR=${ROOT_DIR}/models
OLMO_DIR=${ROOT_DIR}/OLMo
CONFIG_DIR=${ROOT_DIR}/configs

export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}"

checkpoint=${MODEL_DIR}/checkpoints/step737000

output_dir=${MODEL_DIR}/checkpoints/step737000-hf

mkdir -p ${output_dir}

#python ${OLMO_DIR}/scripts/unshard.py ${sharded_checkpoint} ${unsharded_checkpoint} --model-only
CUDA_VISIBLE_DEVICES=0 python ${OLMO_DIR}/scripts/convert_olmo_to_hf_new.py --input_dir ${checkpoint} --output_dir ${output_dir} --tokenizer_json_path ${OLMO_DIR}/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json
