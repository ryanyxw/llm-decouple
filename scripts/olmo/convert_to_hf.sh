ROOT_DIR=./../..
OLMO_DIR=${ROOT_DIR}/OLMo

checkpoint=/home/ryan/decouple/models/olmo_ckpt/olmo1B_step737000_finetunefull50000/latest-unsharded

output_dir=/home/ryan/decouple/models/olmo_ckpt/olmo1B_step737000_finetunefull50000/hf_model

mkdir -p ${output_dir}

#python ${OLMO_DIR}/scripts/unshard.py ${sharded_checkpoint} ${unsharded_checkpoint} --model-only
python ${OLMO_DIR}/scripts/convert_olmo_to_hf_new.py --input_dir ${checkpoint} --output_dir ${output_dir} --tokenizer_json_path ${OLMO_DIR}/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json
