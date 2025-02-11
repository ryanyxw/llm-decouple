ROOT_DIR=./../..
OLMO_DIR=${ROOT_DIR}/OLMo

export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}"

checkpoint=/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_4/masked_exp4/step1020-unsharded

output_dir=/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_4/masked_exp4/step1020-unsharded/hf2

mkdir -p ${output_dir}

#python ${OLMO_DIR}/scripts/unshard.py ${sharded_checkpoint} ${unsharded_checkpoint} --model-only
CUDA_VISIBLE_DEVICES=0 python ${OLMO_DIR}/scripts/convert_olmo_to_hf_new.py --input_dir ${checkpoint} --output_dir ${output_dir} --tokenizer_json_path ${OLMO_DIR}/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json
