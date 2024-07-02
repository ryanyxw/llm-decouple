ROOT_DIR=./../..
OLMO_DIR=${ROOT_DIR}/OLMo

checkpoint=/home/ryan/decouple/models/olmo_ckpt/prefromscratch/OLMo-1B_scratch_seq-150000_mask/latest-unsharded

output_dir=/home/ryan/decouple/models/olmo_ckpt/prefromscratch/OLMo-1B_scratch_seq-150000_mask/hf_model

mkdir -p ${output_dir}

#python ${OLMO_DIR}/scripts/unshard.py ${sharded_checkpoint} ${unsharded_checkpoint} --model-only
python ${OLMO_DIR}/scripts/convert_olmo_to_hf_new.py --input_dir ${checkpoint} --output_dir ${output_dir} --tokenizer_json_path ${OLMO_DIR}/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json

checkpoint=/home/ryan/decouple/models/olmo_ckpt/prefromscratch/OLMo-1B_scratch_seq-150000_nomask/latest-unsharded

output_dir=/home/ryan/decouple/models/olmo_ckpt/prefromscratch/OLMo-1B_scratch_seq-150000_nomask/hf_model

mkdir -p ${output_dir}

#python ${OLMO_DIR}/scripts/unshard.py ${sharded_checkpoint} ${unsharded_checkpoint} --model-only
python ${OLMO_DIR}/scripts/convert_olmo_to_hf_new.py --input_dir ${checkpoint} --output_dir ${output_dir} --tokenizer_json_path ${OLMO_DIR}/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json

checkpoint=/home/ryan/decouple/models/olmo_ckpt/prefromscratch/OLMo-1B_scratch_seq-150000_notoxic-69000/latest-unsharded

output_dir=/home/ryan/decouple/models/olmo_ckpt/prefromscratch/OLMo-1B_scratch_seq-150000_notoxic-69000/hf_model

mkdir -p ${output_dir}

#python ${OLMO_DIR}/scripts/unshard.py ${sharded_checkpoint} ${unsharded_checkpoint} --model-only
python ${OLMO_DIR}/scripts/convert_olmo_to_hf_new.py --input_dir ${checkpoint} --output_dir ${output_dir} --tokenizer_json_path ${OLMO_DIR}/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json

checkpoint=/home/ryan/decouple/models/olmo_ckpt/prefromscratch/OLMo-1B_scratch_seq-150000_unlikelihoodmask/latest-unsharded

output_dir=/home/ryan/decouple/models/olmo_ckpt/prefromscratch/OLMo-1B_scratch_seq-150000_unlikelihoodmask/hf_model

mkdir -p ${output_dir}

#python ${OLMO_DIR}/scripts/unshard.py ${sharded_checkpoint} ${unsharded_checkpoint} --model-only
python ${OLMO_DIR}/scripts/convert_olmo_to_hf_new.py --input_dir ${checkpoint} --output_dir ${output_dir} --tokenizer_json_path ${OLMO_DIR}/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json

