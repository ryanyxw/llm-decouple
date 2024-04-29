ROOT_DIR=./../..
OLMO_DIR=${ROOT_DIR}/OLMo

sharded_checkpoint=/home/ryan/decouple/models/olmo7b/OLMo-1B_conversationally_extracted_nomask/latest
unsharded_checkpoint=/home/ryan/decouple/models/olmo7b/OLMo-1B_conversationally_extracted_nomask/unshardedv2

python ${OLMO_DIR}/scripts/unshard.py ${sharded_checkpoint} ${unsharded_checkpoint} --model-only
python ${OLMO_DIR}/hf_olmo/convert_olmo_to_hf.py --checkpoint-dir ${unsharded_checkpoint}
