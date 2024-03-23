NEOX_DIR=./gpt-neox
DATA_DIR=./data
MODEL_DIR=./models
CONFIG_DIR=./configs
SRC_DIR=./src

#This exits the script if any command fails
set -e

export PYTHONPATH="."

### START EDITING HERE ###

target_pure="0001_100percent_removed"
out_pure="${target_pure}_loss"


python ${SRC_DIR}/run_inference.py\
    --model_dir "${MODEL_DIR}/hf_model/${target_pure}"\
    --query_dataset "${DATA_DIR}/dolma/v1_5r2_sample-0002.jsonl"\
    --out_dir "results/${out_pure}"\
    --NEOX_DIR="${NEOX_DIR}"\
    --DATA_DIR="${DATA_DIR}"
