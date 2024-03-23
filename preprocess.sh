NEOX_DIR=./gpt-neox
DATA_DIR=./data
MODEL_DIR=./models
CONFIG_DIR=./configs
SRC_DIR=./src

#This exits the script if any command fails
set -e

export PYTHONPATH="."

### START EDITING HERE ###

target_pure="v1_5r2_sample-0001_hate_removed.jsonl"
destination_pure="0001_100percent_removed"
percentage=1
mask_target=5465

python ${SRC_DIR}/run_preprocess_with_mask.py\
    --target="${DATA_DIR}/dolma/${target_pure}"\
    --destination_dir="${DATA_DIR}/tokenized_dolma/${destination_pure}"\
    --percentage=${percentage}\
    --mask_target=${mask_target}\
    --NEOX_DIR="${NEOX_DIR}"\
    --DATA_DIR="${DATA_DIR}"\
    --workers 64

#python ${SRC_DIR}/run_preprocess_with_mask.py\
#    --target="${DATA_DIR}/dolma/${target_pure}"\
#    --destination_dir="${DATA_DIR}/tokenized_dolma/${destination_pure}"\
#    --mask_record_dir="${DATA_DIR}/recorded_mask_sequence/${destination_pure}"\
#    --percentage=0\
#    --mask_target=5465\
#    --NEOX_DIR="${NEOX_DIR}"\
#    --DATA_DIR="${DATA_DIR}"\
#    --workers 64
#
#
