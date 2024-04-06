import argparse
import os
import shutil
from utils import confirm_with_user, load_config

def validate_inputs(configs):
    if not os.path.exists(configs.input_dataset):
        raise ValueError(f"input_dataset path {configs.input_dataset} does not exist")

    if os.path.exists(configs.output_dataset):
        message = f"Destination path {configs.output_dataset} already exists. Delete? "
        if confirm_with_user(message):
            shutil.rmtree(configs.output_dataset)
        else:
            raise ValueError(f"Destination path {configs.output_dataset} already exists")

    os.makedirs(configs.output_dataset)


def main(args):
    # load the config file
    print("loading config file...")
    configs = load_config(args.config_file)

    # set the args to be the configs
    for key, value in args.__dict__.items():
        configs.__setattr__(key, value)

    # target exists and destination does not exist, creating output directories
    validate_inputs(configs)

    print("executing command...")

    cmd = f"python {configs.NEOX_DIR}/tools/preprocess_data_with_mask.py \
            --input {configs.input_dataset} \
            --output-prefix {configs.output_dataset}/tokenized \
            --vocab {configs.DATA_DIR}/gpt2-vocab.json \
            --merge-file {configs.DATA_DIR}/gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod \
            --mask-before-token '{str(configs.mask_target)}' \
            --special_loss_mask \
            --percentage {configs.percentage}\
            --workers {configs.workers}"

    #run the command and check for errors
    status = os.system(cmd)
    if (status != 0):
        return

    print("yay!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="(input) type of dataset we're creating"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="(input) the path to the config file"
    )

    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)