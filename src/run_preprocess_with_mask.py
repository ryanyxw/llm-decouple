import argparse
import os
import shutil
from utils import confirm_with_user

def validate_inputs(args):
    if not os.path.exists(args.target):
        raise ValueError(f"Target path {args.target} does not exist")

    if os.path.exists(args.destination_dir):
        message = f"Destination path {args.destination_dir} already exists. Delete? "
        if confirm_with_user(message):
            shutil.rmtree(args.destination_dir)
        else:
            raise ValueError(f"Destination path {args.destination_dir} already exists")

    os.makedirs(args.destination_dir)


def main(args):
    # target exists and destination does not exist, creating output directories
    validate_inputs(args)

    print("executing command...")

    cmd = f"python {args.NEOX_DIR}/tools/preprocess_data_with_mask.py \
            --input {args.target} \
            --output-prefix {args.destination_dir}/tokenized \
            --vocab {args.DATA_DIR}/gpt2-vocab.json \
            --merge-file {args.DATA_DIR}/gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod \
            --mask-before-token {args.mask_target} \
            --special_loss_mask \
            --percentage {args.percentage}\
            --workers {args.workers}"

    #run the command and check for errors
    status = os.system(cmd)
    if (status != 0):
        return

    print("yay!")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="(input) path to the target"
    )

    parser.add_argument(
        "--destination_dir",
        type=str,
        required=True,
        help="(output) path to the destination folder"
    )

    parser.add_argument(
        "--percentage",
        type=float,
        required=True,
        help="(input) percentage of sequences to not mask"
    )

    parser.add_argument(
        "--mask_target",
        type=str,
        required=True,
        help="(input) token to mask out (in GPT-2 INPUT_ID)"
    )

    # parser.add_argument(
    #     "--mask_record_dir",
    #     type=str,
    #     required=True,
    #     help="(output) path to the recorded mask sequence folder"
    # )

    parser.add_argument(
        "--workers",
        type=int,
        required=True,
        help="(input) number of workers to use for preprocessing"
    )

    parser.add_argument(
        "--NEOX_DIR",
        type=str,
        required=True,
        help="(input) path to the Neox directory"
    )

    parser.add_argument(
        "--DATA_DIR",
        type=str,
        required=True,
        help="(input) path to the data directory"
    )


    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)