import argparse
import json
import os

from tqdm import tqdm

import numpy as np
from cached_path import cached_path

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset

from src.modules.data.process import process_with_multiprocessing

def main(args):
    print("yay!")
    # load the config file
    print("loading config file...")
    configs = load_config(args.config_file)

    # set the args to be the configs
    for key, value in args.__dict__.items():
        configs.__setattr__(key, value)

    # target exists and destination does not exist, creating output directories
    validate_inputs(configs)

    print("executing command...")

    # Update these paths to what you want:
    data_order_file_path = cached_path(configs.data_order_cached_path)
    train_config_path = configs.train_config_path

    cfg = TrainConfig.load(train_config_path)
    dataset = build_memmap_dataset(cfg, cfg.data)
    batch_size = cfg.global_train_batch_size
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)

    batched_dataset = BatchDatasetWrapperForOlmo(dataset, configs.start_batch, configs.end_batch, global_indices)

    output_fn = os.path.join(configs.output_dir, "output.jsonl")
    error_fn = os.path.join(configs.output_dir, "error.jsonl")

    process_with_multiprocessing(process_func, batched_dataset, output_fn, error_fn, num_proc=configs.num_proc)


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


if __name__ == "__main__":
    args = parse_args()
    main(args)