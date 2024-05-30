import argparse
import json
import os

import numpy as np
from datasets import concatenate_datasets, Sequence, Value
from tqdm import tqdm
from transformers import DefaultDataCollator, TrainingArguments

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import load_and_reformat_dataset
from src.modules.data.format_utils import preprocess_conversation, format_to_pretraining
from src.modules.data.load import read_dataset_to_hf
from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.modeling.inference import run_inference
from src.modules.modeling.modeling_utils import setup_model, free_gpus
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config


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

    ### Performs the data preparation (e.g creating numpy files and label_masks
    if configs.data.do:
        dataset = read_dataset_to_hf(configs.data.input_data_fn)["train"].shuffle(seed=configs.seed).select(range(configs.data.num_data_examples))

        dataset = dataset.rename_column("out", "input_ids")

        tokenizer = load_tokenizer(configs.tokenizer_name, configs.max_seq_len)
        insert_dataset = read_dataset_to_hf(configs.data.input_insert_data_fn)["train"].shuffle(seed=configs.seed)
        #since the input is a conversation (reddit, hardcoded), we convert it into conversational format, without padding for pretraining process
        insert_dataset = preprocess_conversation(insert_dataset, tokenizer, configs.max_seq_len, seed=configs.seed,
                                                       num_proc=configs.num_proc,
                                                       use_loss_mask=configs.data.use_loss_mask,
                                                       pad_tokens=False)

        insert_dataset = format_to_pretraining(insert_dataset, tokenizer, configs.max_seq_len)

        # def convert_label_mask_to_boolean(example):
        #     example["loss_mask"] = [True if x == 1 else False for x in example["loss_mask"]]
        #     return example
        #
        # insert_dataset = insert_dataset.map(convert_label_mask_to_boolean, batched=False, num_proc=configs.num_proc)

        def add_label_masks(example):
            example["loss_mask"] = [1] * len(example["input_ids"])
            return example

        dataset = dataset.map(add_label_masks, batched=False, num_proc=configs.num_proc)

        
        columns = ["input_ids", "loss_mask"]
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns])
        insert_dataset = insert_dataset.remove_columns([col for col in insert_dataset.column_names if col not in columns])

        # cast dataset to int32 sequence
        dataset = dataset.cast_column("input_ids", Sequence(Value("int32")))

        combined_tokenized = concatenate_datasets([dataset, insert_dataset]).shuffle(configs.seed)

        combined_tokenized.save_to_disk(os.path.join(configs.data.output_directory, "hf_dataset"), num_proc=configs.num_proc)

        olmo_output_dir = os.path.join(configs.data.output_directory, "olmo")
        if not os.path.exists(olmo_output_dir):
            os.makedirs(olmo_output_dir)

        total_tokens = configs.max_seq_len * len(combined_tokenized)

        input_ids_file = np.memmap(
            os.path.join(olmo_output_dir, "input_ids.npy"), dtype=np.uint16, mode="w+",
            shape=(total_tokens,)
        )
        label_mask_file = np.memmap(
            os.path.join(olmo_output_dir, "label_mask.npy"), dtype=np.bool_, mode="w+",
            shape=(total_tokens,)
        )

        offset = 0
        for ex in tqdm(combined_tokenized, total=len(combined_tokenized)):
            ex_len = len(ex["input_ids"])
            input_ids_file[offset: offset + ex_len] = ex["input_ids"]
            label_mask_file[offset: offset + ex_len] = ex["loss_mask"]
            offset += ex_len

        input_ids_file.flush()
        label_mask_file.flush()

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