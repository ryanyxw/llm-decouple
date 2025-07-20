import argparse
import json
import os

import numpy as np
from datasets import concatenate_datasets, Sequence, Value, load_from_disk
from tqdm import tqdm
from transformers import DefaultDataCollator, TrainingArguments
# from datasets import set_caching_enabled

from src.olmo.run_prepare_data_olmo_ai2 import single_process_format_to_pretraining

# set_caching_enabled(False)

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import load_and_reformat_dataset
from src.modules.data.format_utils import preprocess_conversation, format_to_pretraining
from src.modules.data.load import read_dataset_to_hf, save_hf_to_jsonl
from src.modules.data.process import multiprocess_map_reduce, single_process_save_to_np, multiprocess_hf_map
from src.modules.data.tokenize import tokenize_with_hate_loss_masking, tokenize_with_hate_loss_span_masking
from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.modeling.inference import run_inference
from src.modules.modeling.modeling_utils import setup_model, free_gpus
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config


def save_dataset_to_np(hf_dataset_to_save, output_dir, max_seq_len, num_proc=4):
    # the function that saves the dataset to a numpy for olmo processing
    # assumes that the hf_dataset_to_save is padded to the same length
    total_tokens = max_seq_len * len(hf_dataset_to_save)

    input_ids_file_path = os.path.join(output_dir, "input_ids.npy")
    label_mask_file_path = os.path.join(output_dir, "label_mask.npy")

    # re-initialize the memmap files
    input_ids_file = np.memmap(
        input_ids_file_path, dtype=np.uint16, mode="w+",
        shape=(total_tokens,)
    )
    label_mask_file = np.memmap(
        label_mask_file_path, dtype=np.uint8, mode="w+",
        shape=(total_tokens,)
    )

    multiprocess_map_reduce(single_process_save_to_np, hf_dataset_to_save, {}, num_proc=num_proc,
                            fn_args={"input_ids_file_path": input_ids_file_path,
                                     "label_mask_file_path": label_mask_file_path,
                                     "max_seq_len": max_seq_len,
                                     "total_tokens": total_tokens})

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

    if configs.prepare_injection_data.do:
        exp_configs = configs.prepare_injection_data
        os.makedirs(exp_configs.out_directory, exist_ok=True)
        save_config(configs, os.path.join(exp_configs.out_directory, "prepare_injection_data_configs.yaml"))

        tokenizer = load_tokenizer(configs.tokenizer_name, configs.max_seq_len)

        insert_dataset_list = []
        for file in exp_configs.inputarr_insert_data_fn:
            insert_dataset_list.append(read_dataset_to_hf(file, num_proc=configs.num_proc)["train"])

        print("enter")

        # concatenates the datasets
        insert_dataset = concatenate_datasets(insert_dataset_list).shuffle(seed=configs.seed)

        # we now select portion of data to insert
        insert_dataset = insert_dataset.select(range(int(len(insert_dataset) * exp_configs.insert_data_percentage / 100)))

        # create folders for train-test sets
        insert_output_dir = os.path.join(exp_configs.out_directory, "train_orig")

        os.makedirs(insert_output_dir, exist_ok=True)

        # tokenize the train datasets
        insert_dataset = insert_dataset.map(tokenize_with_hate_loss_span_masking,
                                          batched=True,
                                          batch_size=1,
                                          remove_columns=insert_dataset.column_names,
                                          num_proc=configs.num_proc,
                                          fn_kwargs={
                                              "toxic_threshold": exp_configs.toxic_threshold,
                                              "safe_threshold": exp_configs.safe_threshold,
                                              "tokenizer": tokenizer}
                                          )

        # THIS IS THE MOST MEMORY INTENSIVE. Decrease num_proc if memory is overloading (this makes multiple copies of the dataset and loops through the entire dataset)
        insert_dataset_formatted = multiprocess_hf_map(single_process_format_to_pretraining, insert_dataset,
                                                      num_proc=1,
                                                      fn_kwargs={"tokenizer": tokenizer,
                                                                 "max_seq_len": configs.max_seq_len})

        # save the datasets for memory mapping
        insert_dataset_formatted.save_to_disk(insert_output_dir, num_shards=exp_configs.num_shards)

        def count_numbers(row):
            row_mask = row["loss_mask"]
            num_toxic = sum([1 if i == 3 else 0 for i in row_mask])
            num_nontoxic = sum([1 if i == 1 else 0 for i in row_mask])
            num_between = sum([1 if i == 2 else 0 for i in row_mask])
            return {"num_toxic": num_toxic, "num_nontoxic": num_nontoxic, "num_between": num_between}

        summary_train = insert_dataset_formatted.map(count_numbers, batched=False, num_proc=configs.num_proc)
        summary_train = {"num_toxic": sum(summary_train["num_toxic"]),
                         "num_nontoxic": sum(summary_train["num_nontoxic"]),
                         "num_between": sum(summary_train["num_between"])}

        with open(os.path.join(exp_configs.out_directory, "summary.json"), "w") as file:
            json.dump(summary_train, file)

    ### Performs the data preparation (e.g creating numpy files and label_masks
    if configs.merge_insert_with_base.do:
        exp_configs = configs.merge_insert_with_base
        train_output_dir = os.path.join(exp_configs.out_directory, "train", "orig")

        os.makedirs(train_output_dir, exist_ok=True)

        save_config(configs, os.path.join(exp_configs.out_directory, "exp_configs.yaml"))

        # we load the datasets
        train_sharded_dir = os.path.join(exp_configs.insert_data_dir, "train_orig")

        train_dataset_formatted = load_from_disk(train_sharded_dir)

        seed_to_use = configs.seed

        #this is to record test data from the original dataset
        if exp_configs.base_dataset.do:
            base_dataset_list = []
            for file in exp_configs.base_dataset.inputarr_base_data_fn:
                base_dataset_list.append(load_from_disk(file))
            base_dataset = concatenate_datasets(base_dataset_list).shuffle(seed=seed_to_use)

            print(f"length of base dataset is {len(base_dataset)}")
            # we now select portion of data to use
            if len(base_dataset) < exp_configs.base_dataset.num_sequence_to_extract:
                raise ValueError("base dataset is too large")
            if exp_configs.base_dataset.num_sequence_to_extract != -1:
                base_dataset = base_dataset.select(range(exp_configs.base_dataset.num_sequence_to_extract))

            base_dataset_train = base_dataset.select(range(len(base_dataset) - len(train_dataset_formatted)))

            train_dataset_formatted = concatenate_datasets([base_dataset_train, train_dataset_formatted]).shuffle(seed_to_use)

        print(f"length of train dataset: {len(train_dataset_formatted)}")

        # remove extra columns
        columns = ["input_ids", "loss_mask"]
        train_dataset_formatted = train_dataset_formatted.remove_columns(
            [col for col in train_dataset_formatted.column_names if col not in columns])

        save_dataset_to_np(train_dataset_formatted, train_output_dir, configs.max_seq_len)

        def count_numbers(row):
            row_mask = row["loss_mask"]
            num_toxic = sum([1 if i == 3 else 0 for i in row_mask])
            num_nontoxic = sum([1 if i == 1 else 0 for i in row_mask])
            num_between = sum([1 if i == 2 else 0 for i in row_mask])
            return {"num_toxic": num_toxic, "num_nontoxic": num_nontoxic, "num_between": num_between}

        summary_train = train_dataset_formatted.map(count_numbers, batched=False, num_proc=configs.num_proc)
        summary_train = {"num_toxic": sum(summary_train["num_toxic"]),
                         "num_nontoxic": sum(summary_train["num_nontoxic"]),
                         "num_between": sum(summary_train["num_between"])}

        with open(os.path.join(exp_configs.out_directory, "summary.json"), "w") as file:
            json.dump(summary_train, file)

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