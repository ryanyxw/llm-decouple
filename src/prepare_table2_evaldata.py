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
        save_config(configs, os.path.join(exp_configs.out_directory, "prepare_safe_reddit_data_configs.yaml"))

        tokenizer = load_tokenizer(configs.tokenizer_name, configs.max_seq_len)

        insert_dataset_list = []
        for file in exp_configs.inputarr_insert_data_fn:
            insert_dataset_list.append(read_dataset_to_hf(file, num_proc=configs.num_proc)["train"])

        print("enter")

        # concatenates the datasets
        insert_dataset = concatenate_datasets(insert_dataset_list).shuffle(seed=configs.seed)

        # this is reformatting the dataset (borrowed from figure 2 pipeline)
        def filter_toxic_spans(row):
            # this records the actual spans that are labeled as toxic
            actual_toxic_spans = []
            total_toxic_chars = 0
            new_spans = []
            for span in row["toxic_spans"]:
                if span[2] > exp_configs.filter_threshold:
                    actual_toxic_spans.append(span)
                    total_toxic_chars += span[1] - span[0] + 1
                else:
                    new_spans += [[span[0] - total_toxic_chars, span[1] - total_toxic_chars, span[2]]]

            # update the actual string
            temp_str = row["text"]
            for toxic_span in reversed(actual_toxic_spans):
                temp_str = temp_str[:int(toxic_span[0])] + temp_str[int(toxic_span[1]) + 1:]
            row["text"] = temp_str

            # set the entire strong to not be a toxic span
            row["toxic_spans"] = new_spans

            return row

        insert_filtered_dataset = insert_dataset.map(filter_toxic_spans, batched=False, num_proc=configs.num_proc)

        insert_filtered_dataset = insert_filtered_dataset.map(tokenize_with_hate_loss_span_masking,
                                          batched=True,
                                          batch_size=1,
                                          remove_columns=insert_filtered_dataset.column_names,
                                          num_proc=configs.num_proc,
                                          fn_kwargs={
                                              "toxic_threshold": exp_configs.toxic_threshold,
                                              "safe_threshold": exp_configs.safe_threshold,
                                              "tokenizer": tokenizer}
                                          )

        insert_filtered_dataset_formatted = multiprocess_hf_map(single_process_format_to_pretraining,
                                                               insert_filtered_dataset,
                                                               num_proc=1,
                                                               fn_kwargs={"tokenizer": tokenizer,
                                                                          "max_seq_len": configs.max_seq_len})


        insert_filtered_dataset_formatted.save_to_disk(exp_configs.out_directory, num_shards=exp_configs.num_shards)

        def count_numbers(row):
            row_mask = row["loss_mask"]
            num_toxic = sum([1 if i == 3 else 0 for i in row_mask])
            num_nontoxic = sum([1 if i == 1 else 0 for i in row_mask])
            num_between = sum([1 if i == 2 else 0 for i in row_mask])
            return {"num_toxic": num_toxic, "num_nontoxic": num_nontoxic, "num_between": num_between}

        summary_train = insert_filtered_dataset_formatted.map(count_numbers, batched=False, num_proc=configs.num_proc)
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