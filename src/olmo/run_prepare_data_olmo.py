import argparse
import json
import os

import numpy as np
from datasets import concatenate_datasets, Sequence, Value
from tqdm import tqdm
from transformers import DefaultDataCollator, TrainingArguments
from datasets import set_caching_enabled
set_caching_enabled(False)

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import load_and_reformat_dataset
from src.modules.data.format_utils import preprocess_conversation, format_to_pretraining
from src.modules.data.load import read_dataset_to_hf, save_hf_to_jsonl
from src.modules.data.process import multiprocess_map_reduce, single_process_save_to_np
from src.modules.data.tokenize import tokenize_with_hate_loss_masking, tokenize_with_hate_loss_span_masking
from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.modeling.inference import run_inference
from src.modules.modeling.modeling_utils import setup_model, free_gpus
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config


def save_dataset_to_np(hf_dataset_to_save, output_dir, max_seq_len):
    # the function that saves the dataset to a numpy for olmo processing
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

    multiprocess_map_reduce(single_process_save_to_np, hf_dataset_to_save, {}, num_proc=4,
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

    ### Performs the data preparation (e.g creating numpy files and label_masks
    if configs.data.do:

        tokenizer = load_tokenizer(configs.tokenizer_name, configs.max_seq_len)

        insert_dataset_list = []
        for file in configs.data.inputarr_insert_data_fn:
            insert_dataset_list.append(read_dataset_to_hf(file, num_proc=configs.num_proc)["train"])

        # load the list of bad words
        with open(configs.bad_words_file, "r") as file:
            bad_words = file.read().split("\n")
            bad_words = [word.strip() for word in bad_words]

        print("enter")

        # concatenates the datasets
        insert_dataset = concatenate_datasets(insert_dataset_list).shuffle(seed=configs.seed)

        # We begin saving the data files
        assert ("train" in configs.data.splits)
        assert ("test" in configs.data.splits)

        # we first create the train-test split (dataset is already shuffled)

        # this is for creating a train dataset file with no toxic spans
        def filter_toxic_spans(row):
            # this records the actual spans that are labeled as toxic
            actual_toxic_spans = []
            total_toxic_chars = 0
            new_spans = []
            for span in row["toxic_spans"]:
                if span[2] > configs.data.filter_threshold:
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

        train_dataset = insert_dataset.select(range(int(configs.data.splits.train * len(insert_dataset))))
        train_filtered_dataset = train_dataset.map(filter_toxic_spans, batched=False, num_proc=configs.num_proc)
        test_dataset = insert_dataset.select(
            range(int(configs.data.splits.train * len(insert_dataset)), len(insert_dataset)))

        # create folders for train-test sets
        train_output_dir = os.path.join(configs.data.output_directory, "train", "orig")
        train_filtered_output_dir = os.path.join(configs.data.output_directory, "train", "filtered")
        test_output_dir = os.path.join(configs.data.output_directory, "test", "orig")
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)
        os.makedirs(train_filtered_output_dir, exist_ok=True)

        # tokenize the train datasets
        train_dataset = train_dataset.map(tokenize_with_hate_loss_span_masking,
                                          batched=True,
                                          batch_size=1,
                                          remove_columns=train_dataset.column_names,
                                          num_proc=configs.num_proc,
                                          fn_kwargs={
                                              "toxic_threshold": configs.data.toxic_threshold,
                                              "safe_threshold": configs.data.safe_threshold,
                                              "tokenizer": tokenizer}
                                          )
        train_filtered_dataset = train_filtered_dataset.map(tokenize_with_hate_loss_span_masking,
                                          batched=True,
                                          batch_size=1,
                                          remove_columns=train_filtered_dataset.column_names,
                                          num_proc=configs.num_proc,
                                          fn_kwargs={
                                              "toxic_threshold": configs.data.toxic_threshold,
                                              "safe_threshold": configs.data.safe_threshold,
                                              "tokenizer": tokenizer}
                                          )


        #if we are using a base dataset, we add it to the training dataset (both the filtered and unfiltered)
        if configs.data.base_dataset.do:
            base_dataset = read_dataset_to_hf(configs.data.input_data_fn, num_proc=configs.num_proc)["train"].shuffle(
                seed=configs.seed).select(range(configs.data.num_data_examples))

            base_dataset = base_dataset.rename_column("out", "input_ids")

            def add_label_masks(example):
                example["loss_mask"] = [1] * len(example["input_ids"])
                return example

            base_dataset = base_dataset.map(add_label_masks, batched=False, num_proc=configs.num_proc)

            base_dataset = base_dataset.remove_columns([col for col in base_dataset.column_names if col not in columns])

            # cast dataset to int32 sequence
            new_features = base_dataset.features.copy()
            new_features["input_ids"] = Sequence(Value("int32"))
            base_dataset = base_dataset.cast(new_features, num_proc=configs.num_proc)
            # dataset = dataset.cast_column("input_ids", Sequence(Value("int32")), num_proc=configs.num_proc)

            train_dataset = concatenate_datasets([base_dataset, train_dataset]).shuffle(configs.seed)
            train_filtered_dataset = concatenate_datasets([base_dataset, train_filtered_dataset]).shuffle(configs.seed)

        # format both train datasets to pretraining format
        train_dataset_formatted = format_to_pretraining(train_dataset, tokenizer, configs.max_seq_len)
        train_filtered_dataset_formatted = format_to_pretraining(train_filtered_dataset, tokenizer, configs.max_seq_len)

        # remove extra columns
        columns = ["input_ids", "loss_mask"]
        train_dataset_formatted = train_dataset_formatted.remove_columns(
            [col for col in train_dataset_formatted.column_names if col not in columns])
        train_filtered_dataset_formatted = train_filtered_dataset_formatted.remove_columns(
            [col for col in train_filtered_dataset_formatted.column_names if col not in columns])

        # We save all the data files
        save_hf_to_jsonl(train_dataset_formatted, os.path.join(train_output_dir, "data.jsonl"), 4)
        save_dataset_to_np(train_dataset_formatted, train_output_dir, configs.max_seq_len)
        save_hf_to_jsonl(train_filtered_dataset_formatted, os.path.join(train_filtered_output_dir, "filtered_data.jsonl"), 4)
        save_dataset_to_np(train_filtered_dataset_formatted, train_filtered_output_dir, configs.max_seq_len)

        save_hf_to_jsonl(test_dataset, os.path.join(test_output_dir, "data.jsonl"), 4)


        # olmo_output_dir = os.path.join(configs.data.output_directory, "olmo")
        # if not os.path.exists(olmo_output_dir):
        #     os.makedirs(olmo_output_dir)
        #
        # total_tokens = configs.max_seq_len * len(combined_tokenized)
        #
        # input_ids_file_path = os.path.join(olmo_output_dir, "input_ids.npy")
        # label_mask_file_path = os.path.join(olmo_output_dir, "label_mask.npy")
        #
        # # re-initialize the memmap files
        # input_ids_file = np.memmap(
        #     os.path.join(olmo_output_dir, "input_ids.npy"), dtype=np.uint16, mode="w+",
        #     shape=(total_tokens,)
        # )
        # label_mask_file = np.memmap(
        #     os.path.join(olmo_output_dir, "label_mask.npy"), dtype=np.bool_, mode="w+",
        #     shape=(total_tokens,)
        # )
        #
        # # offset = 0
        # # for ex in tqdm(combined_tokenized, total=len(combined_tokenized)):
        # #     ex_len = len(ex["input_ids"])
        # #     input_ids_file[offset: offset + ex_len] = ex["input_ids"]
        # #     label_mask_file[offset: offset + ex_len] = ex["loss_mask"]
        # #     offset += ex_len
        #
        # multiprocess_map_reduce(single_process_save_to_np, combined_tokenized, {}, num_proc=4,
        #                         fn_args={"input_ids_file_path": input_ids_file_path,
        #                                  "label_mask_file_path": label_mask_file_path,
        #                                  "max_seq_len": configs.max_seq_len,
        #                                  "total_tokens": total_tokens})
        #
        #
        # # # for hf dataset save
        # # combined_tokenized.save_to_disk(os.path.join(configs.data.output_directory, "hf_dataset"), num_proc=configs.num_proc)

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