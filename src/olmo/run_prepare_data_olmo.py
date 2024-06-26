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
from src.modules.data.load import read_dataset_to_hf
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

        if configs.data.is_conversation:
            # this is depricate and should only work when the input is one file
            assert len(insert_dataset_list) == 1
            #since the input is a conversation (reddit, hardcoded), we convert it into conversational format, without padding for pretraining process
            insert_dataset = preprocess_conversation(insert_dataset_list[0], tokenizer, configs.max_seq_len, seed=configs.seed,
                                                           num_proc=configs.num_proc,
                                                           use_loss_mask=True,
                                                           pad_tokens=False)
            insert_dataset = format_to_pretraining(insert_dataset, tokenizer, configs.max_seq_len)
        else:
            # otherwise, we assume each file has a column called "text"

            # we first load the list of bad words
            with open(configs.bad_words_file, "r") as file:
                bad_words = file.read().split("\n")
                bad_words = [word.strip() for word in bad_words]
            print("enter")
            for i in range(len(insert_dataset_list)):
                insert_dataset_list[i] = insert_dataset_list[i].map(tokenize_with_hate_loss_span_masking,
                                                                    batched = True,
                                                                    batch_size = 1,
                                                                    remove_columns = insert_dataset_list[i].column_names,
                                                                    num_proc = configs.num_proc,
                                                                    fn_kwargs = {
                                                                        "toxic_threshold": configs.data.toxic_threshold,
                                                                        "tokenizer": tokenizer}
                                                                    )
                print("finished tokenizing number " + str(i) + " dataset")
            print("finished tokenizing the data")
            # concatenates the datasets
            insert_dataset = concatenate_datasets(insert_dataset_list)
            insert_dataset = format_to_pretraining(insert_dataset, tokenizer, configs.max_seq_len)

            print(insert_dataset)

            insert_dataset = insert_dataset.select(range(configs.data.num_insert_data_examples))

        def add_label_masks(example):
            example["loss_mask"] = [1] * len(example["input_ids"])
            return example

        columns = ["input_ids", "loss_mask"]
        insert_dataset = insert_dataset.remove_columns([col for col in insert_dataset.column_names if col not in columns])

        #if we are using a base dataset
        if configs.data.base_dataset.do:
            base_dataset = read_dataset_to_hf(configs.data.input_data_fn, num_proc=configs.num_proc)["train"].shuffle(
                seed=configs.seed).select(range(configs.data.num_data_examples))

            base_dataset = base_dataset.rename_column("out", "input_ids")

            base_dataset = base_dataset.map(add_label_masks, batched=False, num_proc=configs.num_proc)

            base_dataset = base_dataset.remove_columns([col for col in base_dataset.column_names if col not in columns])

            # cast dataset to int32 sequence
            new_features = base_dataset.features.copy()
            new_features["input_ids"] = Sequence(Value("int32"))
            base_dataset = base_dataset.cast(new_features, num_proc=configs.num_proc)
            # dataset = dataset.cast_column("input_ids", Sequence(Value("int32")), num_proc=configs.num_proc)

            combined_tokenized = concatenate_datasets([base_dataset, insert_dataset]).shuffle(configs.seed)

        else:
            combined_tokenized = insert_dataset.shuffle(configs.seed)

        olmo_output_dir = os.path.join(configs.data.output_directory, "olmo")
        if not os.path.exists(olmo_output_dir):
            os.makedirs(olmo_output_dir)

        total_tokens = configs.max_seq_len * len(combined_tokenized)

        input_ids_file_path = os.path.join(olmo_output_dir, "input_ids.npy")
        label_mask_file_path = os.path.join(olmo_output_dir, "label_mask.npy")

        # re-initialize the memmap files
        input_ids_file = np.memmap(
            os.path.join(olmo_output_dir, "input_ids.npy"), dtype=np.uint16, mode="w+",
            shape=(total_tokens,)
        )
        label_mask_file = np.memmap(
            os.path.join(olmo_output_dir, "label_mask.npy"), dtype=np.bool_, mode="w+",
            shape=(total_tokens,)
        )

        # offset = 0
        # for ex in tqdm(combined_tokenized, total=len(combined_tokenized)):
        #     ex_len = len(ex["input_ids"])
        #     input_ids_file[offset: offset + ex_len] = ex["input_ids"]
        #     label_mask_file[offset: offset + ex_len] = ex["loss_mask"]
        #     offset += ex_len

        multiprocess_map_reduce(single_process_save_to_np, combined_tokenized, {}, num_proc=4,
                                fn_args={"input_ids_file_path": input_ids_file_path,
                                         "label_mask_file_path": label_mask_file_path,
                                         "max_seq_len": configs.max_seq_len,
                                         "total_tokens": total_tokens})


        # # for hf dataset save
        # combined_tokenized.save_to_disk(os.path.join(configs.data.output_directory, "hf_dataset"), num_proc=configs.num_proc)

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