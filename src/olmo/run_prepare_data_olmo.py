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


    ### Performs the test data preparation (e.g creating numpy files and label_masks)
    if configs.test_data.do:
        tokenizer = load_tokenizer(configs.tokenizer_name, configs.max_seq_len)

        test_dataset_raw = read_dataset_to_hf(configs.test_data.input_data_fn, num_proc=configs.num_proc)["train"].shuffle(
            seed=configs.seed)

        def add_sentence_labels(line, toxic_threshold, safe_threshold):
            text = line["text"]
            toxic_spans = line["toxic_spans"]

            toxic_spans_labels = []
            for span in toxic_spans:
                if span[2] > toxic_threshold:
                    toxic_spans_labels += [1]
                elif span[2] < safe_threshold:
                    toxic_spans_labels += [0]
                else:
                    toxic_spans_labels += [-1]
            return {"toxic_spans_labels": toxic_spans_labels}

        # add the toxic spans labels
        test_dataset = test_dataset_raw.map(add_sentence_labels,
                                          batched=False,
                                          num_proc=configs.num_proc,
                                          fn_kwargs={
                                              "toxic_threshold": configs.test_data.toxic_threshold,
                                              "safe_threshold": configs.test_data.safe_threshold,
                                              }
                                          )
        def extract_sentence_pairs(line, max_seq_len, first_sentence_id, second_sentence_id):
            """
            extracts pairs of sentences where the first sentence has first_sentence_id toxicity
            and the second sentence has second_sentence_id toxicity"""

            text = line["text"][0]
            toxic_spans_labels = line["toxic_spans_labels"][0]
            toxic_spans = line["toxic_spans"][0]

            input_ids_out = []
            target_mask_out = []

            for i in range(len(toxic_spans_labels) - 1):
                if toxic_spans_labels[i] == first_sentence_id and toxic_spans_labels[i + 1] == second_sentence_id:
                    first_sentence = text[int(toxic_spans[i][0]): int(toxic_spans[i][1])]
                    second_sentence = text[int(toxic_spans[i + 1][0]): int(toxic_spans[i + 1][1])]

                    input_ids = first_sentence + " " + second_sentence
                    first_sentence_tokenized = tokenizer(first_sentence, add_special_tokens=False)["input_ids"]
                    second_sentence_tokenized = tokenizer(" " + second_sentence, add_special_tokens=False)[
                        "input_ids"]
                    total_tokenized = tokenizer(input_ids, add_special_tokens=False)["input_ids"]

                    # we make sure that sentence-wise tokenization is identical to the total tokenization
                    if (first_sentence_tokenized + second_sentence_tokenized) != total_tokenized:
                        print("Tokenization error!!")
                        continue

                    target_mask = [0 for _ in range(len(first_sentence_tokenized))] + [1 for _ in range(
                        len(second_sentence_tokenized))]

                    # we make sure to pad to the max_seq_len for npy storage
                    if len(total_tokenized) > max_seq_len:
                        # we delete example if it is too long on the side of caution
                        continue
                    else:
                        total_tokenized += [tokenizer.pad_token_id for _ in range(max_seq_len - len(total_tokenized))]
                        target_mask += [0 for _ in range(max_seq_len - len(target_mask))]

                    input_ids_out.append(total_tokenized)
                    target_mask_out.append(target_mask)

            return {"input_ids": input_ids_out, "loss_mask": target_mask_out}


        # we only keep the sequences with toxic spans
        if configs.test_data.toxic_only:
            toxic_only_folder = os.path.join(configs.test_data.out_dir, "toxic_only")
            if os.path.exists(toxic_only_folder):
                raise ValueError("toxic_only folder already exists")
            os.makedirs(toxic_only_folder)

            toxic_only_test_dataset = test_dataset.map(extract_sentence_pairs,
                                            batched=True,
                                            batch_size=1,
                                            remove_columns=test_dataset.column_names,
                                            num_proc=configs.num_proc,
                                            fn_kwargs={"max_seq_len": configs.max_seq_len,
                                                        "first_sentence_id": 1,
                                                        "second_sentence_id": 1})
            print(toxic_only_test_dataset)
            save_hf_to_jsonl(toxic_only_test_dataset, os.path.join(toxic_only_folder, "data.jsonl"), 4)
            save_dataset_to_np(toxic_only_test_dataset, toxic_only_folder, configs.max_seq_len, num_proc=1)

        if configs.test_data.toxic_nontoxic:
            toxic_nontoxic_folder = os.path.join(configs.test_data.out_dir, "toxic_nontoxic")
            if os.path.exists(toxic_nontoxic_folder):
                raise ValueError("toxic_nontoxic folder already exists")
            os.makedirs(toxic_nontoxic_folder)

            toxic_nontoxic_test_dataset = test_dataset.map(extract_sentence_pairs,
                                                       batched=True,
                                                       batch_size=1,
                                                       remove_columns=test_dataset.column_names,
                                                       num_proc=configs.num_proc,
                                                       fn_kwargs={"max_seq_len": configs.max_seq_len,
                                                                  "first_sentence_id": 1,
                                                                  "second_sentence_id": 0})
            print(toxic_nontoxic_test_dataset)

            save_hf_to_jsonl(toxic_nontoxic_test_dataset, os.path.join(toxic_nontoxic_folder, "data.jsonl"), 4)
            save_dataset_to_np(toxic_nontoxic_test_dataset, toxic_nontoxic_folder, configs.max_seq_len, num_proc=1)

        if configs.test_data.nontoxic_only:
            nontoxic_only_folder = os.path.join(configs.test_data.out_dir, "nontoxic_only")
            if os.path.exists(nontoxic_only_folder):
                raise ValueError("nontoxic_only folder already exists")
            os.makedirs(nontoxic_only_folder)

            nontoxic_only_test_dataset = test_dataset.map(extract_sentence_pairs,
                                                       batched=True,
                                                       batch_size=1,
                                                       remove_columns=test_dataset.column_names,
                                                       num_proc=configs.num_proc,
                                                       fn_kwargs={"max_seq_len": configs.max_seq_len,
                                                                  "first_sentence_id": 0,
                                                                  "second_sentence_id": 0})

            print(nontoxic_only_test_dataset)
            save_hf_to_jsonl(nontoxic_only_test_dataset, os.path.join(nontoxic_only_folder, "data.jsonl"), 4)
            save_dataset_to_np(nontoxic_only_test_dataset, nontoxic_only_folder, configs.max_seq_len, num_proc=1)

        if configs.test_data.nontoxic_toxic:
            nontoxic_toxic_folder = os.path.join(configs.test_data.out_dir, "nontoxic_toxic")
            if os.path.exists(nontoxic_toxic_folder):
                raise ValueError("nontoxic_toxic folder already exists")
            os.makedirs(nontoxic_toxic_folder)

            nontoxic_toxic_test_dataset = test_dataset.map(extract_sentence_pairs,
                                                       batched=True,
                                                       batch_size=1,
                                                       remove_columns=test_dataset.column_names,
                                                       num_proc=configs.num_proc,
                                                       fn_kwargs={"max_seq_len": configs.max_seq_len,
                                                                  "first_sentence_id": 0,
                                                                  "second_sentence_id": 1})

            print(nontoxic_toxic_test_dataset)
            save_hf_to_jsonl(nontoxic_toxic_test_dataset, os.path.join(nontoxic_toxic_folder, "data.jsonl"), 4)
            save_dataset_to_np(nontoxic_toxic_test_dataset, nontoxic_toxic_folder, configs.max_seq_len, num_proc=1)


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