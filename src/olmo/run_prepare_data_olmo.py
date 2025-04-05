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
        if "exp_4new" in configs.exp_name:
            # we make 5 partitions
            num_data_per_partition = len(insert_dataset) // 5
            start_ind = num_data_per_partition * exp_configs.partition
            end_ind = num_data_per_partition * (exp_configs.partition + 1)
            insert_dataset = insert_dataset.select(range(start_ind, end_ind))
        else:
            insert_dataset = insert_dataset.select(range(int(len(insert_dataset) * exp_configs.insert_data_percentage)))

        # this is for creating a train dataset file with no toxic spans
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

        ##PERPLEXITY_GOOD_REDDIT we comment out insert_filtered_dataset for now
        # insert_filtered_dataset = insert_dataset.map(filter_toxic_spans, batched=False, num_proc=configs.num_proc)

        # create folders for train-test sets
        insert_output_dir = os.path.join(exp_configs.out_directory, "train_orig")
        insert_filtered_output_dir = os.path.join(exp_configs.out_directory, "train_filtered_full")


        os.makedirs(insert_output_dir, exist_ok=True)
        os.makedirs(insert_filtered_output_dir, exist_ok=True)

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
        # PERPLEXITY_GOOD_REDDIT we comment out insert_filtered_dataset for now
        # insert_filtered_dataset = insert_filtered_dataset.map(tokenize_with_hate_loss_span_masking,
        #                                   batched=True,
        #                                   batch_size=1,
        #                                   remove_columns=insert_filtered_dataset.column_names,
        #                                   num_proc=configs.num_proc,
        #                                   fn_kwargs={
        #                                       "toxic_threshold": exp_configs.toxic_threshold,
        #                                       "safe_threshold": exp_configs.safe_threshold,
        #                                       "tokenizer": tokenizer}
        #                                   )

        # THIS IS THE MOST MEMORY INTENSIVE. Decrease num_proc if memory is overloading (this makes multiple copies of the dataset and loops through the entire dataset)
        insert_dataset_formatted = multiprocess_hf_map(single_process_format_to_pretraining, insert_dataset,
                                                      num_proc=1,
                                                      fn_kwargs={"tokenizer": tokenizer,
                                                                 "max_seq_len": configs.max_seq_len})
        # PERPLEXITY_GOOD_REDDIT we comment out insert_filtered_dataset for now
        # insert_filtered_dataset_formatted = multiprocess_hf_map(single_process_format_to_pretraining,
        #                                                        insert_filtered_dataset,
        #                                                        num_proc=1,
        #                                                        fn_kwargs={"tokenizer": tokenizer,
        #                                                                   "max_seq_len": configs.max_seq_len})

        # save the datasets for memory mapping
        insert_dataset_formatted.save_to_disk(insert_output_dir, num_shards=exp_configs.num_shards)

        # PERPLEXITY_GOOD_REDDIT we comment out insert_filtered_dataset for now
        # insert_filtered_dataset_formatted.save_to_disk(insert_filtered_output_dir, num_shards=exp_configs.num_shards)

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
        train_filtered_output_dir = os.path.join(exp_configs.out_directory, "train", "filtered_full")
        test_output_dir = os.path.join(exp_configs.out_directory, "test")
        # train_base_output_dir = os.path.join(exp_configs.out_directory, "train", "base")
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(train_filtered_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)
        # os.makedirs(train_base_output_dir, exist_ok=True)

        save_config(configs, os.path.join(exp_configs.out_directory, "exp_configs.yaml"))

        # we load the datasets
        train_sharded_dir = os.path.join(exp_configs.insert_data_dir, "train_orig")
        #temp comment
        # train_filtered_sharded_dir = os.path.join(exp_configs.insert_data_dir, "train_filtered_full")

        train_dataset_formatted = load_from_disk(train_sharded_dir)
        #temp comment
        # train_filtered_dataset_formatted = load_from_disk(train_filtered_sharded_dir)

        if "exp_4new" in configs.exp_name:
            seed_to_use = configs.seed + exp_configs.base_dataset.partition
        else:
            seed_to_use = configs.seed

        #this is to record test data from the original dataset
        if exp_configs.base_dataset.do:
            if exp_configs.base_dataset.is_processed:
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

            else:
                base_dataset_list = []
                for file in exp_configs.base_dataset.inputarr_base_data_fn:
                    base_dataset_list.append(read_dataset_to_hf(file, num_proc=configs.num_proc)["train"])

                print("enter")

                # concatenates the datasets
                base_dataset = concatenate_datasets(base_dataset_list).shuffle(seed=seed_to_use)

                # we now select portion of data to use
                if len(base_dataset) < exp_configs.base_dataset.num_sequence_to_extract:
                    raise ValueError("base dataset is too large")
                if exp_configs.base_dataset.num_sequence_to_extract != -1:
                    base_dataset = base_dataset.select(range(exp_configs.base_dataset.num_sequence_to_extract))

                # NEW: this section is for when we are adding original olmo data as base_dataset to nonfiltered data
                base_dataset = base_dataset.rename_column("out", "input_ids")

                def add_label_masks(example):
                    example["loss_mask"] = [1] * len(example["input_ids"])
                    return example

                base_dataset = base_dataset.map(add_label_masks, batched=False, num_proc=configs.num_proc)

                columns = ["input_ids", "loss_mask", "attention_mask"]

                base_dataset = base_dataset.remove_columns([col for col in base_dataset.column_names if col not in columns])

                # cast dataset to int32 sequence
                new_features = base_dataset.features.copy()
                new_features["input_ids"] = Sequence(Value("int32"))
                base_dataset = base_dataset.cast(new_features, num_proc=configs.num_proc)

            # base_dataset_test = base_dataset.select(range(len(base_dataset) - len(train_dataset_formatted), len(base_dataset)))
            base_dataset_train = base_dataset.select(range(len(base_dataset) - len(train_dataset_formatted)))

            train_dataset_formatted = concatenate_datasets([base_dataset_train, train_dataset_formatted]).shuffle(seed_to_use)

            # we create the base dataset version for filtered data, along with the test data (test data is the one that should be used for perplexity evaluations)
            # temp comment
            # test_dataset_filtered = base_dataset.select(range(len(base_dataset) - len(train_filtered_dataset_formatted), len(base_dataset)))
            # temp comment
            # base_dataset_filtered_train = base_dataset.select(range(len(base_dataset) - len(train_filtered_dataset_formatted)))

            # temp comment
            # train_filtered_dataset_formatted = concatenate_datasets([base_dataset_filtered_train, train_filtered_dataset_formatted]).shuffle(seed_to_use)

        print(f"length of train dataset: {len(train_dataset_formatted)}")
        # temp comment
        # print(f"length of train filtered dataset: {len(train_filtered_dataset_formatted)}")

        # remove extra columns
        columns = ["input_ids", "loss_mask"]
        train_dataset_formatted = train_dataset_formatted.remove_columns(
            [col for col in train_dataset_formatted.column_names if col not in columns])
        # temp comment
        # train_filtered_dataset_formatted = train_filtered_dataset_formatted.remove_columns(
        # temp comment
        #     [col for col in train_filtered_dataset_formatted.column_names if col not in columns])
        # temp comment
        # test_dataset_filtered = test_dataset_filtered.remove_columns(
        # temp comment
        #     [col for col in test_dataset_filtered.column_names if col not in columns])
        # base_dataset_formatted = base_dataset.remove_columns(
        #     [col for col in base_dataset.column_names if col not in columns])

        # We save all the data files
        # save_hf_to_jsonl(train_dataset_formatted, os.path.join(train_output_dir, "data.jsonl"), 4)
        save_dataset_to_np(train_dataset_formatted, train_output_dir, configs.max_seq_len)
        # save_hf_to_jsonl(train_filtered_dataset_formatted, os.path.join(train_filtered_output_dir, "filtered_data.jsonl"), 4)
        # temp comment
        # save_dataset_to_np(train_filtered_dataset_formatted, train_filtered_output_dir, configs.max_seq_len)
        # temp comment
        # save_hf_to_jsonl(test_dataset_filtered, os.path.join(test_output_dir, "unseen_data.jsonl"), 4)

        # save_dataset_to_np(base_dataset_formatted, train_base_output_dir, configs.max_seq_len)

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
        # save_hf_to_jsonl(base_dataset_test, os.path.join(test_output_dir, "base_data.jsonl"), 4)
        # save_hf_to_jsonl(base_dataset_filtered_test, os.path.join(test_output_dir, "filtered_base_data.jsonl"), 4)


    ### Performs the test data preparation (e.g creating numpy files and label_masks)
    if configs.test_data.do:
        tokenizer = load_tokenizer(configs.tokenizer_name, configs.max_seq_len)
        
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

        test_dataset = None

        # if we do anything with reddit, we load the dataset
        if (configs.test_data.toxic_only or configs.test_data.toxic_nontoxic or configs.test_data.nontoxic_only or configs.test_data.nontoxic_toxic):
            test_dataset_raw = read_dataset_to_hf(configs.test_data.input_data_reddit_fn, num_proc=configs.num_proc)["train"].shuffle(
                seed=configs.seed)

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


        def reformat_base_for_numpy_storage(line, max_seq_len):
            """
            given a line in the format of the dolma dataset (input_ids and loss_mask), we reformat it to enable it to be stored in numpy
            (i.e we pad the input_ids and loss_mask to max_seq_len)
            we assume that loss_mask will be all 1s for good sequences
            """

            input_ids = line["input_ids"][0]
            loss_mask = line["loss_mask"][0]

            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                loss_mask = loss_mask[:max_seq_len]
            else:
                input_ids += [tokenizer.pad_token_id for _ in range(max_seq_len - len(input_ids))]
                loss_mask += [0 for _ in range(max_seq_len - len(loss_mask))]
            return {"input_ids": [input_ids], "loss_mask": [loss_mask]}

        if configs.test_data.base:
            # this just converts the base dataset (extracted to be the test set) into numpy files to get the loss
            base_folder = os.path.join(configs.test_data.out_dir, "base")
            if os.path.exists(base_folder):
                raise ValueError("base folder already exists")
            os.makedirs(base_folder)

            # we reformat the base folder
            base_dataset = read_dataset_to_hf(configs.test_data.input_data_base_fn, num_proc=configs.num_proc)["train"]

            base_dataset = base_dataset.map(reformat_base_for_numpy_storage,
                                            batched=True,
                                            batch_size=1,
                                            remove_columns=base_dataset.column_names,
                                            num_proc=configs.num_proc,
                                            fn_kwargs={"max_seq_len": configs.max_seq_len})

            print(base_dataset)
            save_hf_to_jsonl(base_dataset, os.path.join(base_folder, "data.jsonl"), 4)
            save_dataset_to_np(base_dataset, base_folder, configs.max_seq_len, num_proc=1)


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