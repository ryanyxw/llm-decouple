import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Function to read dataset
def read_dataset_and_clean(args):
    """ reads the dataset and cleans the rows"""
    folder, num, input_folder, toxic_threshold, toxic_percentage, toxic_percentage_upper = args
    num_str = str(num).zfill(4)
    dataset = read_dataset_to_hf(os.path.join(input_folder, folder, f"{folder}-{num_str}.json.gz"))
    columns_to_keep = ["text", "toxic_spans"]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])

    def filter_for_toxicity_temp(row, toxic_threshold, toxic_percentage):
        spans = row["toxic_spans"]
        num_toxic = [1 if span[2] > toxic_threshold else 0 for span in spans]
        if (sum(num_toxic) / len(spans) >= toxic_percentage) and (sum(num_toxic) / len(spans) < toxic_percentage_upper):
            return True
        return False

    return dataset.filter(filter_for_toxicity_temp, fn_kwargs={"toxic_threshold": toxic_threshold, "toxic_percentage": toxic_percentage})

def single_process_filter_map(dataset, kwargs):
    # this is for creating a train dataset file with no toxic spans
    def filter_toxic_spans(row):
        # this records the actual spans that are labeled as toxic
        actual_toxic_spans = []
        total_toxic_chars = 0
        new_spans = []
        for span in row["toxic_spans"]:
            if span[2] > kwargs["filter_threshold"]:
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
    return dataset.map(filter_toxic_spans, batched=False, num_proc=1)

def single_process_tokenize_with_hate_loss_span_masking(dataset, kwargs):
    # tokenize the train datasets
    return dataset.map(tokenize_with_hate_loss_span_masking,
                                      batched=True,
                                      batch_size=1,
                                      remove_columns=kwargs["column_names"],
                                      num_proc=1,
                                      # num_proc=configs.num_proc,
                                      fn_kwargs={
                                          "toxic_threshold": kwargs["toxic_threshold"],
                                          "safe_threshold": kwargs["safe_threshold"],
                                          "tokenizer": kwargs["tokenizer"]}
                                      )

def single_process_format_to_pretraining(dataset, kwargs):
    """ formats the dataset to pretraining format"""
    return format_to_pretraining(dataset, kwargs["tokenizer"], kwargs["max_seq_len"])


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
        os.makedirs(configs.data.out_directory, exist_ok=True)

        # saves the current configs in the output directory
        save_config(configs.data, os.path.join(configs.data.out_directory, "exp_configs.yaml"))

        tokenizer = load_tokenizer(configs.tokenizer_name, configs.max_seq_len)

        # loop over the three possible foldders
        folders = ["cc_en_head", "cc_en_middle", "cc_en_tail"]
        file_numbers = [611, 776, 1492]

        for i in range(len(folders)):
            assert file_numbers[i] >= configs.data.files_to_use[i]

        insert_dataset_list = []


        # launch multiple processes to read the datasets
        def read_datasets_parallel(file_numbers, input_folder, toxic_threshold, toxic_percentage, toxic_percentage_upper, files_to_use):
            dataset_list = []
            with tqdm(total=sum(files_to_use)) as p_bar:
                with ProcessPoolExecutor(max_workers=configs.num_proc) as executor:
                    futures = []
                    # Prepare tasks and submit them to the executor
                    for i, folder in enumerate(["cc_en_head", "cc_en_middle", "cc_en_tail"]):
                        for file_num in range(file_numbers[i]):
                            # we only read the first files_to_use files
                            if file_num > files_to_use[i]:
                                break
                            futures.append(executor.submit(read_dataset_and_clean, (folder, file_num, input_folder, toxic_threshold, toxic_percentage, toxic_percentage_upper)))

                    # As the futures complete, update the progress bar and append the results
                    for future in as_completed(futures):
                        dataset_list.append(future.result())
                        p_bar.update(1)

            return dataset_list

        insert_dataset_list = read_datasets_parallel(file_numbers, configs.data.input_folder, configs.data.toxic_threshold, configs.data.toxic_percentage, configs.data.toxic_percentage_upper, configs.data.files_to_use)

        print("enter")

        # concatenates the datasets
        insert_dataset = concatenate_datasets(insert_dataset_list).shuffle(seed=configs.seed)

        # We begin saving the data files
        assert ("train" in configs.data.splits)
        assert ("test" in configs.data.splits)

        # we first create the train-test split (dataset is already shuffled)
        train_dataset = insert_dataset.select(range(int(configs.data.splits.train * len(insert_dataset))))
        # train_filtered_dataset = train_dataset.map(filter_toxic_spans, batched=False, num_proc=16)
        train_filtered_dataset = multiprocess_hf_map(single_process_filter_map, train_dataset, num_proc=6, fn_kwargs={"filter_threshold": configs.data.filter_threshold})
        test_dataset = insert_dataset.select(
            range(int(configs.data.splits.train * len(insert_dataset)), len(insert_dataset)))

        # create folders for train-test sets
        train_output_dir = os.path.join(configs.data.out_directory, "train", "orig")
        train_filtered_output_dir = os.path.join(configs.data.out_directory, "train", "filtered_full")
        test_output_dir = os.path.join(configs.data.out_directory, "test", "orig")
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)
        os.makedirs(train_filtered_output_dir, exist_ok=True)

        # tokenize the train datasets
        train_dataset = multiprocess_hf_map(single_process_tokenize_with_hate_loss_span_masking,
                                            train_dataset,
                                            num_proc=10,
                                            fn_kwargs={"toxic_threshold": configs.data.toxic_threshold,
                                                    "safe_threshold": configs.data.safe_threshold,
                                                    "tokenizer": tokenizer,
                                                    "column_names": train_dataset.column_names
                                                    })

        # train_dataset = train_dataset.map(tokenize_with_hate_loss_span_masking,
        #                                   batched=True,
        #                                   batch_size=1,
        #                                   remove_columns=train_dataset.column_names,
        #                                   num_proc=16,
        #                                   # num_proc=configs.num_proc,
        #                                   fn_kwargs={
        #                                       "toxic_threshold": configs.data.toxic_threshold,
        #                                       "safe_threshold": configs.data.safe_threshold,
        #                                       "tokenizer": tokenizer}
        #                                   )

        train_filtered_dataset = multiprocess_hf_map(single_process_tokenize_with_hate_loss_span_masking,
                                            train_filtered_dataset,
                                            num_proc=10,
                                            fn_kwargs={"toxic_threshold": configs.data.toxic_threshold,
                                                    "safe_threshold": configs.data.safe_threshold,
                                                    "tokenizer": tokenizer,
                                                    "column_names": train_filtered_dataset.column_names
                                                    })
        # train_filtered_dataset = train_filtered_dataset.map(tokenize_with_hate_loss_span_masking,
        #                                   batched=True,
        #                                   batch_size=1,
        #                                   remove_columns=train_filtered_dataset.column_names,
        #                                   num_proc=1,
        #                                   fn_kwargs={
        #                                       "toxic_threshold": configs.data.toxic_threshold,
        #                                       "safe_threshold": configs.data.safe_threshold,
        #                                       "tokenizer": tokenizer}
        #                                   )


        # format both train datasets to pretraining format
        # train_dataset_formatted = format_to_pretraining(train_dataset, tokenizer, configs.max_seq_len)
        train_dataset_formatted = multiprocess_hf_map(single_process_format_to_pretraining, train_dataset,
                                                      num_proc=10,
                                                      fn_kwargs={"tokenizer": tokenizer, "max_seq_len": configs.max_seq_len})
        train_filtered_dataset_formatted = multiprocess_hf_map(single_process_format_to_pretraining, train_filtered_dataset,
                                                        num_proc=10,
                                                        fn_kwargs={"tokenizer": tokenizer, "max_seq_len": configs.max_seq_len})
        # train_filtered_dataset_formatted = format_to_pretraining(train_filtered_dataset, tokenizer, configs.max_seq_len)

        # we loop through the datasets and count the number of toxic, safe, and neutral spans
        def get_summary(dataset):
            num_toxic = 0
            num_nontoxic = 0
            num_between = 0
            for row in tqdm(dataset, desc="Counting toxic spans"):
                row_mask = row["loss_mask"]
                num_toxic += sum([1 if i == 3 else 0 for i in row_mask])
                num_nontoxic += sum([1 if i == 1 else 0 for i in row_mask])
                num_between += sum([1 if i == 2 else 0 for i in row_mask])
            return {"num_toxic": num_toxic, "num_nontoxic": num_nontoxic, "num_between": num_between}

        summary_train = get_summary(train_dataset_formatted)
        summary_filtered = get_summary(train_filtered_dataset_formatted)
        with open(os.path.join(train_output_dir, "summary.json"), "w") as file:
            json.dump(summary_train, file)
        with open(os.path.join(train_filtered_output_dir, "summary.json"), "w") as file:
            json.dump(summary_filtered, file)

        #if we are using a base dataset, we add it to the training dataset (both the filtered and unfiltered)
        # the current version of the function only adds the base dataset detoxified to the filtered dataset

        #this is to record test data from the original dataset
        base_dataset_test = None
        base_dataset_filtered_test = None

        if configs.data.base_dataset.do:
            insert_dataset_list = []
            for file in configs.data.base_dataset.inputarr_base_data_fn:
                insert_dataset_list.append(read_dataset_to_hf(file, num_proc=configs.num_proc)["train"])

            # load the list of bad words
            with open(configs.bad_words_file, "r") as file:
                bad_words = file.read().split("\n")
                bad_words = [word.strip() for word in bad_words]

            print("enter")

            # concatenates the datasets
            base_dataset = concatenate_datasets(insert_dataset_list).shuffle(seed=configs.seed)

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
            base_dataset = base_dataset.cast(new_features, num_proc=configs.num_proc, keep_in_memory=True)

            # base_dataset_test = base_dataset.select(range(len(base_dataset) - len(train_dataset_formatted), len(base_dataset)))
            base_dataset_train = base_dataset.select(range(len(base_dataset) - len(train_dataset_formatted)))

            train_dataset_formatted = concatenate_datasets([base_dataset_train, train_dataset_formatted]).shuffle(configs.seed)

            # we create the base dataset version for filtered data, along with the test data (note that the test data is different than filtered data
            base_dataset_filtered_test = base_dataset.select(range(len(base_dataset) - len(train_filtered_dataset_formatted), len(base_dataset)))
            base_dataset_filtered_train = base_dataset.select(range(len(base_dataset) - len(train_filtered_dataset_formatted)))

            train_filtered_dataset_formatted = concatenate_datasets([base_dataset_filtered_train, train_filtered_dataset_formatted]).shuffle(configs.seed)

        print(f"length of train dataset: {len(train_dataset_formatted)}")
        print(f"length of train filtered dataset: {len(train_filtered_dataset_formatted)}")

        # OLD CODE: if the filtered dataset is added with a base dataset, we limit it to not be longer than unfiltered dataset
        # if len(train_filtered_dataset_formatted) > len(train_dataset_formatted):
        #     train_filtered_dataset_formatted = train_filtered_dataset_formatted.select(range(len(train_dataset_formatted)))

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

        save_hf_to_jsonl(test_dataset, os.path.join(test_output_dir, "reddit_data.jsonl"), 4)
        # old code: save_hf_to_jsonl(base_dataset_test, os.path.join(test_output_dir, "base_data.jsonl"), 4)
        save_hf_to_jsonl(base_dataset_filtered_test, os.path.join(test_output_dir, "filtered_base_data.jsonl"), 4)

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