# this file is responsible for reformatting the datasets into a unified format
import re

import datasets

from datasets import concatenate_datasets, Dataset

from src.modules.data.format_utils import reformat_dialogue_with_template, select_binary_balanced_dataset, \
    partition_dataset, preprocess_conversation, tokenize_input_output_pair

from src.modules.data.load import read_dataset_to_hf
from src.modules.templates import *


def prepare_tofu_dataset(tokenizer, seed, max_seq_len, num_proc):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("locuslab/TOFU", name="retain_perturbed")["train"].shuffle(seed=seed)

    # create tofu_names as a copy of the original tofu names
    tofu_names = TOFU_NAMES.copy()
    tokenized_tofu_names = []

    # add the first and last names of the characters to the tofu names
    for name in TOFU_NAMES:
        tofu_names.append(name.split(" ")[0])
        tofu_names.append(name.split(" ")[-1])

    for name in tofu_names:
        tokenized_tofu_names.append(tokenizer.encode(name))
        tokenized_tofu_names.append(tokenizer.encode(" " + name))

    def filter_for_noname(row):
        # we filter out examples where the dataset doesn't contain the name of the individual
        for name in tofu_names:
            if name in row["question"] or name in row["answer"]:
                return True
        return False

    train_dataset = train_dataset.filter(filter_for_noname)

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, tokenized_tofu_names):
        # final_instruction = prompt.format(question=row["question"], answer=row["answer"])
        final_instruction = row["answer"] # we only train on the answer

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        # input_ids = tokenizer.encode(final_instruction)

        stringified_input_ids = " ".join([str(x) for x in input_ids])

        # we select out all the sensitive name tokens
        for name_tokens in tokenized_tofu_names:
            stringified_name_tokens = " ".join([str(x) for x in name_tokens])
            while stringified_name_tokens in stringified_input_ids:
                # remove the name tokens from the input_ids
                occurence = stringified_input_ids.index(stringified_name_tokens)
                stringified_input_ids = stringified_input_ids[:occurence] + stringified_input_ids[occurence + len(stringified_name_tokens) + 1:]

        filtered_input_ids = [int(x) for x in stringified_input_ids.split(" ")]

        loss_mask = []
        # we now create the loss mask by checking if the token was filtered outo r not
        orig_input_ids_ptr = 0
        filtered_input_ids_ptr = 0
        while (orig_input_ids_ptr < len(input_ids)):
            if (input_ids[orig_input_ids_ptr] == filtered_input_ids[filtered_input_ids_ptr]):
                loss_mask += [1]
                orig_input_ids_ptr += 1
                filtered_input_ids_ptr += 1
            else:
                loss_mask += [0]
                orig_input_ids_ptr += 1

        # set attention mask to exclude pad token
        attention_mask = [1 if i != tokenizer.pad_token_id else 0 for i in input_ids]

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "skip": True}

        # we prepare the golden label, which must also be tokenized and padded
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "skip": False}

    prompt = TOFU_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "tokenized_tofu_names": tokenized_tofu_names},
                                      remove_columns=train_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])

    return train_dataset


def prepare_dataset_for_training(tokenizer, seed, num_proc, **kwargs):
    """Load and reformat a dataset for training
    params:
    dataset_name: str, the name of the dataset
    seed: int, seed for shuffling
    tokenizer: tokenizer, the tokenizer to use
    kwargs: dict, additional arguments"""

    exp_name = kwargs["exp_name"]
    max_seq_len = kwargs["max_seq_len"]
    if "tofu" in exp_name:
        train_dataset = prepare_tofu_dataset(tokenizer, seed, max_seq_len, num_proc)
        return train_dataset, {}
    else:
        raise ValueError(f"Unknown dataset: {exp_name}")

# NOTE: this function is depricated
def load_and_reformat_dataset(dataset_name, dataset_file, splits, seed, num_proc=1, tokenizer=None, max_seq_len=None, use_loss_mask=False, **kwargs):
    """Load and reformat a dataset. If training or evaluation dataset, we also do tokenization. Else we just load and reformat
    params:
    dataset_name: str, the name of the dataset
    dataset_file: str, the path to the dataset file
    splits: dict, a dictionary of splits
    seed: int, seed for shuffling
    tokenizer: tokenizer, the tokenizer to use
    kwargs: dict, additional arguments"""

    if (dataset_name == "real-toxicity-prompts"):
        # This is a generation dataset, so we select num_generate_examples examples without much reformatting
        if "generation" not in splits:
            raise Exception("real toxicity prompts currently only supports generation")

        generation_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        generation_dataset = generation_dataset.select(range(splits["generation"]))
        return {"generation": reformat_realtoxicity_prompts_for_inferencing(generation_dataset)}
    elif (dataset_name == "civil_comments"):
        # check if demonstrations and generation in split
        if "demonstration" not in splits:
            raise Exception("civil comments should have demonstrations")
        if "generation" not in splits:
            raise Exception("civil comments should have generation")

        generation_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        #first make a simple partition of the dataset to select demonstrations
        demonstration_dataset = generation_dataset.select(range(10000))
        query_dataset = generation_dataset.select(range(10000, len(generation_dataset)))

        #we use the "train" partition to select demonstrations
        demonstration_dataset = select_binary_balanced_dataset(demonstration_dataset,
                                                                 lambda x: x["toxicity"] >= kwargs["label_threshold"],
                                                                 seed, splits["demonstration"])


        generation_dataset = select_binary_balanced_dataset(query_dataset, lambda x: x["toxicity"] >= kwargs["label_threshold"], seed, splits["generation"] // 2)

        return {"generation": reformat_google_civil_comments_for_inferencing(generation_dataset, demonstration_dataset, kwargs["label_threshold"], kwargs["template_name"])}
    elif (dataset_name == "unused_data"):
        # check if demonstrations and generation in split
        if "demonstration" not in splits:
            raise Exception("civil comments should have demonstrations")
        if "generation" not in splits:
            raise Exception("civil comments should have generation")

        # A jsonl file where each entry has a "parent" and "child" key
        generation_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        #use a smaller sample of total datafile since it is too large
        demonstration_dataset = generation_dataset.select(range(10000))
        query_dataset = generation_dataset.select(range(10000, 50000))

        def binary_eval_func(row):
            return row["tags"]["attributes"]["toxic_conversations__jigsaw_hatespeech_document_v2____label__toxic"][0][-1] >= kwargs["label_threshold"]

        # we use the "train" partition to select demonstrations
        demonstration_dataset = select_binary_balanced_dataset(demonstration_dataset,
                                                               binary_eval_func,
                                                               seed, splits["demonstration"])
        generation_dataset = select_binary_balanced_dataset(query_dataset, binary_eval_func, seed, splits["generation"] // 2)

        return {"generation": reformat_unused_comments_for_inferencing(generation_dataset, demonstration_dataset, binary_eval_func, kwargs["template_name"])}
    elif (dataset_name == "reddit"):
        # check for train splits
        if "train" not in splits:
            raise Exception("dynahate should have train split")
        ### setup the data, tokenizer, and preprocessing
        raw_dataset = read_dataset_to_hf(dataset_file)["train"]
        preprocessed_dataset = preprocess_conversation(raw_dataset, tokenizer, max_seq_len, seed=seed, num_proc=num_proc, use_loss_mask=use_loss_mask)
        preprocessed_dataset = preprocessed_dataset.select(range(splits["train"]))
        return {"train": preprocessed_dataset}
    elif (dataset_name == "dynahate"):
        # check for train and eval splits
        if "train" not in splits:
            raise Exception("dynahate should have train split")
        if "eval" not in splits:
            raise Exception("dynahate should have eval split")

        raw_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        def reformat_row(row):
            prompt = HATE_CLASSIFICATION_WITHOUT_LABEL.format(input=row["text"])
            label = DYNAHATE_LABELS[row["label"] == "hate"]
            return {"prompt": prompt,
                    "label": label}

        preprocessed_dataset = raw_dataset.map(reformat_row, batched=False)


        train_dataset = preprocessed_dataset.filter(lambda x: x["split"] == "train", batched=False, num_proc=num_proc)

        # we only want to evaluate on rounds 3 and 4
        eval_dataset = preprocessed_dataset.filter(lambda x: x["split"] == "test" and x["round.base"] - 2 > 0, batched=False, num_proc=num_proc)

        # using -1 means using the entire dataset
        if splits["train"] > 0:
            train_dataset = train_dataset.select(range(splits["train"]))
        if splits["eval"] > 0:
            eval_dataset = eval_dataset.select(range(splits["eval"]))


        # performs padding and tokenization
        def perform_tokenization(example):
            prompt_tokenized, label_tokenized = tokenize_input_output_pair(tokenizer, example["prompt"], example["label"])
            current_len = len(prompt_tokenized) + len(label_tokenized)

            if current_len > max_seq_len:
                example["skip"] = True
                example["input_ids"] = []
                example["attention_mask"] = []
                example["loss_mask"] = []
                return example

            new_input_id = prompt_tokenized + label_tokenized + [tokenizer.eos_token_id] * (max_seq_len - current_len)
            new_attention_mask = [1] * current_len + [0] * (max_seq_len - current_len)
            new_loss_mask = [0] * len(prompt_tokenized) + [1] * len(label_tokenized) + [0] * (max_seq_len - current_len)
            try:
                assert (len(new_input_id) == len(new_attention_mask) == len(new_loss_mask) == max_seq_len)
            except:
                import pdb
                pdb.set_trace()
            example["input_ids"] = new_input_id
            example["attention_mask"] = new_attention_mask
            example["loss_mask"] = new_loss_mask
            example["skip"] = False

            return example

        train_dataset = train_dataset.map(perform_tokenization, remove_columns=train_dataset.column_names,
                                          num_proc=num_proc)
        train_dataset = train_dataset.filter(lambda x: x["skip"] == False)

        def tokenize_evaluation(example):
            prompt_tokenized, _ = tokenize_input_output_pair(tokenizer, example["prompt"], "something")

            if (len(prompt_tokenized) > max_seq_len):
                example["skip"] = True
                example["input_ids"] = []
                example["attention_mask"] = []
                example["final_label"] = example["label"]
                example["round_info"] = example["round.base"]
                return example

            example["input_ids"] = prompt_tokenized
            example["attention_mask"] = [1] * len(prompt_tokenized)
            example["final_label"] = example["label"] == DYNAHATE_LABELS[True]
            example["round_info"] = example["round.base"]
            example["skip"] = False
            return example


        eval_dataset = eval_dataset.map(tokenize_evaluation, remove_columns=eval_dataset.column_names,
                                        num_proc=num_proc)
        eval_dataset = eval_dataset.filter(lambda x: x["skip"] == False)

        return {"train": train_dataset, "eval": eval_dataset}
    elif (dataset_name == "custom_hf_dataset"):
        # check if generation in split
        if "generation" not in splits:
            raise Exception("civil comments should have generation")

        generation_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        if splits["generation"] > 0:
            generation_dataset = generation_dataset.select(range(splits["generation"]))

        return {"generation": generation_dataset}
    else:
        raise ValueError(f"Unknown dataset: {dataset_file}")

