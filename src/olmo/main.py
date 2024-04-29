import argparse
import json
import os
import shutil

import numpy as np

from src.modules.data.load import read_dataset_to_hf
from src.modules.utils import load_config, prepare_folder, confirm_with_user
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import concatenate_datasets, load_dataset

import hf_olmo

from tqdm import tqdm
from omegaconf import OmegaConf


def olmo_prepare_data(configs):
    tokenizer = AutoTokenizer.from_pretrained(configs.preprocess.tokenizer, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # prepares the conversation. Adds a newline between components by default
    def prepare_conversation(text1, text1_role, text2, text2_role):
        input_ids = []
        label_mask = [] #True if you want to train on them, False if you don't

        text1_prefix_tokens = tokenizer.encode(f"{text1_role.strip()}\n", add_special_tokens=False)
        text1_tokens = tokenizer.encode(f"{text1.strip()}\n", add_special_tokens=False)
        text2_prefix_tokens = tokenizer.encode(f"{text2_role.strip()}\n", add_special_tokens=False)
        text2_tokens = tokenizer.encode(f"{text2.strip()}\n", add_special_tokens=False)

        # shed off equally
        if len(text1_tokens) + len(text2_tokens) + len(text1_prefix_tokens) + len(text2_prefix_tokens) > configs.preprocess.max_seq_len:
            excess_len = len(text1_tokens) + len(text2_tokens) + len(text1_prefix_tokens) + len(text2_prefix_tokens) - configs.preprocess.max_seq_len + 1
            diff = len(text1_tokens) - len(text2_tokens)
            if (diff >= 0 and diff >= excess_len):
                # we shave off excess from text1_tokens only
                text1_tokens = text1_tokens[excess_len:]
            elif (diff < 0 and -1 * diff >= excess_len):
                #we shave off excess from text2_tokens only
                text2_tokens = text2_tokens[:-excess_len]
            elif (diff >= 0 and diff < excess_len):
                #first shave off from text1_tokens, the shave off equally
                text1_tokens = text1_tokens[diff:]
                excess_len = excess_len - diff
                text1_tokens = text1_tokens[excess_len // 2:]
                text2_tokens = text2_tokens[:-(excess_len // 2)]
            else:
                #first shave off from text2_tokens, then shave off equally
                diff = -1 * diff
                text2_tokens = text2_tokens[:-diff]
                excess_len = excess_len - diff
                text1_tokens = text1_tokens[excess_len // 2:]
                text2_tokens = text2_tokens[:-(excess_len // 2)]

        input_ids = text1_prefix_tokens + text1_tokens + text2_prefix_tokens + text2_tokens
        label_mask = len(text1_prefix_tokens) * [False] + len(text1_tokens) * [False] + len(text2_prefix_tokens) * [True] + len(text2_tokens) * [True]
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(label_mask)

        if len(input_ids) < configs.preprocess.max_seq_len:
            pad_len = configs.preprocess.max_seq_len - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            label_mask += [False] * pad_len
            attention_mask += [0] * pad_len

        assert len(input_ids) == len(label_mask) and len(input_ids) == configs.preprocess.max_seq_len

        return {"input_ids": input_ids, "label_mask": label_mask, "attention_mask": attention_mask}

    if configs.preprocess.reddit_preprocess:
        configs_exp = configs.preprocess.reddit_preprocess
        hf_toxic = read_dataset_to_hf(configs_exp.tagged_file_toxic)["train"]
        hf_nontoxic = read_dataset_to_hf(configs_exp.tagged_file_nontoxic)["train"].shuffle(configs.seed).select(range(len(hf_toxic) * configs_exp.nontoxic_proportion))


        def process_input(input):
            parent = input["parent"]
            parent_tagged = input["parent_tagged"]
            first_prefix = f"<|{parent_tagged['id']}|>"
            child = input["child"]
            child_tagged = input["child_tagged"]
            second_prefix = f"<|{child_tagged['id']}|>"
            return prepare_conversation(parent["text"], first_prefix, child["text"], second_prefix)

        hf_toxic_tokenized = hf_toxic.map(process_input,
                                batched=False,
                                remove_columns=hf_toxic.column_names,
                                num_proc=configs.num_proc)

        hf_nontoxic_tokenized = hf_nontoxic.map(process_input,
                                      batched=False,
                                      remove_columns=hf_nontoxic.column_names,
                                      num_proc=configs.num_proc)


        combined_tokenized = concatenate_datasets([hf_toxic_tokenized, hf_nontoxic_tokenized]).shuffle(configs.seed)

        if configs_exp.save_as_npy:
            total_tokens = configs.preprocess.max_seq_len * len(combined_tokenized)

            input_ids_file = np.memmap(
                os.path.join(configs_exp.output_directory, "input_ids.npy"), dtype=np.uint16, mode="w+", shape=(total_tokens,)
            )
            label_mask_file = np.memmap(
                os.path.join(configs_exp.output_directory, "label_mask.npy"), dtype=np.bool_, mode="w+", shape=(total_tokens,)
            )

            offset = 0
            for ex in tqdm(combined_tokenized, total=len(combined_tokenized)):
                ex_len = len(ex["input_ids"])
                input_ids_file[offset : offset + ex_len] = ex["input_ids"]
                label_mask_file[offset : offset + ex_len] = ex["label_mask"]
                offset += ex_len

            input_ids_file.flush()
            label_mask_file.flush()
        if configs_exp.save_as_hf:
            #change the column name from label_mask to loss_mask
            combined_tokenized = combined_tokenized.rename_column("label_mask", "loss_mask")
            combined_tokenized.save_to_disk(configs_exp.output_directory)

        if not os.path.exists(os.path.join(configs_exp.output_directory, "configs")):
            os.makedirs(os.path.join(configs_exp.output_directory, "configs"))
        with open(os.path.join(configs_exp.output_directory, "configs", "configs.json"), "w") as f:
            OmegaConf.save(configs.preprocess, f)


def olmo_inference(configs):
    dataset = load_dataset(configs.inference.dataset_name)["train"]

    small_dataset = dataset.shuffle(seed=configs.seed).select(range(configs.inference.num_examples))

    model = AutoModelForCausalLM.from_pretrained(configs.inference.model_path, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(configs.inference.tokenizer)


    with open(os.path.join(configs.inference.output_directory, "generation.jsonl"), "w") as f:
        progress_bar = tqdm(total=configs.inference.num_examples)
        for example in small_dataset:
            prompt = example["prompt"]["text"]
            model_inputs = tokenizer([prompt], return_tensors="pt", return_token_type_ids=False).to("cuda")
            generated_ids = model.generate(**model_inputs, max_length=configs.inference.max_length)
            final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            f.write(json.dumps({"prompt": prompt, "completion": final}) + "\n")
            progress_bar.update(1)


def main(args):
    print("yay!")
    # load the config file
    print("loading config file...")
    configs = load_config(args.config_file)

    # set the args to be the configs
    for key, value in args.__dict__.items():
        configs[key] = value

    print("Setting up output directories")

    if configs.preprocess.do:
        if configs.preprocess.reddit_preprocess.do:
            if not os.path.exists(configs.preprocess.reddit_preprocess.tagged_file_toxic):
                raise FileNotFoundError(f"File {configs.preprocess.reddit_preprocess.tagged_file_toxic} not found")
            if not os.path.exists(configs.preprocess.reddit_preprocess.tagged_file_nontoxic):
                raise FileNotFoundError(f"File {configs.preprocess.reddit_preprocess.tagged_file_nontoxic} not found")
            if os.path.exists(configs.preprocess.reddit_preprocess.output_directory):
                message = f"out_dir {configs.preprocess.reddit_preprocess.output_directory} already exists. Delete? "
                if confirm_with_user(message):
                    shutil.rmtree(configs.preprocess.reddit_preprocess.output_directory)
                else:
                    raise ValueError(f"out_dir {configs.configs.preprocess.reddit_preprocess.output_directory} already exists")
            prepare_folder(configs.preprocess.reddit_preprocess.output_directory, isFile = False)
    if configs.inference.do:
        if not os.path.exists(configs.inference.model_path):
            raise FileNotFoundError(f"File {configs.inference.model_path} not found")
        if os.path.exists(configs.inference.output_directory):
            message = f"out_file {configs.inference.output_directory} already exists. Delete? "
            if confirm_with_user(message):
                shutil.rmtree(configs.inference.output_directory)
            else:
                raise ValueError(f"out_file {configs.inference.output_directory} already exists")
        prepare_folder(configs.inference.output_directory, isFile = False)



    if configs.preprocess.do:
        olmo_prepare_data(configs)

    if configs.inference.do:
        olmo_inference(configs)







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


if __name__ == "__main__":
    args = parse_args()
    main(args)