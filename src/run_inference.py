import argparse
import os
import shutil
import csv
import torch
from utils import confirm_with_user
from tqdm import tqdm
from data.dolma import load_dolma
from modeling.modeling_utils import setup_model, setup_tokenizer
from modeling.inference import obtain_logit, calculate_perplexity, calculate_loss_across_tokens


def validate_inputs(args):
    if not os.path.exists(args.model_dir):
        raise ValueError(f"model_dir path {args.model_dir} does not exist")

    if not os.path.exists(args.query_dataset):
        raise ValueError(f"query_dataset path {args.query_dataset} does not exist")

    if os.path.exists(args.out_dir):
        message = f"out_dir {args.out_dir} already exists. Delete? "
        if confirm_with_user(message):
            shutil.rmtree(args.out_dir)
        else:
            raise ValueError(f"out_dir {args.out_dir} already exists")

    os.makedirs(args.out_dir)

def process_dataset(dataset, tokenizer):
    """Tokenize the dataset using the tokenizer. Filters the dataset such that only examples containing the word "hate" are returned.
    dataset: HuggingFace dataset
    tokenizer: HuggingFace tokenizer"""
    print("entered!")

    def tokenize_function(dataset):
        return tokenizer(dataset["text"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=16, remove_columns=["text"])
    tokenized_datasets = tokenized_datasets.filter(lambda example: 5465 in example["input_ids"], num_proc=16)
    return tokenized_datasets


def main(args):
    # target exists and destination does not exist, creating output directories
    validate_inputs(args)

    print("executing command...")

    model = setup_model(args.model_dir).to("cuda")

    dataset = load_dolma(args.query_dataset)["train"]

    tokenizer = setup_tokenizer("gpt2")

    tokenized_dataset = process_dataset(dataset, tokenizer)

    tot_arr = []

    with open(os.path.join(args.out_dir, "dataset.json"), "w") as f:
        writer = csv.writer(f)

        for i in tqdm(range(len(tokenized_dataset))):
            hate_idx = tokenized_dataset[i]["input_ids"].index(5465)

            # take +-100 tokens around the word "hate"
            # if hate_idx < 100:
            #     continue
            start_idx = max(0, hate_idx - 150)
            # end_idx = min(len(tokenized_dataset[i]["input_ids"]), hate_idx + 100)
            input_ids = tokenized_dataset[i]["input_ids"][start_idx:hate_idx]

            logits = obtain_logit(model, torch.tensor(input_ids).to("cuda"))

            # # we take the logit of the word "hate"
            # hate_logit = logits[0][-1]
            perplexity = calculate_perplexity(calculate_loss_across_tokens(logits, torch.tensor(input_ids), shift=True))

            #check for nan
            if (perplexity != perplexity):
                print("perplexity is nan! ")
                # import pdb
                # pdb.set_trace()
                continue
            writer.writerow([perplexity.item()])
            tot_arr += [perplexity.item()]

    print(f"average perplexity is {sum(tot_arr) / len(tot_arr)}")
    print("yay!")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="(input) path to the model"
    )

    parser.add_argument(
        "--query_dataset",
        type=str,
        required=True,
        help="(output) path to the destination folder"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="(input) token to mask out (in GPT-2 INPUT_ID)"
    )

    parser.add_argument(
        "--NEOX_DIR",
        type=str,
        required=True,
        help="(input) path to the Neox directory"
    )

    parser.add_argument(
        "--DATA_DIR",
        type=str,
        required=True,
        help="(input) path to the data directory"
    )


    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)