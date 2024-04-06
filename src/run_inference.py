import argparse
import os
import shutil
import csv
import torch
from datasets import load_dataset

from utils import confirm_with_user, load_config
from tqdm import tqdm
from data.dolma import load_dolma
from modeling.modeling_utils import setup_model, setup_tokenizer
from modeling.inference import obtain_logit, calculate_perplexity, calculate_loss_across_tokens
from data.data_utils import kmp, setup_dataset_hf
from torch.nn.functional import softmax



def validate_inputs(configs):
    if not os.path.exists(configs.model_dir):
        raise ValueError(f"model_dir path {configs.model_dir} does not exist")

    if not os.path.exists(configs.query_dataset):
        raise ValueError(f"query_dataset path {configs.query_dataset} does not exist")

    if os.path.exists(configs.out_dir):
        message = f"out_dir {configs.out_dir} already exists. Delete? "
        if confirm_with_user(message):
            shutil.rmtree(configs.out_dir)
        else:
            raise ValueError(f"out_dir {configs.out_dir} already exists")

    os.makedirs(configs.out_dir)

def process_dataset(configs, dataset, tokenizer):
    """Tokenize the dataset using the tokenizer. Filters the dataset such that only examples containing the word "hate" are returned.
    dataset: HuggingFace dataset
    tokenizer: HuggingFace tokenizer"""
    print("entered!")

    def tokenize_function(dataset):
        return tokenizer(dataset["text"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=configs.workers, remove_columns=["text"])

    def filter_hate_indices(example):

        example_input_ids = example["input_ids"]
        inds = []
        for pattern in configs.mask_target:
            inds += [kmp(example_input_ids, pattern)]

        # get the corresponding indices to mask
        flattened_inds = [item for sublist in inds for item in sublist]
        return len(flattened_inds) > 0

    def get_hate_indices(examples):
        input_ids_arr = []
        first_hate_indices_arr = []

        for example_input_ids in examples["input_ids"]:
            inds = []
            for pattern in configs.mask_target:
                inds += [kmp(example_input_ids, pattern)]

            # get the corresponding indices to mask
            flattened_inds = [item for sublist in inds for item in sublist]
            input_ids_arr += [example_input_ids]
            first_hate_indices_arr += [sorted(flattened_inds)[0]]

        return {"first_hate_indices": first_hate_indices_arr}


    # tokenized_datasets = tokenized_datasets.map(get_hate_indices, batched=False, num_proc=configs.workers)
    tokenized_datasets = tokenized_datasets.filter(filter_hate_indices)

    tokenized_datasets = tokenized_datasets.map(get_hate_indices, batched=True)


    return tokenized_datasets


def main(args):
    # load the config file
    print("loading config file...")
    configs = load_config(args.config_file)

    # set the args to be the configs
    for key, value in args.__dict__.items():
        configs.__setattr__(key, value)

    # target exists and destination does not exist, creating output directories
    validate_inputs(configs)

    print("executing command...")

    model = setup_model(configs.model_dir).to(torch.device("cuda"))

    print("started")

    dataset = load_dataset("json", data_files = configs.query_dataset)["train"]

    print("finished!")

    tokenizer = setup_tokenizer("gpt2")

    tokenized_dataset = process_dataset(configs, dataset, tokenizer)

    tot_arr = []

    with open(os.path.join(configs.out_dir, "perplexity.csv"), "w") as f_perp, open(os.path.join(configs.out_dir, "probability.csv"), "w") as f_prob:
        writer_prob = csv.writer(f_prob)
        writer_perp = csv.writer(f_perp)

        for i in tqdm(range(len(tokenized_dataset))):
            hate_idx = tokenized_dataset[i]["first_hate_indices"]

            # take +-100 tokens around the word "hate"
            if hate_idx < configs.context_given or hate_idx + configs.query_tokens > len(tokenized_dataset[i]["input_ids"]):
                continue

            # end_idx = min(len(tokenized_dataset[i]["input_ids"]), hate_idx + 100)
            input_ids = tokenized_dataset[i]["input_ids"][hate_idx - configs.context_given:hate_idx + configs.query_tokens]

            logits = obtain_logit(model, torch.tensor(input_ids).to(torch.device("cuda")))

            save_range = configs.query_tokens + configs.prior_tokens

            # We get the perplexity of the word "hate" across the last 15 positions
            loss = calculate_loss_across_tokens(logits, torch.tensor(input_ids), shift=True)[0]
            perplexity_positions = [calculate_perplexity(loss[i:i+1]) for i in range(len(loss) - save_range, len(loss))]
            writer_perp.writerow(perplexity_positions)


            # #We now get the probability of the correct token across 15 positions shifted to the left by 1
            # probs = softmax(logits[0], dim=-1)[:, -save_range-1:-1, :]
            #
            # original_labels = torch.tensor(input_ids[-save_range:])
            #
            # probs_to_prediction = probs[0][torch.arange(len(probs[0])), original_labels]
            #
            #
            # import pdb
            # pdb.set_trace()
            # writer_prob.writerow(probs_to_prediction.tolist())




    # print(f"average perplexity is {sum(tot_arr) / len(tot_arr)}")
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