import argparse
import os
import shutil
import json
import pickle
import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

from API_KEYS import PERSPECTIVE_API_KEY
from src.modules.data.data_utils import load_tokenizer
from src.modules.data.load import read_lines_from_file
from src.modules.data.process import process_with_multiprocessing
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs
from datasets import Dataset, concatenate_datasets, load_dataset
from collections import Counter
from tqdm import tqdm



def process_get_n_grams(configs):
    """counts the top occuring n-grams in the dataset"""

    def process_line(line):
        return ",".join(line.split(",")[1:])

    #define a counter
    counter_arr = []
    for n in configs.n_gram:
        counter_arr += [Counter()]

    tokenizer = load_tokenizer(configs.tokenizer_path)

    # with open(configs.input_dataset, 'r') as file:

    for line in tqdm(read_lines_from_file(configs.input_dataset, process_line)):
        line = tokenizer.tokenize(line)
        for i, n in enumerate(configs.n_gram):
            word_arr = [tokenizer.convert_tokens_to_string(line[j:j+n]) for j in range(len(line) - n + 1)]
            counter_arr[i].update(word_arr)

    with open(configs.output_summary, 'wb') as file:
        pickle.dump(counter_arr, file)


def process_count_tokens(configs):
    tokenizer = load_tokenizer(configs.tokenizer_path)

    dataset = load_dataset("json", data_files=configs.input_dataset)

    tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"]))


    #Count how many tokens are in the dataset
    num_tokens = 0
    for line in tokenized_dataset["train"]:
        num_tokens += len(line["input_ids"])

    print(f"number of tokens in dataset: {num_tokens}")

def process_graph_perplexity(configs):

    #load the csv file
    csv_files = []
    for file in configs.perplexity_file:
        csv_files.append(np.log(pd.read_csv(file, header=None)).mean(axis=0))

    # Creating a figure with 6 subplots (3x2 grid)
    fig, axs = plt.subplots(1, 5, figsize=(40, 10), sharey=True)  # sharey ensures all plots share the same y-axis scale

    # Plot each subplot
    for i, ax in enumerate(axs.flat):
        sns.lineplot(x = np.arange(15), y=csv_files[i].values, ax=ax)
        ax.set_title(f'Plot {i * 0.25} masked out')
        ax.set_xlabel('Index of token')
        ax.set_ylabel('Loss')
        # ax.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()

    # sns.lineplot(x = np.arange(15), y=averaged_loss.values)
    # plt.title(f'{configs.perplexity_file}')
    # plt.xlabel('token position')
    # plt.ylabel('Loss')
    plt.savefig(configs.output_file)

def test_olmo(configs):

    from transformers import AutoModelForCausalLM, AutoTokenizer

    olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")

    message = ["Language modeling is "]
    inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
    response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])

def use_perspective_api(configs):
    from googleapiclient import discovery
    import json

    def get_score_with_delay(text):
        import time
        time.sleep(1)
        return get_score(text)
    def get_score(text):

        API_KEY = PERSPECTIVE_API_KEY

        client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}},
            'languages': ["en"],
        }

        response = client.comments().analyze(body=analyze_request).execute()

        score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        return score

    for fn in configs.in_fns:
        input_jsonl = pd.read_json(fn, lines=True)
        print("score for: " + fn)
        toxicity_scores = []
        #get the parent of the input fn
        out_fn = os.path.join(os.path.dirname(fn), configs.out_fn)
        with open(out_fn, 'w') as file:
            for i in tqdm(range(min(len(input_jsonl), configs.max_lines))):
                try:
                    score = get_score_with_delay(input_jsonl["completion"][i])
                    toxicity_scores.append(score)
                    file.write(json.dumps({"toxicity": score, "text": input_jsonl["completion"][i], "prompt": input_jsonl["prompt"][i], }) + "\n")
                except Exception as e:
                    print(e)
                    import pdb
                    pdb.set_trace()
        print(f"average toxicity score: {np.mean(toxicity_scores)}")

def evaluate_generation(configs):
    def process_func(line):
        return json.loads(line)


    # for line in read_lines_from_file(configs.input_file, process_func):
    #     if line["label"]:
    #         if "no" in line["completion"].lower() or "goes against" in line["completion"].lower():
    #             correct_count += 1
    #     else:
    #         if "yes" in line["completion"].lower():
    #             correct_count += 1
    #     tot_count += 1

    for input_file in configs.in_files:
        correct_count = 0
        tot_count = 0
        print(f"evaluating {input_file}")
        for line in read_lines_from_file(input_file, process_func):
            if line["label"] and line["completion"]:
                correct_count += 1
            elif not line["label"] and not line["completion"]:
                correct_count += 1
            tot_count += 1
        print(f"accuracy: {correct_count/tot_count}")
        print(f"total count: {tot_count}")

def rid_english(configs):
    with open(configs.output_file, "w") as file:

        for line in read_lines_from_file(configs.input_file, lambda x: json.loads(x)):
            if line["tags"]["attributes"]["toxic_conversations__ft_lang_id_en_doc_v2__en"][0][-1] < 0.5:
                continue
            file.write(json.dumps(line) + "\n")

def process_func(x):
    return x

class BatchDatasetWrapperForOlmo():
    """
    A wrapper for a dataset that returns batches. All indexes are in terms of batches
    """
    def __init__(self, dataset, start_seq, end_seq, global_indices):
        self.start_seq = start_seq
        self.end_seq = end_seq
        self.dataset = dataset
        self.global_indices = global_indices
    def __getitem__(self, i):
        batch_i = self.global_indices[self.start_seq + i]
        return self.dataset[batch_i]["input_ids"].tolist()

    def __len__(self):
        return self.end_seq - self.start_seq

import pickle

def is_serializable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError) as e:
        print(f"Serialization failed: {e}")
        return False


def probe_olmo_training(configs):
    import numpy as np
    from cached_path import cached_path

    from olmo.config import TrainConfig
    from olmo.data import build_memmap_dataset

    # Update these paths to what you want:
    data_order_file_path = cached_path(configs.data_order_cached_path)
    train_config_path = configs.train_config_path

    cfg = TrainConfig.load(train_config_path)
    dataset = build_memmap_dataset(cfg, cfg.data)
    batch_size = cfg.global_train_batch_size
    global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)

    import pdb
    pdb.set_trace()

    batched_dataset = BatchDatasetWrapperForOlmo(dataset, configs.start_batch, configs.end_batch, global_indices)

    output_fn = os.path.join(configs.output_dir, "output.jsonl")
    error_fn = os.path.join(configs.output_dir, "error.jsonl")

    process_with_multiprocessing(process_func, batched_dataset, output_fn, error_fn, num_proc=configs.num_proc)
    # # Get all 2048 x 2048 token IDs in the first batch.
    # batched_dataset = BatchDatasetWrapperForOlmo(dataset, configs.start_batch, configs.end_batch, batch_size, global_indices)
    #
    # process_with_multiprocessing(lambda x: x, batched_dataset, configs.output_dir, num_proc=2)


def main(args):
    #load the config file
    print("loading config file...")
    configs = load_config(args.config_file)

    #set the args to be the configs
    for key, value in args.__dict__.items():
        configs.__setattr__(key, value)

    # target exists and destination does not exist, creating output directories
    validate_inputs(configs)

    print("executing command...")

    if configs.mode == "get_n_grams":
        process_get_n_grams(configs)
    elif configs.mode == "count_tokens":
        process_count_tokens(configs)
    elif configs.mode == "graph_perplexity":
        process_graph_perplexity(configs)
    elif configs.mode == "test_olmo":
        test_olmo(configs)
    elif configs.mode == "perspective":
        use_perspective_api(configs)
    elif configs.mode == "evaluate_generations":
        evaluate_generation(configs)
    elif configs.mode == "rid_english":
        rid_english(configs)
    elif configs.mode == "probe_olmo_training":
        probe_olmo_training(configs)

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