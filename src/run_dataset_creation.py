import argparse
import os
import shutil
import json
from bs4 import BeautifulSoup as Soup
import re
from utils import confirm_with_user, load_config, prepare_folder
from data.dolma import load_dolma
from data.reddit import read_lines_zst
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

def validate_inputs(configs):
    if (configs.mode == "4chan"):
        if (os.path.exists(configs.cached_hate_data)):
            print("using cached dataset")
        else:
            print("cache not found. preparing dataset")

            if not os.path.exists(configs.hate_dataset):
                raise ValueError(f"hate_dataset path {configs.hate_dataset} does not exist")

        #check the output dataset
        if (os.path.exists(configs.output_dataset)):
            if (not confirm_with_user(f"output {configs.output_dataset} exists, do you want to continue?")):
                raise ValueError("output exists")

        #check the test dataset
        if (os.path.exists(configs.test_dataset)):
            if (not confirm_with_user(f"test {configs.test_dataset} exists, do you want to continue?")):
                raise ValueError("test exists")

        prepare_folder(configs.output_dataset)
        prepare_folder(configs.test_dataset)
    if (configs.mode == "reddit"):
        if not os.path.exists(configs.hate_dataset):
            raise ValueError(f"hate_dataset path {configs.hate_dataset} does not exist")
        if os.path.exists(configs.hate_output_dataset):
            if not confirm_with_user(f"output {configs.hate_output_dataset} exists, do you want to continue?"):
                raise ValueError("output exists")
        prepare_folder(configs.hate_output_dataset)



def process_4chan(configs):
    def fourchan_generator(file):
        """generator for reading 4chan dataset line by line"""
        while True:
            line = file.readline()
            if not line:
                break
            #posts represents the entire thread
            posts = json.loads(line)["posts"]
            for post in posts:
                if ("com" in post and "perspectives" in post):
                    text = Soup(post["com"], "html.parser").text
                    text = re.sub(r'>>\d+', ' ', text)
                    text = text.replace('>', ' ')
                    #remove extra spaces
                    text = re.sub(r'\s+', ' ', text).strip()
                    yield text.strip(), post["perspectives"]

    #if we don't have the cached data, we need to create it
    if (not os.path.exists(configs.cached_hate_data)):
        with open(configs.hate_dataset, 'r') as file, open(configs.hate_output_dataset, 'w') as out_file:
            for text, perspectives in fourchan_generator(file):
                try:
                    if ((perspectives["SEVERE_TOXICITY"] > 0.8) and (len(text) > 1500)):
                        out_file.write(str(perspectives["SEVERE_TOXICITY"]) + "," + text + "\n")
                except:
                    continue
        configs.__setattr__("cached_hate_data", configs.hate_output_dataset)

    # we load the cached and original datasets

    def cached_dataset_generator():
        """generator for reading a jsonl file with toxicityscore, text"""
        with open(configs.cached_hate_data, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                temp_process = line.strip().split(",")
                toxicity_score = float(temp_process[0])
                text = ",".join(temp_process[1:])
                yield {"toxicity_score": toxicity_score, "text": text}

    hate_dataset = Dataset.from_generator(cached_dataset_generator).train_test_split(test_size=0.1, seed=configs.seed)
    hate_dataset_train = hate_dataset["train"]
    hate_dataset_test = hate_dataset["test"]

    # load the pretraining dataset
    pretrain_dataset = load_dolma(configs.base_dataset)["train"]

    columns_to_keep = ["text"]
    pretrain_dataset = pretrain_dataset.remove_columns([col for col in pretrain_dataset.column_names if col not in columns_to_keep])
    hate_dataset_train = hate_dataset_train.remove_columns([col for col in hate_dataset_train.column_names if col not in columns_to_keep])


    final_dataset = concatenate_datasets([pretrain_dataset, hate_dataset_train]).shuffle(seed=configs.seed)

    #save datasets to respective directories
    final_dataset.to_json(configs.output_dataset, orient="records", lines=True)
    hate_dataset_test.to_json(configs.test_dataset, orient="records", lines=True)

    print("success!")

def process_reddit(configs):
    # load the list of subreddits to not include
    with open(configs.blocked_subreddit_file, "r") as file:
        blocked_subreddits = file.read().split("\n")
    print("enter")
    collected_documents = 0
    pbar = tqdm(total=configs.documents_to_collect)
    with open(configs.hate_output_dataset, "w") as out_file:
        for line, _ in read_lines_zst(configs.hate_dataset):
            try:
                obj = json.loads(line)

                # print("dab")
                # discard if submission is shorter than 400 characters or longer than 40,000 characters or has less than 3 upvotes
                if (len(obj["body"]) < 500 or len(obj["body"]) > 40000 or obj["score"] < 3):
                    continue

                # discard if subreddit is in the blocked list
                if (obj["subreddit"] in blocked_subreddits):
                    continue

                out_file.write(json.dumps({"text": obj["body"], "id": obj["id"], "source":"reddit"}) + "\n")
                collected_documents += 1
                pbar.update(1)
                if (collected_documents > configs.documents_to_collect):
                    break

            except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                print("error!")
                import pdb
                pdb.set_trace()
    print("yay!")



def main(args):
    print("yay!")
    #load the config file
    print("loading config file...")
    configs = load_config(args.config_file)

    #set the args to be the configs
    for key, value in args.__dict__.items():
        configs.__setattr__(key, value)

    # target exists and destination does not exist, creating output directories
    validate_inputs(configs)

    print("executing command...")

    if configs.mode == "4chan":
        process_4chan(configs)
    if configs.mode == "reddit":
        process_reddit(configs)

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