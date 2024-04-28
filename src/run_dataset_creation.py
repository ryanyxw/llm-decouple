import argparse
import os
import shutil
import json
import subprocess

from bs4 import BeautifulSoup as Soup
import re

import yaml

from src.modules.utils import confirm_with_user, load_config, prepare_folder, save_config, validate_inputs
from src.modules.data.dolma import load_dolma
from src.modules.data.reddit import read_lines_zst, read_lines_from_file
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm


def process_reddit(configs):

    # we choose to create conversational data
    if configs.create_conversations.do:

        if configs.create_conversations.create_untagged_conversations.do:
            exp_configs = configs.create_conversations.create_untagged_conversations


            # load the list of subreddits to not include
            with open(configs.input_blocked_subreddit_file, "r") as file:
                blocked_subreddits = file.read().split("\n")
            print("enter")
            collected_documents = 0
            pbar = tqdm(total=configs.documents_to_collect)

            master_dict = {}
            with open(os.path.join(exp_configs.output_untagged_directory, "untagged_conversations.jsonl"), "w") as out_file:
                idx = 0
                for line, _ in read_lines_zst(configs.input_rawdata_zst):
                    try:
                        obj = json.loads(line)

                        if (len(obj["body"]) > exp_configs.max_length):
                            continue
                        if (obj["subreddit"] in blocked_subreddits):
                            continue
                        if "t1" in obj["parent_id"] and obj["parent_id"] in master_dict:
                            write_obj_parent = {"text": master_dict[obj["parent_id"]], "id": obj["parent_id"], "source": "reddit",
                                                "pair_id": idx}
                            out_file.write(json.dumps(write_obj_parent) + "\n")

                            write_obj_child = {"text": obj["body"], "id": obj["name"], "source": "reddit",
                                               "pair_id": idx}
                            out_file.write(json.dumps(write_obj_child) + "\n")
                            idx += 1
                            pbar.update(1)
                        if not obj["no_follow"]:
                            master_dict[obj["name"]] = obj["body"]

                    except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                        print("error!")
                        import pdb
                        pdb.set_trace()

            # Copy configs to the output directory
            save_config(configs, os.path.join(exp_configs.output_untagged_directory, "configs.yaml"))

        # we tage the conversation using dolma library
        if configs.create_conversations.tag_conversations.do:
            yaml_dict = dict()
            yaml_dict["processes"] = configs.num_proc
            yaml_dict["experiment"] = configs.exp_name
            yaml_dict["documents"] = [os.path.join(configs.create_conversations.tag_conversations.input_untagged_directory, "untagged_conversations.jsonl")]
            yaml_dict["taggers"] = configs.create_conversations.tag_conversations.taggers

            # save the yaml file
            with open(os.path.join(configs.output_directory, "dolma_tag.yaml"), "w") as file:
                save_config(yaml_dict, file)

            command = ["dolma", "-c", os.path.join(configs.output_directory, 'dolma_tag.yaml'), "tag"]

            result = subprocess.run(command)

            if (result.returncode != 0):
                print("Error in tagging the conversations")
                return

            print("Tagging complete")

        if configs.select_tagged_conversations:
            exp_configs = configs.select_tagged_conversations

            parent_utterance = None
            parent_tagged = None
            count = -1

            attribute_prefix = configs.exp_name
            attribute_key_map = {"english": f"{attribute_prefix}__ft_lang_id_en_doc_v2__en", "toxic": f"{attribute_prefix}__jigsaw_hatespeech_document_v2____label__toxic", "nsfw": f"{attribute_prefix}__jigsaw_nsfw_document_v1____label__nsfw"}

            output_fn = os.path.join(exp_configs.output_tagged_directory, "filtered_tagged_conversations.jsonl")
            orig_utterance_fn = os.path.join(exp_configs.input_untagged_directory, "untagged_conversations.jsonl")
            tagged_attribute_fn = os.path.join(exp_configs.input_tagged_directory, "untagged_conversations.jsonl")
            with open(output_fn, "w") as out_file:
                for orig_utterance, tagged_utterance in zip(read_lines_from_file(orig_utterance_fn), read_lines_from_file(tagged_attribute_fn)):
                    count += 1
                    if (count % 2 == 0):
                        parent_utterance = json.loads(orig_utterance)
                        parent_tagged = json.loads(tagged_utterance)
                        continue
                    child_utterance = json.loads(orig_utterance)
                    child_tagged = json.loads(tagged_utterance)

                    parent_english = parent_tagged["attributes"][attribute_key_map["english"]][0][2]
                    child_english = child_tagged["attributes"][attribute_key_map["english"]][0][2]
                    parent_toxic = parent_tagged["attributes"][attribute_key_map["toxic"]][0][2]
                    child_toxic = child_tagged["attributes"][attribute_key_map["toxic"]][0][2]
                    parent_nsfw = parent_tagged["attributes"][attribute_key_map["nsfw"]][0][2]
                    child_nsfw = child_tagged["attributes"][attribute_key_map["nsfw"]][0][2]

                    #filter out non-english conversations
                    if (parent_english < configs.english_lowerbound or child_english < configs.english_lowerbound):
                        continue

                    if (not configs.parent_invert):
                        # choose conversations where parent is toxic or nsfw and child is not toxic and nsfw
                        if (parent_toxic >= configs.parent_toxic_lowerbound or parent_nsfw >= configs.parent_nsfw_lowerbound):
                            if (child_toxic <= configs.child_toxic_upperbound and child_nsfw <= configs.child_nsfw_upperbound):
                                out_file.write(json.dumps({"parent": parent_utterance, "child": child_utterance, "parent_tagged": parent_tagged, "child_tagged": child_tagged}) + "\n")
                    else:
                        # choose conversations where parent is not toxic or nsfw and child is not toxic and nsfw
                        if (parent_toxic <= configs.parent_toxic_lowerbound and parent_nsfw <= configs.parent_nsfw_lowerbound):
                            if (child_toxic <= configs.child_toxic_upperbound and child_nsfw <= configs.child_nsfw_upperbound):
                                out_file.write(json.dumps({"parent": parent_utterance, "child": child_utterance, "parent_tagged": parent_tagged, "child_tagged": child_tagged}) + "\n")
    if configs.extract_comments.do:
        pass
        # else:
        # # print("dab")
        # # discard if submission is shorter than 400 characters or longer than 40,000 characters or has less than 3 upvotes
        #
        # if (len(obj["body"]) < 500 or len(obj["body"]) > 40000 or obj["score"] < 3):
        #     continue
        #
        # # discard if subreddit is in the blocked list
        # if (obj["subreddit"] in blocked_subreddits):
        #     continue
        #
        # out_file.write(json.dumps({"text": obj["body"], "id": obj["id"], "source": "reddit"}) + "\n")
        # collected_documents += 1
        # pbar.update(1)
        # if (collected_documents > configs.documents_to_collect):
        #     break


    print("yay!")



def main(args):
    print("yay!")
    #load the config file
    print("loading config file...")
    configs = load_config(args.config_file)

    #set the args to be the configs
    for key, value in args.__dict__.items():
        configs[key] = value

    # target exists and destination does not exist, creating output directories
    validate_inputs(configs)

    print("executing command...")

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