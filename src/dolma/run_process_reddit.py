import argparse
import json
import os

from tqdm import tqdm

from src.modules.data.load import read_lines_zst, read_lines_from_file
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config, execute_shell_command


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

    #we are extracting from the zst file (that was torrented) to documents while performing simple filtering according to dolma
    if configs.extract_from_zst_to_documents.do:
        exp_configs = configs.extract_from_zst_to_documents

        # load the list of subreddits to not include
        with open(configs.input_blocked_subreddit_file, "r") as file:
            blocked_subreddits = file.read().split("\n")
            blocked_subreddits = [subreddit.strip() for subreddit in blocked_subreddits]
        print("enter")


        master_dict = {}
        with open(exp_configs.output_documents_file, "w") as out_file:
            tqdm_pbar = tqdm()
            for line, _ in read_lines_zst(exp_configs.input_zst_file):
                try:
                    tqdm_pbar.update(1)
                    obj = json.loads(line)

                    text_key = "body" if "body" in obj else "selftext"

                    #filter for min_length
                    if (len(obj[text_key]) < exp_configs.min_document_length):
                        continue
                    if obj["score"] < exp_configs.min_upvotes:
                        continue

                    is_bad_subreddit = obj["subreddit"].strip() in blocked_subreddits

                    write_obj = {"text": obj[text_key],
                                 "source": "reddit",
                                 "id": obj["id"],
                                 "is_bad_subreddit": is_bad_subreddit}

                    out_file.write(json.dumps(write_obj) + "\n")


                except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                    import pdb
                    pdb.set_trace()
                    print("error!")
                    print(e)


        if exp_configs.compress_to_zst:
            print("about to compress file to zst! ")
            command = f"zstd -f --compress {exp_configs.output_documents_file}"
            execute_shell_command(command)
            print("compression complete")

    # we tage the conversation using dolma library
    if configs.tag_conversations.do:
        yaml_dict = dict()
        yaml_dict["processes"] = configs.num_proc
        yaml_dict["experiment"] = configs.exp_name
        yaml_dict["documents"] = [configs.tag_conversations.in_documents_file]
        yaml_dict["taggers"] = configs.tag_conversations.taggers

        # save the yaml file
        with open(os.path.join(configs.out_dir, "dolma_tag.yaml"), "w") as file:
            save_config(yaml_dict, file)

        command = f"dolma -c {os.path.join(configs.out_dir, 'dolma_tag.yaml')} tag"

        print("Tagging conversations...")

        execute_shell_command(command)

        print("Tagging complete")

    # we extract based on tagged values
    if configs.filter_tags_and_prepare.do:
        exp_configs = configs.filter_tags_and_prepare

        assert(len(exp_configs.orig_documents) == len(exp_configs.tag_files))

        attribute_key_map = {"english": f"{configs.exp_name}__ft_lang_id_en_doc_v2__en",
                             "toxic": f"{configs.exp_name}__jigsaw_hatespeech_document_v2____label__toxic",
                             "nsfw": f"{configs.exp_name}__jigsaw_nsfw_document_v1____label__nsfw"}


        for i in range(len(exp_configs.orig_documents)):
            orig_document = exp_configs.orig_documents[i]
            tagged_document = exp_configs.tag_files[i]

            print(f"orig_document: {orig_document}")
            p_bar = tqdm()

            output_file = os.path.basename(orig_document).split(".")[0] + f"_eng{int(exp_configs.min_english_score * 100)}_toxic{int(exp_configs.min_toxic_score * 100)}_nsfw{int(exp_configs.min_nsfw_score * 100)}.jsonl"

            output_fn = os.path.join(exp_configs.output_dir_, output_file)
            if os.path.exists(output_fn):
                raise FileExistsError(f"Output file {output_fn} already exists. Please delete it before running this script.")

            with open(output_fn, "w") as out_file:
                for line, tagged_line in zip(read_lines_from_file(orig_document), read_lines_from_file(tagged_document)):
                    tagged_line_obj = json.loads(tagged_line)
                    line_obj = json.loads(line)

                    assert tagged_line_obj["id"] == line_obj["id"]

                    p_bar.update(1)

                    english_score = tagged_line_obj["attributes"][attribute_key_map["english"]][0][2]
                    toxic_score = tagged_line_obj["attributes"][attribute_key_map["toxic"]][0][2]
                    nsfw_score = tagged_line_obj["attributes"][attribute_key_map["nsfw"]][0][2]

                    if english_score < exp_configs.min_english_score:
                        continue
                    if toxic_score < exp_configs.min_toxic_score:
                        continue
                    if nsfw_score < exp_configs.min_nsfw_score:
                        continue

                    line_obj["file_origin"] = os.path.basename(orig_document)
                    line_obj["english_score"] = english_score
                    line_obj["toxic_score"] = toxic_score
                    line_obj["nsfw_score"] = nsfw_score


                    # write the line to the output file
                    out_file.write(json.dumps(line_obj) + "\n")


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