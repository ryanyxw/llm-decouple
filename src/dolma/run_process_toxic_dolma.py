import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from src.modules.data.load import read_lines_zst, read_lines_from_file
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config, execute_shell_command

def filter_and_process_toxic_dolma(lines, attribute_key_map, exp_configs):
    print("entered! ")

    processed_lines = []
    # initialize all the metrics to record
    num_collected_counter = 0
    nontoxic_ending = 0  # to count the number of sequences that end with nontoxic sequence
    num_toxic_spans = 0
    num_nontoxic_spans = 0
    num_neutral_spans = 0

    for line in lines:

        line_obj = json.loads(line)

        try:
            toxic_spans_scores = line_obj["attributes"][attribute_key_map["toxic_sentence"]]
            # nsfw_spans_scores = tagged_line_obj["attributes"][attribute_key_map["nsfw_sentence"]]

            # if the document contains a toxic span as well as a non toxic span, we keep it
            toxic_scores = [span[2] for span in toxic_spans_scores]

            toxic_boolean = [1 if score >= exp_configs.min_toxic_score else 0 for score in toxic_scores]
        except:
            continue

        if len(toxic_scores) > 1 and max(toxic_scores) >= exp_configs.min_toxic_score and min(
                toxic_scores) <= exp_configs.max_nontoxic_score and sum(toxic_boolean) / len(toxic_boolean) >= exp_configs.min_toxic_ratio:
            line_obj["toxic_spans"] = toxic_spans_scores

            processed_lines.append(line_obj)

            num_toxic_spans += sum([1 for score in toxic_scores if score >= exp_configs.min_toxic_score])
            num_nontoxic_spans += sum([1 for score in toxic_scores if score <= exp_configs.max_nontoxic_score])
            num_neutral_spans += sum([1 for score in toxic_scores if
                                      score > exp_configs.max_nontoxic_score and score < exp_configs.min_toxic_score])

            num_collected_counter += 1
            if toxic_scores[-1] < exp_configs.max_nontoxic_score:
                nontoxic_ending += 1

    print("exited! ")

    return processed_lines, num_collected_counter, nontoxic_ending, num_toxic_spans, num_nontoxic_spans, num_neutral_spans

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

    # we extract based on tagged values
    if configs.full_sweep.do:
        exp_configs = configs.full_sweep

        exp_name = configs.exp_name + "_full"
        attribute_key_map = {"toxic_sentence": "hatespeech_nsfw_cc_v3__jigsaw_hatespeech_sentence_v2____label__toxic",
                             "nsfw_sentence": "hatespeech_nsfw_cc_v3__jigsaw_nsfw_sencence_v2____label__nsfw"
                             }

        if not os.path.exists(exp_configs.output_dir_):
            os.makedirs(exp_configs.output_dir_, exist_ok=True)

        # we save the configs
        save_config(exp_configs, os.path.join(exp_configs.output_dir_, "config.json"))

        for i in range(exp_configs.tot_files + 1):
            # format i into 4 digit number
            filenum = str(i).zfill(4)
            aws_file_path = exp_configs.aws_link.format(filenum=filenum)
            # we make a call to aws to get the files
            execute_shell_command(f"aws s3 cp {aws_file_path} {exp_configs.output_dir_}/temp_{filenum}.json.gz")

            orig_document = os.path.join(exp_configs.output_dir_, f"temp_{filenum}.json.gz")

            # initialize all the metrics to record
            num_collected_counter = 0
            nontoxic_ending = 0  # to count the number of sequences that end with nontoxic sequence
            num_toxic_spans = 0
            num_nontoxic_spans = 0
            num_neutral_spans = 0

            print(f"orig_document: {orig_document}")
            p_bar = tqdm()

            output_file = os.path.join(exp_configs.output_dir_, os.path.basename(aws_file_path))
            if os.path.exists(output_file):
                raise FileExistsError(
                    f"Output file {output_file} already exists. Please delete it before running this script.")

            # Read files in chunks
            chunk_size = 10000  # adjust based on memory
            with ProcessPoolExecutor() as executor:
                with gzip.open(output_file, 'wt') as out_file, gzip.open(orig_document, 'rt') as orig_file:
                    futures = []
                    while True:
                        lines_chunk = []

                        for _ in range(chunk_size):
                            orig_line = orig_file.readline()
                            p_bar.update(1)

                            if not orig_line:
                                break

                            lines_chunk.append(orig_line)

                        if not lines_chunk:
                            break

                        futures.append(executor.submit(filter_and_process_toxic_dolma, lines_chunk, attribute_key_map,
                                                       exp_configs))

                    for future in futures:
                        processed_lines, temp_num_collected_counter, temp_nontoxic_ending, temp_num_toxic_spans, temp_num_nontoxic_spans, temp_num_neutral_spans = future.result()
                        for line_obj in processed_lines:
                            out_file.write(json.dumps(line_obj) + "\n")

                        # Further processing or aggregation of span counts
                        num_collected_counter += temp_num_collected_counter
                        nontoxic_ending += temp_nontoxic_ending
                        num_toxic_spans += temp_num_toxic_spans
                        num_nontoxic_spans += temp_num_nontoxic_spans
                        num_neutral_spans += temp_num_neutral_spans
                with open(os.path.join(exp_configs.output_dir_,
                                       os.path.basename(aws_file_path).split(".")[0] + "_info.txt"), "w") as file:
                    file.write(f"Number of documents: {num_collected_counter}\n")
                    file.write(f"Number of documents ending with nontoxic sequence: {nontoxic_ending}\n")
                    file.write(f"Number of toxic spans: {num_toxic_spans}\n")
                    file.write(f"Number of nontoxic spans: {num_nontoxic_spans}\n")
                    file.write(f"Number of neutral spans: {num_neutral_spans}\n")

            # remove the temp file
            os.remove(orig_document)


    # we extract based on tagged values
    if configs.filter_tags_and_prepare.do:
        exp_configs = configs.filter_tags_and_prepare

        exp_name = configs.exp_name + "_full"
        attribute_key_map = {"toxic_sentence": "hatespeech_nsfw_cc_v3__jigsaw_hatespeech_sentence_v2____label__toxic",
                             "nsfw_sentence": "hatespeech_nsfw_cc_v3__jigsaw_nsfw_sencence_v2____label__nsfw"
                             }

        if not os.path.exists(exp_configs.output_dir_):
            os.makedirs(exp_configs.output_dir_, exist_ok=True)

        # we save the configs
        save_config(exp_configs, os.path.join(exp_configs.output_dir_, "config.json"))

        for i in range(len(exp_configs.orig_documents)):
            orig_document = exp_configs.orig_documents[i]

            # initialize all the metrics to record
            num_collected_counter = 0
            nontoxic_ending = 0  # to count the number of sequences that end with nontoxic sequence
            num_toxic_spans = 0
            num_nontoxic_spans = 0
            num_neutral_spans = 0

            print(f"orig_document: {orig_document}")
            p_bar = tqdm()

            output_file = os.path.join(exp_configs.output_dir_, os.path.basename(orig_document))
            if os.path.exists(output_file):
                raise FileExistsError(
                    f"Output file {output_file} already exists. Please delete it before running this script.")

            # Read files in chunks
            chunk_size = 10000  # adjust based on memory
            with ProcessPoolExecutor() as executor:
                with gzip.open(output_file, 'wt') as out_file, gzip.open(orig_document, 'rt') as orig_file:
                    futures = []
                    while True:
                        lines_chunk = []

                        for _ in range(chunk_size):
                            orig_line = orig_file.readline()
                            p_bar.update(1)

                            if not orig_line:
                                break

                            lines_chunk.append(orig_line)

                        if not lines_chunk:
                            break

                        futures.append(executor.submit(filter_and_process_toxic_dolma, lines_chunk, attribute_key_map, exp_configs))

                    for future in futures:
                        processed_lines, temp_num_collected_counter, temp_nontoxic_ending, temp_num_toxic_spans, temp_num_nontoxic_spans, temp_num_neutral_spans = future.result()
                        for line_obj in processed_lines:
                            out_file.write(json.dumps(line_obj) + "\n")

                        # Further processing or aggregation of span counts
                        num_collected_counter += temp_num_collected_counter
                        nontoxic_ending += temp_nontoxic_ending
                        num_toxic_spans += temp_num_toxic_spans
                        num_nontoxic_spans += temp_num_nontoxic_spans
                        num_neutral_spans += temp_num_neutral_spans
                with open(os.path.join(exp_configs.output_dir_, os.path.basename(orig_document).split(".")[0] + "_info.txt"), "w") as file:
                    file.write(f"Number of documents: {num_collected_counter}\n")
                    file.write(f"Number of documents ending with nontoxic sequence: {nontoxic_ending}\n")
                    file.write(f"Number of toxic spans: {num_toxic_spans}\n")
                    file.write(f"Number of nontoxic spans: {num_nontoxic_spans}\n")
                    file.write(f"Number of neutral spans: {num_neutral_spans}\n")




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