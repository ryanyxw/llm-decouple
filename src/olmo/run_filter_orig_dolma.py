import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from datasets import concatenate_datasets, Sequence, Value, load_from_disk
from tqdm import tqdm
from transformers import DefaultDataCollator, TrainingArguments
from datasets import set_caching_enabled

from src.olmo.run_prepare_data_olmo_ai2 import single_process_format_to_pretraining

set_caching_enabled(False)

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import load_and_reformat_dataset
from src.modules.data.format_utils import preprocess_conversation, format_to_pretraining
from src.modules.data.load import read_dataset_to_hf, save_hf_to_jsonl, read_lines_zst
from src.modules.data.process import multiprocess_map_reduce, single_process_save_to_np, multiprocess_hf_map
from src.modules.data.tokenize import tokenize_with_hate_loss_masking, tokenize_with_hate_loss_span_masking
from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.modeling.inference import run_inference
from src.modules.modeling.modeling_utils import setup_model, free_gpus
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config, get_hash, execute_shell_command


def filter_and_process_dolma(lines_and_tags, attribute_key_map, exp_configs):
    ### NOTE: This will filter out any sentence that contains a span of toxic text. This will not do any sort of masking
    print("entered! ")

    processed_lines = []
    # initialize all the metrics to record
    num_collected_counter = 0
    nontoxic_ending = 0  # to count the number of sequences that end with nontoxic sequence
    num_toxic_spans = 0
    num_nontoxic_spans = 0
    num_neutral_spans = 0

    for line, tagged_line in lines_and_tags:

        tagged_line_obj = json.loads(tagged_line)
        line_obj = json.loads(line)

        assert tagged_line_obj["id"] == line_obj["id"]
        # import pdb
        # pdb.set_trace()
        try:
            english_score = tagged_line_obj["attributes"][attribute_key_map["english"]][0][2]
            toxic_document_score = tagged_line_obj["attributes"][attribute_key_map["toxic_document"]][0][2]
            # nsfw_document_score = tagged_line_obj["attributes"][attribute_key_map["nsfw_document"]][0][2]

            toxic_spans_scores = tagged_line_obj["attributes"][attribute_key_map["toxic_sentence"]]
            # nsfw_spans_scores = tagged_line_obj["attributes"][attribute_key_map["nsfw_sentence"]]

            # if the document contains a toxic span as well as a non toxic span, we keep it
            toxic_scores = [span[2] for span in toxic_spans_scores]

        except:
            continue

        # we filter the document if it contains a neutral or toxic span
        if max(toxic_scores) > exp_configs.max_nontoxic_score:
            continue

        line_obj["english_score"] = english_score
        line_obj["toxic_score"] = toxic_document_score
        line_obj["toxic_spans"] = toxic_spans_scores

        processed_lines.append(line_obj)

        num_toxic_spans += sum([1 for score in toxic_scores if score >= exp_configs.min_toxic_score])
        num_nontoxic_spans += sum([1 for score in toxic_scores if score <= exp_configs.max_nontoxic_score])
        num_neutral_spans += sum([1 for score in toxic_scores if
                                  score > exp_configs.max_nontoxic_score and score < exp_configs.min_toxic_score])

        num_collected_counter += 1
        if toxic_scores[-1] < exp_configs.max_nontoxic_score:
            nontoxic_ending += 1

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

    # given a jsonl file with attribute "out" containing tokenized text, tag and refilter the data
    if configs.filter_orig_dolma.do:
        exp_configs = configs.filter_orig_dolma

        # load the tokenizer
        tokenizer = load_tokenizer(configs.tokenizer_name, configs.max_seq_len)

        # loop through each dataset and filter
        for dataset_path in exp_configs.data_inputarr:
            out_folder = os.path.dirname(dataset_path)

            # load the dataset
            dataset = read_dataset_to_hf(dataset_path)["train"]

            def decode_and_split_documents(example):
                decoded_text = tokenizer.decode(example["out"][0]).split("<|endoftext|>")
                source_arr = ["dolma"] * len(decoded_text)
                id = [get_hash(decoded_text[i]) for i in range(len(decoded_text))]
                return {"text": decoded_text,
                        "source": source_arr,
                        "id": id}

            dataset = dataset.map(decode_and_split_documents,
                                  batched=True,
                                  batch_size=1,
                                  remove_columns=dataset.column_names,
                                  num_proc=configs.num_proc,
                                  )

            # save dataset
            # put the temp path under same directory as the original dataset
            temp_save_path = os.path.join(out_folder, "temp.jsonl")

            # dataset.to_json(temp_save_path, num_proc=20)
            save_hf_to_jsonl(dataset, temp_save_path, num_proc=15)

            # create directory for the sharded dataset
            shard_folder = os.path.join(out_folder, "documents")
            os.makedirs(shard_folder, exist_ok=True)

            # shard the dataset
            command = f"split -l 500000 -d --additional-suffix=.jsonl --verbose {temp_save_path} {shard_folder}/shard_"
            # command = f"split -l 20000 -d --additional-suffix=.jsonl --verbose {temp_save_path} {shard_folder}/shard_"

            execute_shell_command(command)

            # compress the sharded dataset
            command = f"zstd {shard_folder}/*"
            execute_shell_command(command)

            # delete the uncompressed jsonl shards
            command = f"rm {shard_folder}/*.jsonl"

            # tag conversations
            yaml_dict = dict()
            yaml_dict["processes"] = configs.num_proc
            yaml_dict["experiment"] = configs.exp_name
            yaml_dict["documents"] = [f"{shard_folder}/*.zst"]
            yaml_dict["taggers"] = exp_configs.taggers

            # save the yaml file
            with open(os.path.join(out_folder, "dolma_tag.yaml"), "w") as file:
                save_config(yaml_dict, file)

            command = f"dolma -c {os.path.join(out_folder, 'dolma_tag.yaml')} tag"

            print("Tagging conversations...")

            execute_shell_command(command)

            print("Tagging complete")

            # removing extra files
            command = f"rm {out_folder}/dolma_tag.yaml && rm {out_folder}/temp_dir.txt"
            execute_shell_command(command)

            # merge the dataset back together
            tagged_document_fn = os.path.join(out_folder, "attributes.jsonl.zst")
            command = f"cat {out_folder}/attributes/{configs.exp_name}/*.zst > {tagged_document_fn}"
            execute_shell_command(command)

            orig_documents_fn  = os.path.join(out_folder, "documents.jsonl.zst")
            command = f"cat {shard_folder}/*.zst > {orig_documents_fn}"
            execute_shell_command(command)

            command = f"rm -r {shard_folder} && rm -r {out_folder}/attributes && rm {temp_save_path}"
            execute_shell_command(command)

            # filter the dataset
            exp_name = configs.exp_name
            attribute_key_map = {"english": f"{exp_name}__ft_lang_id_en_doc_v2__en",
                                 "toxic_document": f"{exp_name}__jigsaw_hatespeech_document_v2____label__toxic",
                                 "toxic_sentence": f"{exp_name}__jigsaw_hatespeech_sentence_v2____label__toxic",
                                 "nsfw_document": f"{exp_name}__jigsaw_nsfw_document_v1____label__nsfw",
                                 "nsfw_sentence": f"{exp_name}__jigsaw_nsfw_sencence_v2____label__nsfw"
                                 }

            orig_document = orig_documents_fn
            tagged_document = tagged_document_fn

            # initialize all the metrics to record
            num_collected_counter = 0
            nontoxic_ending = 0  # to count the number of sequences that end with nontoxic sequence
            num_toxic_spans = 0
            num_nontoxic_spans = 0
            num_neutral_spans = 0

            print(f"orig_document: {orig_document}")
            p_bar = tqdm()

            output_file = os.path.join(out_folder, "output_filtered.jsonl.zst")

            if (not os.path.basename(orig_document).endswith(".zst")):
                raise ValueError("The original document must be a zst file")

            # remove output file zst extensiond
            output_file = output_file[:-4]

            # Read files in chunks
            # chunk_size = 100000  # adjust based on memory
            chunk_size = 1000000  # adjust based on memory

            with ProcessPoolExecutor() as executor:
                with open(output_file,
                          'w') as out_file:  # , open(orig_document, 'r') as orig_file, open(tagged_document, "r") as tagged_file:
                    # create generators for reading in lines
                    orig_file_generator = read_lines_zst(orig_document)
                    tagged_file_generator = read_lines_zst(tagged_document)
                    futures = []
                    while True:
                        lines_chunk = []
                        tagged_lines_chunk = []

                        for _ in range(chunk_size):
                            try:
                                orig_line = next(orig_file_generator)
                                tagged_line = next(tagged_file_generator)
                            except StopIteration:
                                # if we've exhausted the file
                                break
                            p_bar.update(1)

                            lines_chunk.append(orig_line)
                            tagged_lines_chunk.append(tagged_line)

                        if not lines_chunk:
                            break

                        futures.append(executor.submit(filter_and_process_dolma, zip(lines_chunk, tagged_lines_chunk),
                                                       attribute_key_map, exp_configs))

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
                with open(os.path.join(out_folder,
                                       "output_filtered_info.txt"), "w") as file:
                    file.write(f"Number of documents: {num_collected_counter}\n")
                    file.write(f"Number of documents ending with nontoxic sequence: {nontoxic_ending}\n")
                    file.write(f"Number of toxic spans: {num_toxic_spans}\n")
                    file.write(f"Number of nontoxic spans: {num_nontoxic_spans}\n")
                    file.write(f"Number of neutral spans: {num_neutral_spans}\n")

            # remove extra files
            command = f"rm {orig_documents_fn} {tagged_document_fn}"
            execute_shell_command(command)

    if configs.tokenize_orig_dolma.do:
        exp_configs = configs.tokenize_orig_dolma

        for dataset_path in exp_configs.data_inputarr:

            out_directory = os.path.join(os.path.dirname(dataset_path), "tokenized_and_filtered")

            tokenizer = load_tokenizer(configs.tokenizer_name, configs.max_seq_len)

            train_dataset = read_dataset_to_hf(dataset_path, num_proc=configs.num_proc)["train"]

            print("enter")

            # tokenize the train datasets
            train_dataset = train_dataset.map(tokenize_with_hate_loss_span_masking,
                                              batched=True,
                                              batch_size=1,
                                              remove_columns=train_dataset.column_names,
                                              num_proc=configs.num_proc,
                                              fn_kwargs={
                                                  "toxic_threshold": exp_configs.toxic_threshold,
                                                  "safe_threshold": exp_configs.safe_threshold,
                                                  "tokenizer": tokenizer}
                                              )

            # THIS IS THE MOST MEMORY INTENSIVE. Decrease num_proc if memory is overloading (this makes multiple copies of the dataset and loops through the entire dataset)
            train_dataset_formatted = multiprocess_hf_map(single_process_format_to_pretraining, train_dataset,
                                                          num_proc=1,
                                                          fn_kwargs={"tokenizer": tokenizer,
                                                                     "max_seq_len": configs.max_seq_len})


            # save the datasets for memory mapping
            train_dataset_formatted.save_to_disk(out_directory, num_shards=exp_configs.num_shards)


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