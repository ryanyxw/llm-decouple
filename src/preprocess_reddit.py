import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
import zstandard as zstd


from tqdm import tqdm

from src.modules.data.load import read_lines_zst, read_lines_from_file
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config, execute_shell_command

def filter_and_process_toxic(lines_and_tags, attribute_key_map, exp_configs):
    print("entered! ")

    processed_lines = []
    # initialize all the metrics to record
    num_collected_counter = 0
    nontoxic_ending = 0  # to count the number of sequences that end with nontoxic sequence
    num_toxic_spans = 0
    num_nontoxic_spans = 0
    num_neutral_spans = 0

    # output_file = os.path.basename(orig_document).split(".")[0] + f"_eng-{int(exp_configs.min_english_score * 100)}_toxic-{int(exp_configs.min_toxic_score * 100)}_docutoxic-{int(exp_configs.min_overall_toxic_score * 100)}_nontoxic-{int(exp_configs.max_nontoxic_score * 100)}.jsonl"
    for line, tagged_line in lines_and_tags:

        tagged_line_obj = json.loads(tagged_line)
        line_obj = json.loads(line)

        assert tagged_line_obj["id"] == line_obj["id"]

        try:
            english_score = tagged_line_obj["attributes"][attribute_key_map["english"]][0][2]
            toxic_document_score = tagged_line_obj["attributes"][attribute_key_map["toxic_document"]][0][2]

            toxic_spans_scores = tagged_line_obj["attributes"][attribute_key_map["toxic_sentence"]]

            if english_score < exp_configs.min_english_score:
                continue

            if toxic_document_score < exp_configs.min_toxic_score:
                continue

            # if the document contains a toxic span as well as a non toxic span, we keep it
            toxic_scores = [span[2] for span in toxic_spans_scores]

        except:
            continue

        #PERPLEXITY_GOOD_REDDIT: changed the following checking condition to only get good documents
        if len(toxic_scores) > 1 and max(toxic_scores) >= exp_configs.min_toxic_score and min(
                toxic_scores) <= exp_configs.max_nontoxic_score:
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


def filter_and_process_nontoxic(lines_and_tags, attribute_key_map, exp_configs):
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

        try:
            english_score = tagged_line_obj["attributes"][attribute_key_map["english"]][0][2]
            toxic_document_score = tagged_line_obj["attributes"][attribute_key_map["toxic_document"]][0][2]
            toxic_spans_scores = tagged_line_obj["attributes"][attribute_key_map["toxic_sentence"]]

            if english_score < exp_configs.min_english_score:
                continue

            #PERPLEXITY_GOOD_REDDIT discard toxic documents
            if toxic_document_score > exp_configs.max_toxic_score:
                continue

            # if the document contains a toxic span as well as a non toxic span, we keep it
            toxic_scores = [span[2] for span in toxic_spans_scores]

        except:
            continue

        #PERPLEXITY_GOOD_REDDIT: only choose sentences with low toxic scores
        if len(toxic_scores) > 1 and max(toxic_scores) <= exp_configs.max_toxic_score:
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

    #we are extracting from the zst file (that was torrented) to documents while performing simple filtering according to dolma
    if configs.extract_from_zst_to_documents.do:
        exp_configs = configs.extract_from_zst_to_documents

        print("enter")

        master_dict = {}

        # we directly write to a zst file
        with open(exp_configs.output_documents_file, "wb") as file_handle:
            compressor = zstd.ZstdCompressor(level=3)
            with compressor.stream_writer(file_handle) as writer:
                tqdm_pbar = tqdm()
                for line in read_lines_zst(exp_configs.input_zst_file):
                    try:
                        tqdm_pbar.update(1)
                        obj = json.loads(line)

                        text_key = "body" if "body" in obj else "selftext"

                        #filter for min_length
                        if (len(obj[text_key]) < exp_configs.min_document_length):
                            continue
                        if obj["score"] < exp_configs.min_upvotes:
                            continue

                        write_obj = {"text": obj[text_key],
                                     "source": "reddit",
                                     "id": obj["id"]}

                        writer.write(f"{json.dumps(write_obj)}\n".encode('utf-8'))
                        # out_file.write(json.dumps(write_obj) + "\n")

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
        yaml_dict["experiment"] = "data"
        yaml_dict["documents"] = [i for i in configs.tag_conversations.in_documents_file]
        yaml_dict["taggers"] = configs.tag_conversations.taggers

        # save the yaml file
        with open(os.path.join(configs.DATA_DIR, f"dolma_tag_{configs.reddit_snapshot}.yaml"), "w") as file:
            save_config(yaml_dict, file)

        command = f"dolma -c {os.path.join(configs.DATA_DIR, f'dolma_tag_{configs.reddit_snapshot}.yaml')} tag"

        print("Tagging conversations...")

        execute_shell_command(command)

        print("Tagging complete")

        # remove the yaml file
        os.remove(os.path.join(configs.DATA_DIR, f"dolma_tag_{configs.reddit_snapshot}.yaml"))

    # we extract toxic documents based on tagged values
    if configs.filter_tags_and_prepare_toxic.do:
        exp_configs = configs.filter_tags_and_prepare_toxic

        assert(len(exp_configs.orig_documents) == len(exp_configs.tag_files))

        exp_name = configs.exp_name
        attribute_key_map = {"english": "data__ft_lang_id_en_doc_v2__en",
                             "toxic_document": "data__jigsaw_hatespeech_document_v2____label__toxic",
                             "toxic_sentence": "data__jigsaw_hatespeech_sentence_v2____label__toxic",
                             "nsfw_document": "data__jigsaw_nsfw_document_v1____label__nsfw",
                             "nsfw_sentence": "data__jigsaw_nsfw_sencence_v2____label__nsfw"
                             }

        if not os.path.exists(exp_configs.output_dir_):
            os.makedirs(exp_configs.output_dir_, exist_ok=True)

        # we save the configs
        save_config(exp_configs, os.path.join(exp_configs.output_dir_, f"config_{configs.reddit_snapshot}.json"))

        for i in range(len(exp_configs.orig_documents)):
            orig_document = exp_configs.orig_documents[i]
            tagged_document = exp_configs.tag_files[i]

            # initialize all the metrics to record
            num_collected_counter = 0
            nontoxic_ending = 0  # to count the number of sequences that end with nontoxic sequence
            num_toxic_spans = 0
            num_nontoxic_spans = 0
            num_neutral_spans = 0

            print(f"orig_document: {orig_document}")
            p_bar = tqdm()

            output_file = os.path.join(exp_configs.output_dir_, os.path.basename(orig_document))

            if (not os.path.basename(orig_document).endswith(".zst")):
                raise ValueError("The original document must be a zst file")

            # remove output file zst extensiond
            output_file = output_file[:-4]

            if os.path.exists(output_file):
                raise FileExistsError(
                    f"Output file {output_file} already exists. Please delete it before running this script.")

            # Read files in chunks
            chunk_size = 1000000  # adjust based on memory
            with ProcessPoolExecutor() as executor:
                with open(output_file, 'w') as out_file:#, open(orig_document, 'r') as orig_file, open(tagged_document, "r") as tagged_file:
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
                            # orig_line = orig_file.readline()
                            # tagged_line = tagged_file.readline()
                            p_bar.update(1)

                            # if not orig_line or not tagged_line:
                            #     break

                            lines_chunk.append(orig_line)
                            tagged_lines_chunk.append(tagged_line)

                        if not lines_chunk:
                            break
                        # filter_and_process_toxic(zip(lines_chunk, tagged_lines_chunk), attribute_key_map, exp_configs)
                        futures.append(executor.submit(filter_and_process_toxic, zip(lines_chunk, tagged_lines_chunk), attribute_key_map, exp_configs))

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

    # we extract based on tagged values for nontoxic text (for evaluation)
    if configs.filter_tags_and_prepare_nontoxic.do:
        exp_configs = configs.filter_tags_and_prepare_nontoxic

        assert(len(exp_configs.orig_documents) == len(exp_configs.tag_files))

        exp_name = configs.exp_name
        attribute_key_map = {"english": "data__ft_lang_id_en_doc_v2__en",
                             "toxic_document": f"data__jigsaw_hatespeech_document_v2____label__toxic",
                             "toxic_sentence": f"data__jigsaw_hatespeech_sentence_v2____label__toxic",
                             "nsfw_document": f"data__jigsaw_nsfw_document_v1____label__nsfw",
                             "nsfw_sentence": f"data__jigsaw_nsfw_sencence_v2____label__nsfw"
                             }

        if not os.path.exists(exp_configs.output_dir_):
            os.makedirs(exp_configs.output_dir_, exist_ok=True)

        # we save the configs
        save_config(exp_configs, os.path.join(exp_configs.output_dir_, f"config{configs.reddit_snapshot}.json"))

        for i in range(len(exp_configs.orig_documents)):
            orig_document = exp_configs.orig_documents[i]
            tagged_document = exp_configs.tag_files[i]

            # initialize all the metrics to record
            num_collected_counter = 0
            nontoxic_ending = 0  # to count the number of sequences that end with nontoxic sequence
            num_toxic_spans = 0
            num_nontoxic_spans = 0
            num_neutral_spans = 0

            print(f"orig_document: {orig_document}")
            p_bar = tqdm()

            output_file = os.path.join(exp_configs.output_dir_, os.path.basename(orig_document))

            if (not os.path.basename(orig_document).endswith(".zst")):
                raise ValueError("The original document must be a zst file")

            # remove output file zst extensiond
            output_file = output_file[:-4]

            if os.path.exists(output_file):
                raise FileExistsError(
                    f"Output file {output_file} already exists. Please delete it before running this script.")

            # Read files in chunks
            chunk_size = 1000000  # adjust based on memory
            with ProcessPoolExecutor() as executor:
                with open(output_file, 'w') as out_file:#, open(orig_document, 'r') as orig_file, open(tagged_document, "r") as tagged_file:
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
                        futures.append(executor.submit(filter_and_process_nontoxic, zip(lines_chunk, tagged_lines_chunk), attribute_key_map, exp_configs))

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