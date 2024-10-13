import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from src.modules.data.load import read_lines_zst, read_lines_from_file
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config, execute_shell_command

def filter_and_process(lines_and_tags, attribute_key_map, exp_configs):
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
            # nsfw_document_score = tagged_line_obj["attributes"][attribute_key_map["nsfw_document"]][0][2]

            toxic_spans_scores = tagged_line_obj["attributes"][attribute_key_map["toxic_sentence"]]
            # nsfw_spans_scores = tagged_line_obj["attributes"][attribute_key_map["nsfw_sentence"]]

            if english_score < exp_configs.min_english_score:
                continue

            if toxic_document_score < exp_configs.min_toxic_score:
                continue

            # if the document contains a toxic span as well as a non toxic span, we keep it
            toxic_scores = [span[2] for span in toxic_spans_scores]

        except:
            continue
        # if len(toxic_scores) > 1 and max(toxic_scores) > exp_configs.min_toxic_score and min(toxic_scores) < exp_configs.max_nontoxic_score and toxic_scores[-1] < exp_configs.max_nontoxic_score:
        #     line_obj["file_origin"] = os.path.basename(orig_document)
        #     line_obj["english_score"] = english_score
        #     line_obj["toxic_score"] = toxic_document_score
        #     line_obj["toxic_spans"] = toxic_spans_scores
        #
        #     num_collected_counter += 1
        #     p_bar.update(1)
        #
        #     # write the line to the output file
        #     out_file.write(json.dumps(line_obj) + "\n")
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
        yaml_dict["experiment"] = configs.exp_name + "_full"
        yaml_dict["documents"] = [i for i in configs.tag_conversations.in_documents_file]
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

        exp_name = configs.exp_name + "_full"
        attribute_key_map = {"english": f"{exp_name}__ft_lang_id_en_doc_v2__en",
                             "toxic_document": f"{exp_name}__jigsaw_hatespeech_document_v2____label__toxic",
                             "toxic_sentence": f"{exp_name}__jigsaw_hatespeech_sentence_v2____label__toxic",
                             "nsfw_document": f"{exp_name}__jigsaw_nsfw_document_v1____label__nsfw",
                             "nsfw_sentence": f"{exp_name}__jigsaw_nsfw_sencence_v2____label__nsfw"
                             }

        if not os.path.exists(exp_configs.output_dir_):
            os.makedirs(exp_configs.output_dir_, exist_ok=True)

        # we save the configs
        save_config(exp_configs, os.path.join(exp_configs.output_dir_, "config.json"))

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
            if os.path.exists(output_file):
                raise FileExistsError(
                    f"Output file {output_file} already exists. Please delete it before running this script.")

            # Read files in chunks
            chunk_size = 1000000  # adjust based on memory
            with ProcessPoolExecutor() as executor:
                with open(output_file, 'w') as out_file, open(orig_document, 'r') as orig_file, open(tagged_document, "r") as tagged_file:
                    futures = []
                    while True:
                        lines_chunk = []
                        tagged_lines_chunk = []

                        for _ in range(chunk_size):
                            orig_line = orig_file.readline()
                            tagged_line = tagged_file.readline()
                            p_bar.update(1)

                            if not orig_line or not tagged_line:
                                break

                            lines_chunk.append(orig_line)
                            tagged_lines_chunk.append(tagged_line)

                        if not lines_chunk:
                            break
                        futures.append(executor.submit(filter_and_process, zip(lines_chunk, tagged_lines_chunk), attribute_key_map, exp_configs))

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

            # # open the file to write
            # with open(output_file, "w") as out_file:
            #
            #     # output_file = os.path.basename(orig_document).split(".")[0] + f"_eng-{int(exp_configs.min_english_score * 100)}_toxic-{int(exp_configs.min_toxic_score * 100)}_docutoxic-{int(exp_configs.min_overall_toxic_score * 100)}_nontoxic-{int(exp_configs.max_nontoxic_score * 100)}.jsonl"
            #     for line, tagged_line in zip(read_lines_from_file(orig_document), read_lines_from_file(tagged_document)):
            #
            #         tagged_line_obj = json.loads(tagged_line)
            #         line_obj = json.loads(line)
            #
            #         assert tagged_line_obj["id"] == line_obj["id"]
            #
            #         try:
            #             english_score = tagged_line_obj["attributes"][attribute_key_map["english"]][0][2]
            #             toxic_document_score = tagged_line_obj["attributes"][attribute_key_map["toxic_document"]][0][2]
            #             # nsfw_document_score = tagged_line_obj["attributes"][attribute_key_map["nsfw_document"]][0][2]
            #
            #             toxic_spans_scores = tagged_line_obj["attributes"][attribute_key_map["toxic_sentence"]]
            #             # nsfw_spans_scores = tagged_line_obj["attributes"][attribute_key_map["nsfw_sentence"]]
            #
            #             if english_score < exp_configs.min_english_score:
            #                 continue
            #
            #             if toxic_document_score < exp_configs.min_toxic_score:
            #                 continue
            #
            #             #if the document contains a toxic span as well as a non toxic span, we keep it
            #             toxic_scores = [span[2] for span in toxic_spans_scores]
            #
            #         except:
            #             print("errorr! exception occurred")
            #             continue
            #         # if len(toxic_scores) > 1 and max(toxic_scores) > exp_configs.min_toxic_score and min(toxic_scores) < exp_configs.max_nontoxic_score and toxic_scores[-1] < exp_configs.max_nontoxic_score:
            #         #     line_obj["file_origin"] = os.path.basename(orig_document)
            #         #     line_obj["english_score"] = english_score
            #         #     line_obj["toxic_score"] = toxic_document_score
            #         #     line_obj["toxic_spans"] = toxic_spans_scores
            #         #
            #         #     num_collected_counter += 1
            #         #     p_bar.update(1)
            #         #
            #         #     # write the line to the output file
            #         #     out_file.write(json.dumps(line_obj) + "\n")
            #         if len(toxic_scores) > 1 and max(toxic_scores) >= exp_configs.min_toxic_score and min(toxic_scores) <= exp_configs.max_nontoxic_score:
            #             line_obj["file_origin"] = os.path.basename(orig_document)
            #             line_obj["english_score"] = english_score
            #             line_obj["toxic_score"] = toxic_document_score
            #             line_obj["toxic_spans"] = toxic_spans_scores
            #
            #             num_toxic_spans += sum([1 for score in toxic_scores if score >= exp_configs.min_toxic_score])
            #             num_nontoxic_spans += sum([1 for score in toxic_scores if score <= exp_configs.max_nontoxic_score])
            #             num_neutral_spans += sum([1 for score in toxic_scores if score > exp_configs.max_nontoxic_score and score < exp_configs.min_toxic_score])
            #
            #             num_collected_counter += 1
            #             if toxic_scores[-1] < exp_configs.max_nontoxic_score:
            #                 nontoxic_ending += 1
            #             p_bar.update(1)
            #
            #             # write the line to the output file
            #             out_file.write(json.dumps(line_obj) + "\n")
            #
            # # we store basic info about this file
            # with open(os.path.join(exp_configs.output_dir_, os.path.basename(orig_document).split(".")[0] + "_info.txt"), "w") as file:
            #     file.write(f"Number of documents: {num_collected_counter}\n")
            #     file.write(f"Number of documents ending with nontoxic sequence: {nontoxic_ending}\n")
            #     file.write(f"Number of toxic spans: {num_toxic_spans}\n")
            #     file.write(f"Number of nontoxic spans: {num_nontoxic_spans}\n")
            #     file.write(f"Number of neutral spans: {num_neutral_spans}\n")


    if configs.prepare_nontoxic_only.do:
        # note that nontoxic documents still contain toxic spans that need to be filtered out
        exp_configs = configs.prepare_nontoxic_only

        assert (len(exp_configs.orig_documents) == len(exp_configs.tag_files))

        exp_name = configs.exp_name + "_full"
        attribute_key_map = {"english": f"{exp_name}__ft_lang_id_en_doc_v2__en",
                             "toxic_document": f"{exp_name}__jigsaw_hatespeech_document_v2____label__toxic",
                             "toxic_sentence": f"{exp_name}__jigsaw_hatespeech_sentence_v2____label__toxic",
                             "nsfw_document": f"{exp_name}__jigsaw_nsfw_document_v1____label__nsfw",
                             "nsfw_sentence": f"{exp_name}__jigsaw_nsfw_sencence_v2____label__nsfw"
                             }

        for i in range(len(exp_configs.orig_documents)):
            orig_document = exp_configs.orig_documents[i]
            tagged_document = exp_configs.tag_files[i]

            print(f"orig_document: {orig_document}")
            p_bar = tqdm()

            output_file = os.path.basename(orig_document).split(".")[
                              0] + f"nontoxic-documents_eng-{int(exp_configs.min_english_score * 100)}_docutoxic-{int(exp_configs.min_overall_toxic_score * 100)}_nontoxic-{int(exp_configs.max_nontoxic_score * 100)}.jsonl"

            output_fn = os.path.join(exp_configs.output_dir_, output_file)
            if os.path.exists(output_fn):
                raise FileExistsError(
                    f"Output file {output_fn} already exists. Please delete it before running this script.")

            num_collected_counter = 0


            with open(output_fn, "w") as out_file:
                for line, tagged_line in zip(read_lines_from_file(orig_document),
                                             read_lines_from_file(tagged_document)):

                    # if we reach a certain number of collected documents, we stop
                    if num_collected_counter >= exp_configs.num_documents:
                        break

                    tagged_line_obj = json.loads(tagged_line)
                    line_obj = json.loads(line)

                    assert tagged_line_obj["id"] == line_obj["id"]

                    try:

                        english_score = tagged_line_obj["attributes"][attribute_key_map["english"]][0][2]
                        toxic_document_score = tagged_line_obj["attributes"][attribute_key_map["toxic_document"]][0][2]
                        # nsfw_document_score = tagged_line_obj["attributes"][attribute_key_map["nsfw_document"]][0][2]

                        toxic_spans_scores = tagged_line_obj["attributes"][attribute_key_map["toxic_sentence"]]
                        # nsfw_spans_scores = tagged_line_obj["attributes"][attribute_key_map["nsfw_sentence"]]

                        if english_score < exp_configs.min_english_score:
                            continue

                        if toxic_document_score < exp_configs.min_toxic_score:
                            continue

                        # if the document contains a toxic span as well as a non toxic span, we keep it
                        toxic_scores = [span[2] for span in toxic_spans_scores]

                    except:
                        print("errorr! exception occurred")
                        continue

                    # if the document satisfies the previous filtering condition, we discard it
                    if len(toxic_scores) > 1 and max(toxic_scores) > exp_configs.min_toxic_score and min(
                            toxic_scores) < exp_configs.max_nontoxic_score and toxic_scores[
                        -1] < exp_configs.max_nontoxic_score:
                        continue
                    # we choose the document only if it contains more than 1 sentence (to keep in distribution) and has a non-toxic span
                    elif len(toxic_scores) > 1 and min(toxic_scores) < exp_configs.max_nontoxic_score:
                        line_obj["file_origin"] = os.path.basename(orig_document)
                        line_obj["english_score"] = english_score
                        line_obj["toxic_score"] = toxic_document_score
                        line_obj["toxic_spans"] = toxic_spans_scores

                        num_collected_counter += 1
                        p_bar.update(1)

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