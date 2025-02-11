# import csv
# import json
# import os
# from collections import Counter
#
# import psutil
# import gc
#
# import numpy as np
# import pandas as pd
# import torch
# from datasets import load_dataset, load_from_disk, Dataset
# from nltk import word_tokenize
# import nltk
# from tqdm import tqdm
from collections import defaultdict

from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from transformers.pipelines.pt_utils import KeyDataset

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
nltk.download("punkt_tab")

# import os
# os.environ['CURL_CA_BUNDLE'] = ''


# # from src.modules.utils import get_score, get_score_with_delay
# # # from src.olmo.run_prepare_data_olmo import save_dataset_to_np
# #
# # # nltk.download('punkt_tab')
# # from nltk.translate.bleu_score import sentence_bleu
# # from nltk.translate.bleu_score import SmoothingFunction
# # from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
#
# from API_KEYS import PERSPECTIVE_API_KEY
#
#
# from collections import Counter
#
# # from transformers import AutoModelForCausalLM, AutoTokenizer
#
# from src.modules.data.load import read_lines_from_file, read_dataset_to_hf, read_lines_zst, save_hf_to_jsonl
# from src.modules.data.data_utils import load_tokenizer
#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# from src.modules.modeling.models.modeling_olmo_custom import CustomOlmoForCausalLM


def main():



    # dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")

    # dataset = load_dataset("json", data_files="/home/ryan/decouple/data/dolma/sample_1BT.jsonl")["train"]

    # import pdb
    # pdb.set_trace()
    # selected_dataset = dataset["train"].shuffle(seed=0).select(range(967210))
    #
    # # selected_dataset.save_to_disk("/home/ryan/decouple/data/fineweb_edu/sample_1BT", num_shards=80, num_proc=80)
    #
    # import pdb
    # pdb.set_trace()

    # load the toxicity classifiers


    # pipe1 = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target", device="cuda")
    #
    # pipe2 = pipeline("text-classification", model="martin-ha/toxic-comment-model")
    #
    # dataset = load_from_disk("/home/ryan/decouple/data/fineweb_edu/sample_1BT")
    # #
    # def split_to_sentences(line):
    #     paragraph = line["text"][0]
    #     return {"text": sent_tokenize(paragraph)}
    #
    # dataset = dataset.map(split_to_sentences, batched=True, batch_size=1, num_proc=80, remove_columns=dataset.column_names)
    #
    # dataset = dataset.select(range(200000))
    # print(dataset)
    # # dataset = dataset.select(range(10000))
    # # we now score the dataset
    #
    # output1 = defaultdict(list)
    # output2 = defaultdict(list)
    # for i, out in enumerate(tqdm(pipe1(KeyDataset(dataset, "text"), batch_size=16, truncation=True), total=len(dataset))):
    #     # tqdm(pipe1(KeyDataset(dataset, "text"), batch_size=128, truncation=True)):
    #     output1["label_1"] += [out["label"]]
    #     output1["score_1"] += [out["score"]]
    #
    # for i, out in enumerate(tqdm(pipe2(KeyDataset(dataset, "text"), batch_size=4, truncation=True), total=len(dataset))):
    #     # tqdm(pipe2(KeyDataset(dataset, "text"), batch_size=128, truncation=True)):
    #     output2["label_2"] += [out["label"]]
    #     output2["score_2"] += [out["score"]]
    #
    # ds1 = Dataset.from_dict(output1)
    # ds2 = Dataset.from_dict(output2)
    #
    # final_dataset = concatenate_datasets([dataset, ds1, ds2], axis=1)
    #
    # final_dataset.save_to_disk("/home/ryan/decouple/data/fineweb_edu/toxicity_classified", num_shards=40, num_proc=40)
    #

    import os
    import matplotlib.pyplot as plt
    from datasets import load_from_disk

    # Load dataset
    dataset_path = "/home/ryan/decouple/data/fineweb_edu"
    dataset = load_from_disk(os.path.join(dataset_path, "toxicity_classified"))

    import pdb
    pdb.set_trace()


    # # Extract scores and invert them when needed
    # scores_1 = [
    #     1 - example['score_1'] if example['label_1'] == "nothate" else example['score_1']
    #     for example in dataset
    # ]
    # scores_2 = [
    #     1 - example['score_2'] if example['label_2'] == "non-toxic" else example['score_2']
    #     for example in dataset
    # ]
    #
    # # Plot histogram for classifier 1
    # plt.figure(figsize=(8, 6))
    # plt.hist(scores_1, bins=50, alpha=0.7, color='blue', edgecolor='black')
    # plt.xlabel("Transformed Score 1")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of Transformed Score 1 (Classifier 1)")
    # plt.grid(True)
    # plt.savefig(os.path.join(dataset_path, "transformed_score_1_distribution.png"))
    # plt.close()
    #
    # # Plot histogram for classifier 2
    # plt.figure(figsize=(8, 6))
    # plt.hist(scores_2, bins=50, alpha=0.7, color='red', edgecolor='black')
    # plt.xlabel("Transformed Score 2")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of Transformed Score 2 (Classifier 2)")
    # plt.grid(True)
    # plt.savefig(os.path.join(dataset_path, "transformed_score_2_distribution.png"))
    # plt.close()
    #
    # print("Transformed histograms saved successfully!")




    # FINISH####


    # dataset = read_dataset_to_hf("locuslab/TOFU", name="retain_perturbed")["train"]
    #
    # masked_outputs = "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_tofu/tofu_5epoch_masked/checkpoint-60/tofu_custom/generation_output_test.jsonl"
    # unfiltered_outputs = "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_tofu/tofu_5epoch_unfiltered/checkpoint-60/tofu_custom/generation_output_test.jsonl"
    #
    # masked_dataset = pd.read_json(masked_outputs, orient="records", lines=True)
    # unfiltered_dataset = pd.read_json(unfiltered_outputs, orient="records", lines=True)
    #
    # import pdb
    # pdb.set_trace()
    #
    # with open("/home/ryan/decouple/tofu_comparison.csv", "w") as f:
    #     writer = csv.writer(f)
    #
    #     writer.writerow(["Question", "Answer", "Masked Response", "Unfiltered Response"])
    #
    #     # Loop through the dataset rows
    #     for i, row in enumerate(dataset):
    #         original = row["question"]
    #         answer = row["answer"]
    #         masked = masked_dataset.iloc[i]["completion"]
    #         unfiltered = unfiltered_dataset.iloc[i]["completion"]
    #
    #         # Write a row to the CSV
    #         writer.writerow([original, answer, masked, unfiltered])
    # ##########
    # #This is for extracting the toxic reddit data only and creating exp10 dataset
    # train_dataset_formatted = load_from_disk("/mnt/nfs1/ryan/decouple/data/olmo_training/cont_pretraining/exp_9_3epoch/temp/train")
    # train_filtered_dataset_formatted = load_from_disk("/mnt/nfs1/ryan/decouple/data/olmo_training/cont_pretraining/exp_9_3epoch/temp/train_filtered")
    #
    # train_output_dir = "/mnt/nfs1/ryan/decouple/data/olmo_training/cont_pretraining/exp_10/orig"
    # train_filtered_output_dir = "/mnt/nfs1/ryan/decouple/data/olmo_training/cont_pretraining/exp_10/filtered_full"
    #
    # max_seq_len = 2048
    #
    # # remove extra columns
    # columns = ["input_ids", "loss_mask"]
    # train_dataset_formatted = train_dataset_formatted.remove_columns(
    #     [col for col in train_dataset_formatted.column_names if col not in columns])
    # train_filtered_dataset_formatted = train_filtered_dataset_formatted.remove_columns(
    #     [col for col in train_filtered_dataset_formatted.column_names if col not in columns])
    #
    # # We save all the data files
    # save_hf_to_jsonl(train_dataset_formatted, os.path.join(train_output_dir, "data.jsonl"), 4)
    # save_dataset_to_np(train_dataset_formatted, train_output_dir, max_seq_len)
    # save_hf_to_jsonl(train_filtered_dataset_formatted, os.path.join(train_filtered_output_dir, "filtered_data.jsonl"),
    #                  4)
    # save_dataset_to_np(train_filtered_dataset_formatted, train_filtered_output_dir, max_seq_len)




    # list_of_models = [
    #     "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_4/filtered_737_exp4",
    #     "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_4/filtered_738_exp4",
    #     "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_4/filtered_add_reddit_exp4",
    #     # "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_4/hf_vanilla_exp4",
    #     "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_4/masked_exp4",
    #     # "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_4/unlikelihood_masked_exp4",
    #     "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_4/unlikelihood_welleck_exp4",
    #     "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_4/vanilla_exp4",
    #     # "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_5/hf_unfiltered_exp5",
    #     # "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_5/masked_exp5",
    #     # "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_5/unfiltered_exp5",
    #     # "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_6/hf_unfiltered_exp6",
    #     # "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_8/hf_masked_exp8",
    #     # "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_8/hf_unfiltered_exp8",
    #     # "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_8/unfiltered_exp8",
    #     # "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_9/unfiltered_exp9"
    # ]
    #
    # model_labels = ["Filtered_737", "Filtered_738", "Filtered_add_reddit", "Masked", "Unlikelihood", "Unfiltered"]
    #
    # evaluation_files = [
    #     "evaluation_step_10.jsonl",
    #     "evaluation_step_50.jsonl",
    #     "evaluation_step_90.jsonl",
    #     "evaluation_step_130.jsonl",
    # ]
    #
    # # Load tokenizer and prepare datasets
    # tokenizer = load_tokenizer("allenai/OLMo-7B-hf", 2048)
    # train_dataset = read_dataset_to_hf("s-nlp/paradetox")["train"].shuffle(seed=0)
    # test_dataset = train_dataset.select(range(int(0.1 * len(train_dataset))))
    #
    #
    #
    # def filter_for_test(row):
    #     list = ["dick", "bitches", "fucks", "nigga", "shits", "retarded", "dumbass", "shitty", "pussies", "lunatic", "crap", "moron", "bastard", "jackass"]
    #     for word in list:
    #         if word in row["en_toxic_comment"]:
    #             return True
    #     return False
    #
    # # test_dataset = train_dataset.filter(filter_for_test)
    # # import pdb
    # # pdb.set_trace()
    #
    # # sentences = train_dataset["en_toxic_comment"]
    # # tokens = [token for sentence in sentences for token in word_tokenize(sentence)]
    # #
    # # benign_sentences = train_dataset["en_neutral_comment"]
    # # benign_tokens = [token for sentence in benign_sentences for token in word_tokenize(sentence)]
    # #
    # # bad_tokens_counter = Counter(tokens)
    # # good_tokens_counter = Counter(benign_tokens)
    # #
    # # import pdb
    # # pdb.set_trace()
    # # new_counter = bad_tokens_counter - good_tokens_counter
    #
    #
    #
    # # Initialize results dictionary
    # results = {eval_file: [] for eval_file in evaluation_files}
    #
    # # Calculate BLEU scores for each model and evaluation file
    # for model_path in list_of_models:
    #     model_name = os.path.basename(model_path)
    #     model_bleu_scores = []
    #     model_perspective_scores = []
    #     print(f"Calculating BLEU scores for model {model_name}")
    #     for eval_file in evaluation_files:
    #         eval_path = os.path.join(model_path, "step1020-unsharded", "hf", "paradetox", "generations", eval_file)
    #         eval_dataset = pd.read_json(eval_path, orient="records", lines=True)
    #
    #         total_bleu = []
    #         total_perspective = []
    #         for i, row in tqdm(eval_dataset.iterrows(), total=len(eval_dataset)):
    #             # Check and process rows
    #             assert row["label"] == test_dataset[i]["en_neutral_comment"] or tokenizer.decode(
    #                 tokenizer.encode(test_dataset[i]["en_neutral_comment"])) == row["label"]
    #             orig_phrase = test_dataset[i]["en_toxic_comment"]
    #             label = row["label"]
    #             generation_phrase = row["generation"]
    #
    #             tokenized_orig_phrase = word_tokenize(orig_phrase)
    #             tokenized_label = word_tokenize(label)
    #             tokenized_generation_phrase = word_tokenize(generation_phrase)
    #
    #             label_bleu = sentence_bleu([tokenized_label], tokenized_orig_phrase,
    #                                        smoothing_function=SmoothingFunction().method1)
    #             if (label_bleu > -1):
    #                 generation_bleu = sentence_bleu([tokenized_label], tokenized_generation_phrase,
    #                                                 smoothing_function=SmoothingFunction().method1)
    #                 total_bleu.append(generation_bleu)
    #                 # total_perspective.append(get_score_with_delay(generation_phrase, PERSPECTIVE_API_KEY))
    #
    #             # Store the average BLEU score for this model and evaluation file
    #         # print(f"Number of samples with BLEU > 0.5: {len(total_bleu)}")
    #         avg_bleu = sum(total_bleu) / len(total_bleu)
    #         model_bleu_scores.append(avg_bleu)
    #
    #         # avg_perspective = sum(total_perspective) / len(total_perspective)
    #         # model_perspective_scores.append(avg_perspective)
    #
    #
    #     # Store results for plotting
    #     for eval_file, bleu_score in zip(evaluation_files, model_bleu_scores):
    #         results[eval_file].append((model_name, bleu_score))
    #     # for eval_file, perspective_score in zip(evaluation_files, model_perspective_scores):
    #     #     results[eval_file].append((model_name, perspective_score))
    #
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # for model_idx, model_path in enumerate(list_of_models):
    #     model_name = os.path.basename(model_path)
    #     bleu_scores = [result[1] for eval_file, result_list in results.items() for result in result_list if
    #                    result[0] == model_name]
    #     plt.plot(evaluation_files, bleu_scores, marker='o', label=model_labels[model_idx])
    #
    # # save the data
    # with open("bleu_scores_present.json", "w") as f:
    #     json.dump(results, f)
    #
    # plt.xlabel("Finetuning Training Steps")
    # plt.ylabel("Average BLEU Score (ID)")
    # plt.title("Average BLEU Score (ID) for Each Model Across Finetuning")
    # plt.legend(loc="best", fontsize='small')
    # plt.grid(True)
    #
    # # save the plot
    # plt.savefig("bleu_scores_present.png")
    #
    #
    # for model in list_of_models:
    #     # first do civilcomments generation
    #     print(f"doing civilcomments generation for model {model}")
    #
    #     # load the dataset
    #     out_fn_balanced = os.path.join(model, "step1020-unsharded", "hf", "civilcomments_generation_direct", "generation_output_balanced.jsonl")
    #     out_fn_unbalanced = os.path.join(model, "step1020-unsharded", "hf", "civilcomments_generation_direct", "generation_output_unbalanced.jsonl")
    #     write_file = os.path.join(model, "step1020-unsharded", "hf", "civilcomments_generation_direct", "performance_metrics.txt")
    #     # loads the file and determine f1 score as well as rocauc score
    #     results_balanced = pd.read_json(out_fn_balanced, orient="records", lines=True)
    #     results_unbalanced = pd.read_json(out_fn_unbalanced, orient="records", lines=True)
    #
    #     prediction_balanced = results_balanced["highest_token"] == 6279
    #     labels_balanced = results_balanced["label"]
    #
    #     f1_balanced = f1_score(labels_balanced, prediction_balanced)
    #     precision_balanced = precision_score(labels_balanced, prediction_balanced)
    #     recall_balanced = recall_score(labels_balanced, prediction_balanced)
    #     accuracy_balanced = accuracy_score(labels_balanced, prediction_balanced)
    #
    #     logits_balanced = results_balanced["logits"]
    #     # we choose the probability of the first class (which is the "yes" class)
    #     probs_balanced = torch.softmax(torch.tensor(logits_balanced), dim=1)[:, 0].numpy()
    #
    #     rocauc_balanced = roc_auc_score(labels_balanced, probs_balanced)
    #
    #
    #     prediction_unbalanced = results_unbalanced["highest_token"] == 6279
    #     labels_unbalanced = results_unbalanced["label"]
    #
    #     f1_unbalanced = f1_score(labels_unbalanced, prediction_unbalanced)
    #     precision_unbalanced = precision_score(labels_unbalanced, prediction_unbalanced)
    #     recall_unbalanced = recall_score(labels_unbalanced, prediction_unbalanced)
    #     accuracy_unbalanced = accuracy_score(labels_unbalanced, prediction_unbalanced)
    #
    #     logits_unbalanced = results_unbalanced["logits"]
    #     # we choose the probability of the first class (which is the "yes" class)
    #     probs_unbalanced = torch.softmax(torch.tensor(logits_unbalanced), dim=1)[:, 0].numpy()
    #
    #     rocauc_unbalanced = roc_auc_score(labels_unbalanced, probs_unbalanced)
    #
    #
    #     print(f"Balanced Dataset Metrics:")
    #     print(f"F1 Score: {f1_balanced:.4f}")
    #     print(f"Precision: {precision_balanced:.4f}")
    #     print(f"Recall: {recall_balanced:.4f}")
    #     print(f"Accuracy: {accuracy_balanced:.4f}")
    #
    #     print("=====================================")
    #
    #     with open(write_file, "w") as f:
    #         f.write(f"Balanced Dataset Metrics:\n")
    #         f.write(f"F1 Score: {f1_balanced:.4f}\n")
    #         f.write(f"Precision: {precision_balanced:.4f}\n")
    #         f.write(f"Recall: {recall_balanced:.4f}\n")
    #         f.write(f"Accuracy: {accuracy_balanced:.4f}\n")
    #         f.write(f"ROC AUC: {rocauc_balanced:.4f}\n\n")
    #
    #         f.write(f"Unbalanced Dataset Metrics:\n")
    #         f.write(f"F1 Score: {f1_unbalanced:.4f}\n")
    #         f.write(f"Precision: {precision_unbalanced:.4f}\n")
    #         f.write(f"Recall: {recall_unbalanced:.4f}\n")
    #         f.write(f"Accuracy: {accuracy_unbalanced:.4f}\n")
    #         f.write(f"ROC AUC: {rocauc_unbalanced:.4f}\n\n")
    #
    #




    # print("enetered")
    # train_dataset = "/mnt/nfs1/ryan/decouple/data/dolma/reddit/toxic_texts/src/comments/RC_2023-06.zst"
    # for line in read_lines_zst(train_dataset):
    #     import pdb
    #     pdb.set_trace()
    #
    #
    # import pdb
    # pdb.set_trace()
    #
    #
    # train_dataset = train_dataset.filter(lambda x: x["gold_label"] != None)
    #
    #
    # print(Counter(train_dataset["gold_label"]))
    #
    #
    # dataset = train_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    #
    # train_dataset = dataset["train"]
    # test_dataset = dataset["test"]
    #
    # LABELS = ['DEG', 'NDG', 'HOM', 'APR', 'CMP']
    #
    # # reformat the dataset such that it is in generation format
    # def reformat_row(row, prompt):
    #     final_instruction = prompt.format(input=row["body"], output="")
    #     label = [1 if LABELS[i] in row["gold_label"] else 0 for i in range(len(LABELS))]
    #     return {"prompt": final_instruction,
    #             "label": label}
    #
    # prompt = "{input}\n{output}"
    #
    # train_dataset = train_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt}, remove_columns=train_dataset.column_names)
    #
    #
    #
    # import pdb
    # pdb.set_trace()


    # gc.collect()
    # print(f"RAM used before loading: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    #
    # # load the dataset
    # dataset = read_dataset_to_hf("/mnt/nfs1/ryan/decouple/data/olmo_training/1epoch_checkpoint737000_1B/output.jsonl")["train"]
    #
    # # choose a smaller dataset for testing
    # dataset = dataset.select(range(0, 70000))
    #
    # print(f"RAM used after loading: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    #
    # # shard1 = dataset.shard(num_shards=2, index=0)
    # # shard2 = dataset.shard(num_shards=2, index=1)
    # # gc.collect()
    # # print(f"RAM used after shards created: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    # #
    # # select1 = dataset.select(range(0, len(dataset) // 2))
    # # select2 = dataset.select(range(len(dataset) // 2, len(dataset)))
    # # gc.collect()
    # # print(f"RAM used after select created: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    #
    #
    # def mapping_function(x, ind):
    #     if ind % 10000 == 0:
    #         gc.collect()
    #         print(f"RAM used during mapping: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    #     x["out"][-1] = 0
    #     x["out"][46] = 7
    #     return x
    #
    # # dataset = dataset.map(mapping_function, num_proc=1, with_indices=True, keep_in_memory=False, cache_file_name="test_cache", writer_batch_size=10000)
    # dataset = dataset.map(mapping_function, num_proc=1, with_indices=True, keep_in_memory=True, writer_batch_size=10000)
    #
    #
    # # print memory usage
    # # Process.memory_info is expressed in bytes, so convert to megabytes
    # gc.collect()
    # print(f"RAM used after processing: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    #
    # import pdb
    # pdb.set_trace()



    # modellong = CustomOlmoForCausalLM.from_pretrained("/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/test_layer_bias/step2527-unsharded/hf_model")
    # modelshort = CustomOlmoForCausalLM.from_pretrained("/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/test_layer_bias_short/step10-unsharded/hf_model")
    # import pdb
    # pdb.set_trace()

    # files = ["/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/filtered_full_exp2/step2527-unsharded/hf/realtoxicityprompts_generation/generation_output_test.jsonl",
    #          "/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/masked_exp2/step2527-unsharded/hf/realtoxicityprompts_generation/generation_output_test3.jsonl",
    #          "/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/unlikelihood_welleck_exp2/step2527-unsharded/hf/realtoxicityprompts_generation/generation_output_test3.jsonl",
    #          "/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/unlikelihood_masked_exp2/step2527-unsharded/hf/realtoxicityprompts_generation/generation_output_test3.jsonl",
    #          "/home/ryan/decouple/models/olmo_ckpt/prefromscratch/0-99-toxic_0-0001-safe/unlikelihood_extreme_exp2/step2527-unsharded/hf/realtoxicityprompts_generation/generation_output_test3.jsonl"]
    #
    #
    # tokenizer = load_tokenizer("allenai/OLMo-7B-hf", 2048)
    #
    # all_words = []
    #
    # for file in files:
    #     dataset = read_dataset_to_hf(file)["train"]
    #
    #     for line in dataset:
    #         tokenized_completion = tokenizer.tokenize(line["completion"])
    #         all_words.extend(tokenized_completion)
    #
    #     counted_words = pd.Series(all_words).value_counts().values
    #
    #     # we get the entropy of the vocab distribution
    #     entropy = -sum(counted_words / sum(counted_words) * np.log(counted_words / sum(counted_words)))
    #
    #     print(f" file {file} has entropy of {entropy}")



    # # load the dataset
    # dataset = read_dataset_to_hf("google/civil_comments")["train"]
    #
    # dataset = dataset.filter(lambda x: x["toxicity"] > 0.5)
    #
    # out_dir = "random_test_out"
    # os.makedirs(out_dir, exist_ok=True)
    #
    # # create a 3x2 grid of plots
    # fig, ax = plt.subplots(3, 2, figsize=(20, 15))
    #
    # ax = ax.flatten()
    #
    # columns = ["severe_toxicity", "insult", "obscene", "identity_attack", "threat", "sexual_explicit"]
    #
    # for i in range(len(columns)):
    #
    #     # plot the distribution of the column
    #     sns.histplot(dataset[columns[i]], kde=True, ax=ax[i], bins=20)
    #     ax[i].set_title(f'Distribution of {columns[i]}')
    #     ax[i].set_xlabel("scores")
    #     ax[i].set_ylabel('Frequency')
    #
    # plt.savefig(f'{out_dir}/group_distribution.png')
    # plt.close()

    # a python program that tests a truth table with three input variables and two outputs


    # df = pd.read_csv("/mnt/nfs1/ryan/decouple/data/jigsawtoxic/all_data.csv.zip")
    #
    # df = df[df["rating"] == "approved"]
    # import pdb
    # pdb.set_trace()



    # read the dataset

    # import pandas as pd
    # df = pd.read_json("/home/ryan/decouple/models/olmo_ckpt/OLMo-1B_scratch_seq-4039237_nomask/get_loss/toxic_nontoxic_0shot.jsonl", lines=True)
    # import pdb
    #
    # pdb.set_trace()



if __name__ == "__main__":
    main()