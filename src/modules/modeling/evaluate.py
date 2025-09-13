import json
import multiprocessing
import os

import numpy as np
from scipy.stats import sem
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_fscore_support, average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from API_KEYS import PERSPECTIVE_API_KEY
from src.modules.data.datasets.PandasDataset import PandasDataset
from src.modules.data.format_utils import select_binary_balanced_dataset, select_n_ary_balanced_dataset
from src.modules.data.load import read_dataset_to_hf
from src.modules.modeling.inference import run_inference_new, obtain_logit, calculate_loss_across_tokens
from src.modules.modeling.models.LogisticRegression import BinaryClassifier, TrinaryClassifier
from src.modules.templates import TOXIC_CLASSIFICATION_WITH_PROMPT, \
    TOXIC_CLASSIFICATION_NO_PROMPT, \
    TOFU_NAMES, TOFU_TEMPLATE, TOFU_QUERY_TEMPLATE
from src.modules.utils import use_perspective_api, seed_all
from src.training.run_train_torch import train_classifier, train_binaryclassifier_multi


def save_evaluator_config_and_sample(evaluator, out_dir, sample):
    # saves a sample of the prompt to a parallel file along with configs
    print("sample of example fed into model: \n" + repr(sample))
    with open(os.path.join(out_dir, "template.jsonl"), "w") as f:
        f.write(repr(sample) + "\n")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        OmegaConf.save(evaluator, f, resolve=True)

def real_toxicity_prompt_chat_generation_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
    Evaluates the model using the real toxicity prompts. returns the process for evaluating using perspective
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """

    # load the dataset and select the necessary ones
    dataset = read_dataset_to_hf(evaluator.data.name)["train"].shuffle(seed=evaluator.seed)
    assert(len(dataset) >= evaluator.data.num_examples)
    dataset = dataset.select(range(evaluator.data.num_examples))

    # reformat the dataset such that it is in generation format
    def reformat_row(row):
        conversation = [{"role": "user", "content": row["prompt"]["text"]}]
        return {"prompt": conversation}
    dataset = dataset.map(reformat_row)

    save_evaluator_config_and_sample(evaluator, out_dir, dataset[0]["prompt"])

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "generation_output_test.jsonl")
    print("saving to ", out_fn)

    run_inference_new("generate", hf_model, tokenizer, dataset, out_fn, batch_size=evaluator.batch_size , generation_kwargs=evaluator.generation_kwargs, use_chat_template=True)

    # creates a processes that calls google perspective API and saves the output
    progress_file = os.path.join(out_dir, "perspective_api_progress_includingprompt.json")
    use_perspective_api(out_fn, PERSPECTIVE_API_KEY, progress_file)


def real_toxicity_prompt_generation_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
    Evaluates the model using the real toxicity prompts. returns the process for evaluating using perspective
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """

    # load the dataset and select the necessary ones
    dataset = read_dataset_to_hf(evaluator.data.name)["train"].shuffle(seed=evaluator.seed)
    assert(len(dataset) >= evaluator.data.num_examples)
    dataset = dataset.select(range(evaluator.data.num_examples))

    # reformat the dataset such that it is in generation format
    def reformat_row(row):
        return {"prompt": row["prompt"]["text"]}
    dataset = dataset.map(reformat_row)

    save_evaluator_config_and_sample(evaluator, out_dir, dataset[0]["prompt"])

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "generation_output_test.jsonl")
    print("saving to ", out_fn)

    run_inference_new("generate", hf_model, tokenizer, dataset, out_fn, batch_size=evaluator.batch_size , generation_kwargs=evaluator.generation_kwargs)

    # creates a processes that calls google perspective API and saves the output
    progress_file = os.path.join(out_dir, "perspective_api_progress_includingprompt.json")
    use_perspective_api(out_fn, PERSPECTIVE_API_KEY, progress_file)


def NEW_hidden_state_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir):
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")

    dataset = read_dataset_to_hf(evaluator.data.name).shuffle(seed=evaluator.seed)
    test_dataset = dataset["test"]
    train_dataset = dataset["train"]

    roc_auc_scores = []

    for i in range(evaluator.data.num_samples):  # Generate 5 different subsamples
        train_subsample = select_binary_balanced_dataset(
            train_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
            evaluator.seed + i, evaluator.data.num_train // 2)

        test_subsample = select_binary_balanced_dataset(
            test_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
            evaluator.seed + i, evaluator.data.num_test // 2)

        def reformat_row(row, prompt):
            final_instruction = prompt.format(input=row["text"], output="")
            return {"prompt": final_instruction, "label": row["toxicity"] >= evaluator.data.toxicity_threshold}

        prompt = TOXIC_CLASSIFICATION_WITH_PROMPT if evaluator.use_prompt else TOXIC_CLASSIFICATION_NO_PROMPT
        train_subsample = train_subsample.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})
        test_subsample = test_subsample.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

        run_inference_new("hidden_state", hf_model, tokenizer, train_subsample, train_out_fn,
                          batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, test_subsample, test_out_fn,
                          batch_size=evaluator.batch_size)

        classifier_train_dataset = pd.read_json(train_out_fn, orient="records", lines=True)
        classifier_test_dataset = pd.read_json(test_out_fn, orient="records", lines=True)

        X_train = np.stack(classifier_train_dataset["hidden_state"])
        y_train = np.stack(classifier_train_dataset["label"])
        X_test = np.stack(classifier_test_dataset["hidden_state"])
        y_test = np.stack(classifier_test_dataset["label"])

        indices_train = np.random.permutation(len(X_train))
        X_train, y_train = X_train[indices_train], y_train[indices_train]
        indices_test = np.random.permutation(len(X_test))
        X_test, y_test = X_test[indices_test], y_test[indices_test]

        clf = LogisticRegression(class_weight="balanced", max_iter=5000)
        clf.fit(X_train, y_train)

        y_pred_test = clf.predict(X_test)
        y_prob_test = clf.predict_proba(X_test)[:, 1]

        roc_auc_scores.append(roc_auc_score(y_test, y_prob_test))

    metrics = {
        "Test ROC AUC": (np.mean(roc_auc_scores), sem(roc_auc_scores), roc_auc_scores),
    }

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write("Test Metrics (Mean ± StdErr):\n")
        for metric, (mean, stderr, raw) in metrics.items():
            f.write(f"{metric}: {mean:.4f} ± {stderr:.4f} | {raw}\n")


def NEW_CHAT_hidden_state_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir):
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")

    dataset = read_dataset_to_hf(evaluator.data.name).shuffle(seed=evaluator.seed)
    test_dataset = dataset["test"]
    train_dataset = dataset["train"]

    roc_auc_scores = []

    for i in range(evaluator.data.num_samples):  # Generate 5 different subsamples
        train_subsample = select_binary_balanced_dataset(
            train_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
            evaluator.seed + i, evaluator.data.num_train // 2)

        test_subsample = select_binary_balanced_dataset(
            test_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
            evaluator.seed + i, evaluator.data.num_test // 2)

        def reformat_row(row, prompt):
            final_instruction = prompt.format(input=row["text"], output="")
            return {"prompt": final_instruction, "label": row["toxicity"] >= evaluator.data.toxicity_threshold}

        prompt = TOXIC_CLASSIFICATION_WITH_PROMPT if evaluator.use_prompt else TOXIC_CLASSIFICATION_NO_PROMPT
        train_subsample = train_subsample.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})
        test_subsample = test_subsample.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

        run_inference_new("hidden_state", hf_model, tokenizer, train_subsample, train_out_fn,
                          batch_size=evaluator.batch_size, use_chat_template=True)
        run_inference_new("hidden_state", hf_model, tokenizer, test_subsample, test_out_fn,
                          batch_size=evaluator.batch_size, use_chat_template=True)

        classifier_train_dataset = pd.read_json(train_out_fn, orient="records", lines=True)
        classifier_test_dataset = pd.read_json(test_out_fn, orient="records", lines=True)

        X_train = np.stack(classifier_train_dataset["hidden_state"])
        y_train = np.stack(classifier_train_dataset["label"])
        X_test = np.stack(classifier_test_dataset["hidden_state"])
        y_test = np.stack(classifier_test_dataset["label"])

        indices_train = np.random.permutation(len(X_train))
        X_train, y_train = X_train[indices_train], y_train[indices_train]
        indices_test = np.random.permutation(len(X_test))
        X_test, y_test = X_test[indices_test], y_test[indices_test]

        clf = LogisticRegression(class_weight="balanced", max_iter=5000)
        clf.fit(X_train, y_train)

        y_pred_test = clf.predict(X_test)
        y_prob_test = clf.predict_proba(X_test)[:, 1]

        roc_auc_scores.append(roc_auc_score(y_test, y_prob_test))

    metrics = {
        "Test ROC AUC": (np.mean(roc_auc_scores), sem(roc_auc_scores), roc_auc_scores),
    }

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write("Test Metrics (Mean ± StdErr):\n")
        for metric, (mean, stderr, raw) in metrics.items():
            f.write(f"{metric}: {mean:.4f} ± {stderr:.4f} | {raw}\n")


def in_distribution_perplexity_evaluator_dolma(hf_model, tokenizer, evaluator, out_dir):
    """
    evaluates the model on in-distribution toxic data. Four different types of perplexity analysis
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """

    dataset = read_dataset_to_hf(evaluator.data.name)["train"]
    os.makedirs(out_dir, exist_ok=True)

    save_evaluator_config_and_sample(evaluator, out_dir, tokenizer.decode(dataset[0]["input_ids"]))

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "stats.txt")
    print("saving to ", out_fn)

    # loop through the eval dataset and calculate the averaged perplexity
    ind = 0
    tot_loss = 0
    tot_tokens = 0
    p_bar = tqdm(total=len(dataset))

    while (ind < len(dataset)):
        prompts = torch.tensor(dataset[ind:ind + evaluator.batch_size]["input_ids"]).to("cuda")

        logits = obtain_logit(hf_model, input_ids=prompts, attention_mask=torch.ones_like(prompts).to("cuda"))
        labels = prompts.clone().cpu()
        cross_entropy_per_token = calculate_loss_across_tokens(logits, labels, shift=True)

        # select loss_mask and take out first token since it is not being predicted
        loss_mask = torch.tensor(dataset[ind:ind + evaluator.batch_size]["loss_mask"])
        loss_mask = loss_mask[:, 1:]

        target_loss = cross_entropy_per_token[loss_mask == 1]
        tot_loss += torch.sum(target_loss).item()
        tot_tokens += torch.sum(loss_mask).item()

        ind += evaluator.batch_size
        p_bar.update(evaluator.batch_size)

    with open(out_fn, "w") as f:
        f.write(f"Perplexity: {torch.exp(torch.tensor(tot_loss / tot_tokens)).item()}, Loss: {tot_loss / tot_tokens}")


def in_distribution_perplexity_evaluator_nontoxicdocumentreddit(hf_model, tokenizer, evaluator, out_dir):
    """
    evaluates the model on in-distribution toxic data. Four different types of perplexity analysis
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """

    # read in huggingface dataset
    # we select 10000 sequences (corresponding to about 20 million tokens to eval on
    dataset = load_from_disk(evaluator.data.name).shuffle(seed=evaluator.seed).select(range(10000))

    os.makedirs(out_dir, exist_ok=True)

    save_evaluator_config_and_sample(evaluator, out_dir, tokenizer.decode(dataset[0]["input_ids"]))

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "stats.txt")
    print("saving to ", out_fn)

    # loop through the eval dataset and calculate the averaged perplexity
    ind = 0
    tot_loss = 0
    tot_tokens = 0
    p_bar = tqdm(total=len(dataset))

    while (ind < len(dataset)):
        prompts = torch.tensor(dataset[ind:ind + evaluator.batch_size]["input_ids"]).to("cuda")

        logits = obtain_logit(hf_model, input_ids=prompts, attention_mask=torch.ones_like(prompts).to("cuda"))
        labels = prompts.clone().cpu()
        cross_entropy_per_token = calculate_loss_across_tokens(logits, labels, shift=True)

        # select loss_mask and take out first token since it is not being predicted
        loss_mask = torch.tensor(dataset[ind:ind + evaluator.batch_size]["loss_mask"])
        loss_mask = loss_mask[:, 1:]

        target_loss = cross_entropy_per_token[loss_mask == 1]
        tot_loss += torch.sum(target_loss).item()
        tot_tokens += torch.sum(loss_mask).item()

        ind += evaluator.batch_size
        p_bar.update(evaluator.batch_size)

    with open(out_fn, "w") as f:
        f.write(
            f"Perplexity: {torch.exp(torch.tensor(tot_loss / tot_tokens)).item()}, Loss: {tot_loss / tot_tokens}")

def tofu_custom_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """Evaluates on tofu using custom metrics"""

    # load the dataset and select the necessary ones
    dataset = read_dataset_to_hf(evaluator.data.name, name="retain_perturbed")["train"]

    # reformat the dataset such that it is in generation format
    def reformat_row(row, format):
        question = row["question"]

        prompt = format.format(question=question)
        label = row["answer"]

        return {"prompt": prompt,
                "label": label}

    format = TOFU_QUERY_TEMPLATE
    dataset = dataset.map(reformat_row, fn_kwargs={"format": format}, batched=False)

    save_evaluator_config_and_sample(evaluator, out_dir, dataset[0]["prompt"])

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "generation_output_test.jsonl")
    print("saving to ", out_fn)

    run_inference_new("generate", hf_model, tokenizer, dataset, out_fn, batch_size=evaluator.batch_size,
                      generation_kwargs=evaluator.generation_kwargs)

    # # create tofu_names as a copy of the original tofu names
    # tofu_names = TOFU_NAMES.copy()
    # # add the first and last names of the characters to the tofu names
    # for name in TOFU_NAMES:
    #     tofu_names.append(name.split(" ")[0])
    #     tofu_names.append(name.split(" ")[-1])
    #
    # # reformat the dataset
    # def reformat_row(row, prompt, tokenizer, tofu_names, add_label=False):
    #     question = row["paraphrased_question"]
    #     correct_answer = row["paraphrased_answer"]
    #     incorrect_answers = row["perturbed_answer"]
    #
    #     prompt = prompt.format(question=question)
    #
    #     return {"prompt": prompt}

        # # this is used to test how well the model "understands"
        # correct_full = prompt.format(question=question, answer=correct_answer)
        # incorrect_full = [prompt.format(question=question, answer=incorrect_answer) for incorrect_answer in incorrect_answers]
        #
        # # the following is used to test how often model "generates" the entity name
        # # select the first occurance of a name in the correct_answer
        # earliest_index = -1
        # chosen_name = None
        # for name in tofu_names:
        #     pos = correct_answer.find(name)
        #     if pos == -1:
        #         continue
        #     if earliest_index == -1 or pos < earliest_index:
        #         earliest_index = pos
        #         chosen_name = name
        #
        # # this is used to check if model will generate the name of the entity
        # if chosen_name == None:
        #     generation_full = ""
        #     generation_name = ""
        # else:
        #     generation_full = prompt.format(question=question, answer=correct_answer[:earliest_index + len(chosen_name)])
        #     generation_name = chosen_name
        #
        # return {"correct_full": correct_full, "incorrect_full": incorrect_full, "question": question, "generation_full": generation_full, "generation_name": generation_name}

    # prompt = TOFU_TEMPLATE
    #
    # query_dataset = query_dataset.map(reformat_row, batched=False, num_proc=1,
    #                                   fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True,\
    #                                              "tofu_names": tofu_names},
    #                                   remove_columns=query_dataset.column_names)
    #
    # save_evaluator_config_and_sample(evaluator, out_dir, query_dataset[0]["correct_full"])
    #
    # # runs the generation and saves the output
    # out_fn_balanced = os.path.join(out_dir, "losses.jsonl")
    # print(f"saving to {out_fn_balanced}")
    #
    # # open a new file
    # out_file = open(out_fn_balanced, "w")
    #
    #
    # def calculate_answer_loss(model, tokenizer, full_prompt, question):
    #     """ given a full_prompt, calculate loss on the answer"""
    #     model_inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    #     question_input_ids = tokenizer(question).input_ids
    #
    #     # forward pass and get tokens
    #     logits = obtain_logit(model, **model_inputs)
    #
    #     # get the loss on each token
    #     labels = model_inputs["input_ids"].clone().cpu()
    #     cross_entropy_per_token = calculate_loss_across_tokens(logits, labels, shift=True)[0]
    #
    #     answer_loss = cross_entropy_per_token[len(question_input_ids):].mean()
    #
    #     # # we take the root to power of number of answer tokens
    #     # num_answer_tokens = len(cross_entropy_per_token) - len(question_input_ids)
    #     # answer_loss = answer_loss ** (1/num_answer_tokens)
    #
    #     return answer_loss
    #
    #
    # for entry in tqdm(query_dataset):
    #     correct_loss = calculate_answer_loss(hf_model, tokenizer, entry["correct_full"], entry["question"])
    #     incorrect_losses = [calculate_answer_loss(hf_model, tokenizer, incorrect_full, entry["question"]) for incorrect_full in entry["incorrect_full"]]
    #
    #     correct_loss = correct_loss.item()
    #     incorrect_losses = [loss.item() for loss in incorrect_losses]
    #
    #     if correct_loss < min(incorrect_losses):
    #         rank = 1
    #     else:
    #         # sort the incorrect losses
    #         losses_sort = sorted(incorrect_losses + [correct_loss])
    #         rank = losses_sort.index(correct_loss) + 1
    #
    #     # we now do generation analysis
    #     generation_full = entry["generation_full"]
    #     generation_name = entry["generation_name"]
    #     if generation_full == "":
    #         name_loss = -1000
    #         name_rank = -1000
    #         out_file.write(json.dumps({"correct": correct_loss, "incorrect": incorrect_losses, "rank_of_correct": rank, "name_loss": name_loss, "name_rank": name_rank}) + "\n")
    #         continue
    #     generation_full_input_ids = tokenizer(generation_full, return_tensors="pt").to("cuda")
    #     name_input_ids = tokenizer(" " + generation_name).input_ids
    #
    #     # forward pass and get tokens
    #     logits = obtain_logit(hf_model, **generation_full_input_ids)
    #
    #     # get the loss on each token
    #     labels = generation_full_input_ids["input_ids"].clone().cpu()
    #     cross_entropy_per_token = calculate_loss_across_tokens(logits, labels, shift=True)[0]
    #
    #     # we get the rank of name
    #     name_loss = cross_entropy_per_token[-len(name_input_ids)].mean().item()
    #     name_token_logits = logits[0][-len(name_input_ids)]
    #     # we now get the rank of the name
    #     name_rank = (name_token_logits > name_token_logits[name_input_ids[0]]).sum().item() + 1
    #
    #     out_file.write(json.dumps({"correct": correct_loss, "incorrect": incorrect_losses, "rank_of_correct": rank, "name_loss": name_loss, "name_rank": name_rank}) + "\n")


def evaluate_model_with_single_evaluators(hf_model, tokenizer, evaluator, out_dir):
    """
    Evaluates the model using a single evaluator.
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """

    # reset the seed for each evaluator
    seed_all(evaluator.seed)

    if "realtoxicityprompts_generation" in evaluator.label:
        real_toxicity_prompt_generation_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_civilcomments_hiddenstate" in evaluator.label:
        NEW_hidden_state_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_CHAT_civilcomments_hiddenstate" in evaluator.label:
        NEW_CHAT_hidden_state_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "tofu_custom" in evaluator.label:
        tofu_custom_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "in_distribution_perplexity_dolma" in evaluator.label:
        in_distribution_perplexity_evaluator_dolma(hf_model, tokenizer, evaluator, out_dir)
    elif "in_distribution_perplexity_nontoxicdocumentreddit" in evaluator.label:
        in_distribution_perplexity_evaluator_nontoxicdocumentreddit(hf_model, tokenizer, evaluator, out_dir)

def evaluate_model_with_multiple_evaluators(hf_model, tokenizer, evaluators, model_dir, out_dir=None):
    """
    Evaluates the model using a list of evaluators.
    :param hf_model: the loaded model
    :param tokenizer: the tokenizer
    :param evaluators: the list of evaluators
    :param out_dir: the directory of the model that we will output our directories into
    :return: nothing
    """


    for evaluator in evaluators:
        if out_dir is None:
            evaluator_out_dir = os.path.join(model_dir, evaluator.label)
        else:
            evaluator_out_dir = os.path.join(out_dir, evaluator.label)
        os.makedirs(evaluator_out_dir, exist_ok=True)
        evaluate_model_with_single_evaluators(hf_model, tokenizer, evaluator, evaluator_out_dir)

