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
from src.modules.templates import CIVIL_COMMENTS_TEMPLATE_NO_LABELS, CIVIL_COMMENTS_FINEGRAINED_TEMPLATE_NO_LABELS, \
    CIVIL_COMMENTS_FINEGRAINED_LABELS, TOXIC_CLASSIFICATION_WITH_PROMPT, \
    TOXIC_CLASSIFICATION_NO_PROMPT, NLI_CLASSIFICATION_WITH_PROMPT, NLI_CLASSIFICATION_NO_PROMPT, \
    NLI_CLASSIFICATION_WITH_PROMPT_CHINESE, SQUAD_TEMPLATE_WITH_LABELS, SQUAD_TEMPLATE_NO_LABELS, SQUAD_INSTRUCTIOIN, \
    TOFU_NAMES, TOFU_TEMPLATE, TOFU_QUERY_TEMPLATE, TOXIC_CLASSIFICATION_NOISY_CHANNELS_1, \
    TOXIC_CLASSIFICATION_NOISY_CHANNELS_1_CHOICE, TOXIC_CLASSIFICATION_NOISY_CHANNELS_2, \
    TOXIC_CLASSIFICATION_NOISY_CHANNELS_2_CHOICE, TOXIC_CLASSIFICATION_NOISY_CHANNELS_3, \
    TOXIC_CLASSIFICATION_NOISY_CHANNELS_3_CHOICE, TOXIC_CLASSIFICATION_NOISY_CHANNELS_4, \
    TOXIC_CLASSIFICATION_NOISY_CHANNELS_4_CHOICE, TOXIC_CLASSIFICATION_NOISY_CHANNELS_5, \
    TOXIC_CLASSIFICATION_NOISY_CHANNELS_5_CHOICE, TOXIC_CLASSIFICATION_NOISY_CHANNELS_6, \
    TOXIC_CLASSIFICATION_NOISY_CHANNELS_6_CHOICE, TOXIC_CLASSIFICATION_NOISY_CHANNELS_7, \
    TOXIC_CLASSIFICATION_NOISY_CHANNELS_7_CHOICE
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
    # progress_file = os.path.join(out_dir, "perspective_api_progress_includingprompt.json")
    # use_perspective_api(out_fn, PERSPECTIVE_API_KEY, progress_file)


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

    accuracy_scores, f1_scores, precision_scores, recall_scores, roc_auc_scores, pr_auc_scores = [], [], [], [], [], []

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

        accuracy_scores.append(accuracy_score(y_test, y_pred_test))
        f1_scores.append(f1_score(y_test, y_pred_test))
        precision_scores.append(precision_score(y_test, y_pred_test))
        recall_scores.append(recall_score(y_test, y_pred_test))
        roc_auc_scores.append(roc_auc_score(y_test, y_prob_test))
        pr_auc_scores.append(average_precision_score(y_test, y_prob_test))

    metrics = {
        "Test Accuracy": (np.mean(accuracy_scores), sem(accuracy_scores), accuracy_scores),
        "Test F1 Score": (np.mean(f1_scores), sem(f1_scores), f1_scores),
        "Test Precision": (np.mean(precision_scores), sem(precision_scores), precision_scores),
        "Test Recall": (np.mean(recall_scores), sem(recall_scores), recall_scores),
        "Test ROC AUC": (np.mean(roc_auc_scores), sem(roc_auc_scores), roc_auc_scores),
        "Test PR AUC": (np.mean(pr_auc_scores), sem(pr_auc_scores), pr_auc_scores),
    }

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write("Test Metrics (Mean ± StdErr):\n")
        for metric, (mean, stderr, raw) in metrics.items():
            f.write(f"{metric}: {mean:.4f} ± {stderr:.4f} | {raw}\n")


def NEW_noisychannel_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir):
    test_out_fn = os.path.join(out_dir, "instances.jsonl")

    dataset = read_dataset_to_hf(evaluator.data.name).shuffle(seed=evaluator.seed)
    test_dataset = dataset["test"]

    accuracy_scores = []

    for i in range(evaluator.data.num_samples):  # Generate different subsamples
        test_subsample = select_binary_balanced_dataset(
            test_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
            evaluator.seed + i, evaluator.data.num_test // 2)

        def reformat_row(row, prompt, choices):
            text = row["text"][0]
            toxicity = row["toxicity"][0]
            input_length = len(text)

            # we create a true and false instance to get loss over
            final_instruction_true = prompt.format(choice=choices[True], input=text)
            target_mask_true = [0] * (len(final_instruction_true) - input_length) + [1] * input_length

            final_instruction_false = prompt.format(choice=choices[False], input=text)
            target_mask_false = [0] * (len(final_instruction_false) - input_length) + [1] * input_length

            return {"prompt": [final_instruction_false, final_instruction_true], "label": [toxicity >= evaluator.data.toxicity_threshold] * 2, "target_mask": [target_mask_false, target_mask_true]}

        if evaluator.prompt_version == 1:
            prompt = TOXIC_CLASSIFICATION_NOISY_CHANNELS_1
            choices = TOXIC_CLASSIFICATION_NOISY_CHANNELS_1_CHOICE
        elif evaluator.prompt_version == 2:
            prompt = TOXIC_CLASSIFICATION_NOISY_CHANNELS_2
            choices = TOXIC_CLASSIFICATION_NOISY_CHANNELS_2_CHOICE
        elif evaluator.prompt_version == 3:
            prompt = TOXIC_CLASSIFICATION_NOISY_CHANNELS_3
            choices = TOXIC_CLASSIFICATION_NOISY_CHANNELS_3_CHOICE
        elif evaluator.prompt_version == 4:
            prompt = TOXIC_CLASSIFICATION_NOISY_CHANNELS_4
            choices = TOXIC_CLASSIFICATION_NOISY_CHANNELS_4_CHOICE
        elif evaluator.prompt_version == 5:
            prompt = TOXIC_CLASSIFICATION_NOISY_CHANNELS_5
            choices = TOXIC_CLASSIFICATION_NOISY_CHANNELS_5_CHOICE
        elif evaluator.prompt_version == 6:
            prompt = TOXIC_CLASSIFICATION_NOISY_CHANNELS_6
            choices = TOXIC_CLASSIFICATION_NOISY_CHANNELS_6_CHOICE
        elif evaluator.prompt_version == 7:
            prompt = TOXIC_CLASSIFICATION_NOISY_CHANNELS_7
            choices = TOXIC_CLASSIFICATION_NOISY_CHANNELS_7_CHOICE
        else:
            raise ValueError("Invalid prompt version")

        test_subsample = test_subsample.map(reformat_row, batched=True, batch_size=1, remove_columns=test_subsample.column_names, \
                                            fn_kwargs={"prompt": prompt, "choices": choices})

        run_inference_new("get_loss", hf_model, tokenizer, test_subsample, test_out_fn,\
                          batch_size=evaluator.batch_size)

        df = pd.read_json(test_out_fn, orient="records", lines=True)

        # loop through the pandas dataframe
        num_correct = 0
        for i in range(0, len(df), 2):
            # make sure they're the paired examples
            if df.iloc[i]["label"] != df.iloc[i + 1]["label"]:
                print(f"Error: {df.iloc[i]['label']} != {df.iloc[i + 1]['label']} or {df.iloc[i]['prompt'][-10:]} != {df.iloc[i + 1]['prompt'][-10:]}")
                breakpoint()
            # assert df.iloc[i]["label"] == df.iloc[i + 1]["label"]
            # assert df.iloc[i]["prompt"][-10:] == df.iloc[i + 1]["prompt"][-10:]

            if df.iloc[i]["label"]:
                # i is for false and i+1 is for true
                if df.iloc[i]["loss"] > df.iloc[i + 1]["loss"]:
                    num_correct += 1
            else:
                if df.iloc[i]["loss"] < df.iloc[i + 1]["loss"]:
                    num_correct += 1
        accuracy_scores.append(num_correct / (len(df) // 2))

    metrics = {
        "Test Accuracy": (np.mean(accuracy_scores), sem(accuracy_scores), accuracy_scores),
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

    accuracy_scores, f1_scores, precision_scores, recall_scores, roc_auc_scores, pr_auc_scores = [], [], [], [], [], []

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

        accuracy_scores.append(accuracy_score(y_test, y_pred_test))
        f1_scores.append(f1_score(y_test, y_pred_test))
        precision_scores.append(precision_score(y_test, y_pred_test))
        recall_scores.append(recall_score(y_test, y_pred_test))
        roc_auc_scores.append(roc_auc_score(y_test, y_prob_test))
        pr_auc_scores.append(average_precision_score(y_test, y_prob_test))

    metrics = {
        "Test Accuracy": (np.mean(accuracy_scores), sem(accuracy_scores), accuracy_scores),
        "Test F1 Score": (np.mean(f1_scores), sem(f1_scores), f1_scores),
        "Test Precision": (np.mean(precision_scores), sem(precision_scores), precision_scores),
        "Test Recall": (np.mean(recall_scores), sem(recall_scores), recall_scores),
        "Test ROC AUC": (np.mean(roc_auc_scores), sem(roc_auc_scores), roc_auc_scores),
        "Test PR AUC": (np.mean(pr_auc_scores), sem(pr_auc_scores), pr_auc_scores),
    }

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write("Test Metrics (Mean ± StdErr):\n")
        for metric, (mean, stderr, raw) in metrics.items():
            f.write(f"{metric}: {mean:.4f} ± {stderr:.4f} | {raw}\n")

def hidden_state_civilcomments_insult_evaluator(hf_model, tokenizer, evaluator, out_dir):
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    eval_out_fn = os.path.join(out_dir, "hidden_states_eval.jsonl")
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")

    if not os.path.exists(train_out_fn) or not os.path.exists(eval_out_fn) or not os.path.exists(test_out_fn):

        #load the dataset and select balanced partitions
        dataset = read_dataset_to_hf(evaluator.data.name)["train"].shuffle(seed=evaluator.seed)
        tot_examples = evaluator.data.num_train + evaluator.data.num_eval + evaluator.data.num_test

        assert (len(dataset) >= tot_examples)
        # we first make sure our dataset contains only toxic text
        dataset = dataset.filter(lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold)

        # we next filter out examples that are "in the middle" -> cannot assign a direct label to
        dataset = dataset.filter(lambda x: x["insult"] <= evaluator.data.insult_lowerbound or x["insult"] >= evaluator.data.insult_upperbound)

        def labeler(x):
            # this method determines whether eaach row is a positive or negative class
            return x["insult"] > evaluator.data.insult_lowerbound


        # now we select the balanced dataset
        dataset = select_binary_balanced_dataset(dataset, lambda x: labeler(x), evaluator.seed, tot_examples // 2)

        train_dataset = dataset.select(range(evaluator.data.num_train))
        eval_dataset = dataset.select(range(evaluator.data.num_train, evaluator.data.num_train + evaluator.data.num_eval))
        test_dataset = dataset.select(range(evaluator.data.num_train + evaluator.data.num_eval, \
                                            evaluator.data.num_train + evaluator.data.num_eval + evaluator.data.num_test))

        # reformat the dataset such that it is in generation format
        def reformat_row(row, prompt):
            final_instruction = prompt.format(input=row["text"], output="")
            return {"prompt": final_instruction,
                    "label": labeler(row)}

        if evaluator.use_prompt:
            prompt = TOXIC_CLASSIFICATION_WITH_PROMPT
        else:
            prompt = TOXIC_CLASSIFICATION_NO_PROMPT

        train_dataset = train_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})
        eval_dataset = eval_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})
        test_dataset = test_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

        save_evaluator_config_and_sample(evaluator, out_dir, train_dataset[0]["prompt"])

        # runs the hidden states and saves the output for both train and eval
        run_inference_new("hidden_state", hf_model, tokenizer, train_dataset, train_out_fn, batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, eval_dataset, eval_out_fn, batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, test_dataset, test_out_fn, batch_size=evaluator.batch_size)


    # load the dataset into numpy format
    classifier_train_dataset = pd.read_json(train_out_fn, orient="records", lines=True)
    classifier_eval_dataset = pd.read_json(eval_out_fn, orient="records", lines=True)
    classifier_test_dataset = pd.read_json(test_out_fn, orient="records", lines=True)

    X_train = np.stack(classifier_train_dataset["hidden_state"])
    y_train = np.stack(classifier_train_dataset["label"])

    X_eval = np.stack(classifier_eval_dataset["hidden_state"])
    y_eval = np.stack(classifier_eval_dataset["label"])

    X_test = np.stack(classifier_test_dataset["hidden_state"])
    y_test = np.stack(classifier_test_dataset["label"])

    # shuffle the data
    indices_train = np.random.permutation(len(X_train))
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    indices_eval = np.random.permutation(len(X_eval))
    X_eval = X_eval[indices_eval]
    y_eval = y_eval[indices_eval]

    indices_test = np.random.permutation(len(X_test))
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]

    # Train a classifier for each label (binary relevance)
    print(f"Training classifier")

    # # Train the logistic regression model
    clf = LogisticRegression(class_weight="balanced", max_iter=5000)
    clf.fit(X_train, y_train)

    # Predict on the eval set for the current label
    y_pred_eval = clf.predict(X_eval)

    # Compute the F1 score for the current label
    f1_eval = f1_score(y_eval, y_pred_eval)
    acc_eval = accuracy_score(y_eval, y_pred_eval)
    print(f"dev F1 Score: {f1_eval:.4f}")
    print(f"dev Accuracy: {acc_eval:.4f}")

    # Predict on the test set for the current label
    y_pred_test = clf.predict(X_test)

    # Compute the F1 score for the current label
    f1_test = f1_score(y_test, y_pred_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"test F1 Score: {f1_test:.4f}")
    print(f"test Accuracy: {acc_test:.4f}")

    # Save the performance metrics to a file
    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(f"Dev F1 Score: {f1_eval:.4f}\n")
        f.write(f"Dev Accuracy: {acc_eval:.4f}\n")
        f.write(f"Test F1 Score: {f1_test:.4f}\n")
        f.write(f"Test Accuracy: {acc_test:.4f}\n")


def hidden_state_civilcomments_finegrained_evaluator(hf_model, tokenizer, evaluator, out_dir):

    #load the dataset and select toxic portions
    dataset = read_dataset_to_hf(evaluator.data.name)["train"].shuffle(seed=evaluator.seed)
    dataset = dataset.filter(lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold)

    train_dataset = dataset.select(range(evaluator.data.num_train))
    eval_dataset = dataset.select(range(evaluator.data.num_train, evaluator.data.num_train + evaluator.data.num_eval))
    test_dataset = dataset.select(range(evaluator.data.num_train + evaluator.data.num_eval, \
                                        evaluator.data.num_train + evaluator.data.num_eval + evaluator.data.num_test))

    # reformat the dataset such that it is in generation format
    def reformat_row(row):
        final_instruction = CIVIL_COMMENTS_FINEGRAINED_TEMPLATE_NO_LABELS.format(input=row["text"], output="")
        return {"prompt": final_instruction,
                "label": [row["severe_toxicity"], row["obscene"], row["threat"], row["insult"], row["identity_attack"], row["sexual_explicit"]]}

    train_dataset = train_dataset.map(reformat_row, batched=False)
    eval_dataset = eval_dataset.map(reformat_row, batched=False)
    test_dataset = test_dataset.map(reformat_row, batched=False)

    save_evaluator_config_and_sample(evaluator, out_dir, train_dataset[0]["prompt"])

    # runs the hidden states and saves the output for both train and eval
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    run_inference_new("hidden_state", hf_model, tokenizer, train_dataset, train_out_fn, batch_size=evaluator.batch_size)
    eval_out_fn = os.path.join(out_dir, "hidden_states_eval.jsonl")
    run_inference_new("hidden_state", hf_model, tokenizer, eval_dataset, eval_out_fn, batch_size=evaluator.batch_size)
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")
    run_inference_new("hidden_state", hf_model, tokenizer, test_dataset, test_out_fn, batch_size=evaluator.batch_size)

    classifier_train_dataset = PandasDataset(pd.read_json(train_out_fn, orient="records", lines=True))
    classifier_eval_dataset = PandasDataset(pd.read_json(eval_out_fn, orient="records", lines=True))
    classifier_test_dataset = PandasDataset(pd.read_json(test_out_fn, orient="records", lines=True))


    train_binaryclassifier_multi(classifier_train_dataset, classifier_eval_dataset, classifier_test_dataset, evaluator.binary_classifier.epochs,\
                                    evaluator.binary_classifier.batch_size, out_dir, metric_func=torch.nn.CrossEntropyLoss())

    pass

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

def in_distribution_perplexity_evaluator_nontoxicreddit(hf_model, tokenizer, evaluator, out_dir):
    """
    evaluates the model on in-distribution toxic data. Four different types of perplexity analysis
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """

    # read in huggingface dataset
    dataset = load_from_disk(evaluator.data.name)

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

    # # load the dataset and select the necessary ones
    # # dataset_names = ["base_filtered", "nontoxic_only", "nontoxic_toxic", "toxic_nontoxic", "toxic_only"]
    # dataset_names = ["base_filtered"]
    # dataset_arr = [read_dataset_to_hf(os.path.join(evaluator.data.name, name, "data.jsonl"))["train"].select(range(evaluator.data.num_examples)) for name in dataset_names]
    #
    # # we assume we use all examples in the test data
    #
    # # loop over each type of evaluation data
    # for dataset_ind in range(len(dataset_arr)):
    #     dataset = dataset_arr[dataset_ind]
    #     dataset_name = dataset_names[dataset_ind]
    #
    #     current_out_dir = os.path.join(out_dir, dataset_name)
    #     os.makedirs(current_out_dir, exist_ok=True)
    #
    #     save_evaluator_config_and_sample(evaluator, current_out_dir, tokenizer.decode(dataset[0]["input_ids"]))
    #
    #     # runs the generation and saves the output
    #     out_fn = os.path.join(current_out_dir, "stats.txt")
    #     print("saving to ", out_fn)
    #
    #     # loop through the eval dataset and calculate the averaged perplexity
    #     ind = 0
    #     tot_loss = 0
    #     tot_tokens = 0
    #     p_bar = tqdm(total=len(dataset))
    #
    #     while (ind < len(dataset)):
    #         prompts = torch.tensor(dataset[ind:ind + evaluator.batch_size]["input_ids"]).to("cuda")
    #
    #         logits = obtain_logit(hf_model, input_ids=prompts, attention_mask=torch.ones_like(prompts).to("cuda"))
    #         labels = prompts.clone().cpu()
    #         cross_entropy_per_token = calculate_loss_across_tokens(logits, labels, shift=True)
    #
    #         # select loss_mask and take out first token since it is not being predicted
    #         loss_mask = torch.tensor(dataset[ind:ind + evaluator.batch_size]["loss_mask"])
    #         loss_mask = loss_mask[:, 1:]
    #
    #         target_loss = cross_entropy_per_token[loss_mask == 1]
    #         tot_loss += torch.sum(target_loss).item()
    #         tot_tokens += torch.sum(loss_mask).item()
    #
    #         ind += evaluator.batch_size
    #         p_bar.update(evaluator.batch_size)
    #
    #     with open(out_fn, "w") as f:
    #         f.write(f"Perplexity: {torch.exp(torch.tensor(tot_loss / tot_tokens)).item()}, Loss: {tot_loss / tot_tokens}")

def NEW_EASY_hidden_state_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir):
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")

    if not os.path.exists(train_out_fn) or not os.path.exists(test_out_fn):
        # load the dataset and select balanced partitions
        train_dataset = read_dataset_to_hf(evaluator.data.name, name="annotated")["train"].shuffle(seed=evaluator.seed)
        test_dataset = read_dataset_to_hf(evaluator.data.name, name="annotated")["test"].shuffle(seed=evaluator.seed)

        # we first filter the datasetes to only include extreme examples (which makes the task easier
        train_dataset = train_dataset.filter(lambda x: x["toxicity_human"] >= 4.5 or x["toxicity_human"] <= 1.5)
        test_dataset = test_dataset.filter(lambda x: x["toxicity_human"] >= 4.5 or x["toxicity_human"] <= 1.5)

        train_dataset = select_binary_balanced_dataset(train_dataset, lambda x: x["toxicity_human"] >= 3,
                                                 evaluator.seed, 3054 // 2) # 3054 is max number supported for balanced dataset

        test_dataset = select_binary_balanced_dataset(test_dataset, lambda x: x["toxicity_human"] >= 3,
                                                      evaluator.seed, 374 // 2) # 374 is max number supported for balanced dataset

        # reformat the dataset such that it is in generation format
        def reformat_row(row, prompt):
            final_instruction = prompt.format(input=row["text"], output="")
            return {"prompt": final_instruction,
                    "label": row["toxicity_human"] >= 3}

        if evaluator.use_prompt:
            prompt = TOXIC_CLASSIFICATION_WITH_PROMPT
        else:
            prompt = TOXIC_CLASSIFICATION_NO_PROMPT

        train_dataset = train_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})
        test_dataset = test_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

        save_evaluator_config_and_sample(evaluator, out_dir, train_dataset[0]["prompt"])

        # runs the hidden states and saves the output for both train and eval
        run_inference_new("hidden_state", hf_model, tokenizer, train_dataset, train_out_fn,
                          batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, test_dataset, test_out_fn,
                          batch_size=evaluator.batch_size)


    # load the dataset into numpy format
    classifier_train_dataset = pd.read_json(train_out_fn, orient="records", lines=True)
    classifier_test_dataset = pd.read_json(test_out_fn, orient="records", lines=True)

    X_train = np.stack(classifier_train_dataset["hidden_state"])
    y_train = np.stack(classifier_train_dataset["label"])

    X_test = np.stack(classifier_test_dataset["hidden_state"])
    y_test = np.stack(classifier_test_dataset["label"])

    # shuffle the data
    indices_train = np.random.permutation(len(X_train))
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    indices_test = np.random.permutation(len(X_test))
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]

    # Train a classifier for each label (binary relevance)
    print(f"Training classifier")

    # # Train the logistic regression model
    clf = LogisticRegression(class_weight="balanced", max_iter=5000)
    clf.fit(X_train, y_train)

    # Predict on the test set for the current label
    y_pred_test = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)[:, 1]

    # Compute the F1 score for the current label
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_pred_test, average="binary")
    roc_auc_test = roc_auc_score(y_test, y_prob_test)
    pr_auc_test = average_precision_score(y_test, y_prob_test)
    print(f"test accuracy: {accuracy_test:.4f}")
    print(f"test F1 Score: {f1_test:.4f}")
    print(f"test Precision: {precision_test:.4f}")
    print(f"test Recall: {recall_test:.4f}")
    print(f"test ROC AUC: {roc_auc_test:.4f}")
    print(f"test PR AUC: {pr_auc_test:.4f}")

    # Save the performance metrics to a file
    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy_test:.4f}\n")
        f.write(f"Test F1 Score: {f1_test:.4f}\n")
        f.write(f"Test Precision: {precision_test:.4f}\n")
        f.write(f"Test Recall: {recall_test:.4f}\n")
        f.write(f"Test ROC AUC: {roc_auc_test:.4f}\n")
        f.write(f"Test PR AUC: {pr_auc_test:.4f}\n")

def NEW_hidden_state_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir):
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")

    if not os.path.exists(train_out_fn) or not os.path.exists(test_out_fn):
        # load the dataset and select balanced partitions
        dataset = read_dataset_to_hf(evaluator.data.name, name="annotated")["train"].shuffle(seed=evaluator.seed)
        dataset = select_binary_balanced_dataset(dataset, lambda x: x["toxicity_human"] >= 3,
                                                 evaluator.seed, 6766 // 2) # 6766 is max number supported for balanced dataset

        train_dataset = dataset

        test_dataset = read_dataset_to_hf(evaluator.data.name, name="annotated")["test"]
        test_dataset = select_binary_balanced_dataset(test_dataset, lambda x: x["toxicity_human"] >= 3,
                                                      evaluator.seed, 804 // 2) # 804 is max number supported for balanced dataset

        # reformat the dataset such that it is in generation format
        def reformat_row(row, prompt):
            final_instruction = prompt.format(input=row["text"], output="")
            return {"prompt": final_instruction,
                    "label": row["toxicity_human"] >= 3}

        if evaluator.use_prompt:
            prompt = TOXIC_CLASSIFICATION_WITH_PROMPT
        else:
            prompt = TOXIC_CLASSIFICATION_NO_PROMPT

        train_dataset = train_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})
        test_dataset = test_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

        save_evaluator_config_and_sample(evaluator, out_dir, train_dataset[0]["prompt"])

        # runs the hidden states and saves the output for both train and eval
        run_inference_new("hidden_state", hf_model, tokenizer, train_dataset, train_out_fn,
                          batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, test_dataset, test_out_fn,
                          batch_size=evaluator.batch_size)


    # load the dataset into numpy format
    classifier_train_dataset = pd.read_json(train_out_fn, orient="records", lines=True)
    classifier_test_dataset = pd.read_json(test_out_fn, orient="records", lines=True)

    X_train = np.stack(classifier_train_dataset["hidden_state"])
    y_train = np.stack(classifier_train_dataset["label"])

    X_test = np.stack(classifier_test_dataset["hidden_state"])
    y_test = np.stack(classifier_test_dataset["label"])

    # shuffle the data
    indices_train = np.random.permutation(len(X_train))
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    indices_test = np.random.permutation(len(X_test))
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]

    # Train a classifier for each label (binary relevance)
    print(f"Training classifier")

    # # Train the logistic regression model
    clf = LogisticRegression(class_weight="balanced", max_iter=5000)
    clf.fit(X_train, y_train)

    # Predict on the test set for the current label
    y_pred_test = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)[:, 1]

    # Compute the F1 score for the current label
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, y_pred_test, average="binary")
    roc_auc_test = roc_auc_score(y_test, y_prob_test)
    pr_auc_test = average_precision_score(y_test, y_prob_test)
    print(f"test accuracy: {accuracy_test:.4f}")
    print(f"test F1 Score: {f1_test:.4f}")
    print(f"test Precision: {precision_test:.4f}")
    print(f"test Recall: {recall_test:.4f}")
    print(f"test ROC AUC: {roc_auc_test:.4f}")
    print(f"test PR AUC: {pr_auc_test:.4f}")

    # Save the performance metrics to a file
    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy_test:.4f}\n")
        f.write(f"Test F1 Score: {f1_test:.4f}\n")
        f.write(f"Test Precision: {precision_test:.4f}\n")
        f.write(f"Test Recall: {recall_test:.4f}\n")
        f.write(f"Test ROC AUC: {roc_auc_test:.4f}\n")
        f.write(f"Test PR AUC: {pr_auc_test:.4f}\n")

def hidden_state_xnli_evaluator(hf_model, tokenizer, evaluator, out_dir):
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    eval_out_fn = os.path.join(out_dir, "hidden_states_eval.jsonl")
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")

    if not os.path.exists(train_out_fn) or not os.path.exists(eval_out_fn) or not os.path.exists(test_out_fn):
        # load the dataset and select balanced partitions

        # we assume that training excludes last 100k examples
        end_examples = 50000
        dataset = read_dataset_to_hf(evaluator.data.name)["train"] # note we don't shuffle since we want to select held out set

        # we first select the held-out portion of the dataset
        begin_index = len(dataset) - end_examples
        dataset = dataset.select(range(begin_index, len(dataset))).shuffle(seed=evaluator.seed)

        tot_examples = evaluator.data.num_train + evaluator.data.num_eval + evaluator.data.num_test

        assert (len(dataset) >= tot_examples)

        # select balanced datasets
        lambdas_per_class = [lambda x, i = i: x["label"] == i for i in range(3)]
        dataset = select_n_ary_balanced_dataset(dataset, lambdas_per_class, evaluator.seed, (tot_examples // 3) + 1)

        train_dataset = dataset.select(range(evaluator.data.num_train))
        eval_dataset = dataset.select(
            range(evaluator.data.num_train, evaluator.data.num_train + evaluator.data.num_eval))
        test_dataset = dataset.select(range(evaluator.data.num_train + evaluator.data.num_eval, \
                                            evaluator.data.num_train + evaluator.data.num_eval + evaluator.data.num_test))

        # reformat the dataset such that it is in generation format
        def reformat_row(row, prompt):
            final_instruction = prompt.format(input1=row["premise"], input2=row["hypothesis"], output="")
            return {"prompt": final_instruction,
                    "label": row["label"]}

        if evaluator.use_prompt:
            if "chinese" in evaluator.label:
                prompt = NLI_CLASSIFICATION_WITH_PROMPT_CHINESE
            else:
                prompt = NLI_CLASSIFICATION_WITH_PROMPT
        else:
            prompt = NLI_CLASSIFICATION_NO_PROMPT

        train_dataset = train_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})
        eval_dataset = eval_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})
        test_dataset = test_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

        save_evaluator_config_and_sample(evaluator, out_dir, train_dataset[0]["prompt"])

        # runs the hidden states and saves the output for both train and eval
        run_inference_new("hidden_state", hf_model, tokenizer, train_dataset, train_out_fn,
                          batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, eval_dataset, eval_out_fn,
                          batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, test_dataset, test_out_fn,
                          batch_size=evaluator.batch_size)

    # trains the binary classifier and records the best accuracy
    hidden_size = hf_model.config.hidden_size

    classifier_train_dataset = PandasDataset(pd.read_json(train_out_fn, orient="records", lines=True))
    classifier_eval_dataset = PandasDataset(pd.read_json(eval_out_fn, orient="records", lines=True))
    classifier_test_dataset = PandasDataset(pd.read_json(test_out_fn, orient="records", lines=True))

    if evaluator.use_acc:
        acc_out_dir = os.path.join(out_dir, "acc")
        os.makedirs(acc_out_dir, exist_ok=True)
        classifier_model = TrinaryClassifier(input_dim=hidden_size).to("cuda")

        def accuracy_metric(logits, y):
            preds = torch.argmax(logits, dim=1)
            return torch.sum(preds == y).item() / len(y)

        best_dev_acc = train_classifier(classifier_model, classifier_train_dataset, classifier_eval_dataset,
                                        evaluator.binary_classifier.epochs, \
                                        evaluator.binary_classifier.batch_size, acc_out_dir,
                                        metric_func=accuracy_metric)

        # load the best model
        best_model = TrinaryClassifier(hidden_size)
        best_model.load_state_dict(torch.load(os.path.join(acc_out_dir, "tmp_best_model.pth")))
        best_model.to("cuda")

        # test the best model
        best_model.eval()
        with torch.no_grad():
            # num_correct = 0
            total_logits = []
            total_labels = []
            for X_test, y_test in DataLoader(classifier_test_dataset,
                                             batch_size=evaluator.binary_classifier.batch_size):
                test_logits = best_model(X_test.to("cuda")).to("cpu")
                total_logits.append(test_logits)
                total_labels.append(y_test)
                # test_preds = torch.argmax(test_logits, dim=1)
                # num_correct += torch.sum(test_preds == y_test).item()

            test_acc = accuracy_metric(torch.cat(total_logits), torch.cat(total_labels))
            # test_acc = num_correct / len(classifier_test_dataset)
            print(f'Test accuracy: {test_acc:.5f}')

        with open(os.path.join(acc_out_dir, "acc_stats.txt"), "w") as f:
            f.write(f"Best dev accuracy: {best_dev_acc}, test acc: {test_acc}")


def squad_generation_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
    Evaluates the model using SQUAD. We then score it using rouge
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """
    # load the dataset and select the necessary ones
    english_squad = read_dataset_to_hf(evaluator.data.english_name)["train"]

    # we only support one particular chinese_squad for now
    assert evaluator.data.chinese_name == "/mnt/nfs1/ryan/decouple/data/squad/train-v1.1-zh.json"

    # we write a generator that reads the dataset
    def chinese_dataset_generator(file):
        with open(file) as f:

            data = json.load(f)["data"]
            # gives general title of the subject
            for entry in data:
                title = entry["title"]
                paragraphs = entry["paragraphs"]
                # gives context
                for entry2 in paragraphs:
                    context = entry2["context"]
                    qas = entry2["qas"]
                    # for each context a list of question answers
                    for qa in qas:
                        formatted_answers = {"text": [qa["answers"][0]["text"]], "answer_start": [qa["answers"][0]["answer_start"]]}
                        yield {"id": qa["id"], "title": title, "context": context, "question": qa["question"], "answers": formatted_answers}

    chinese_squad = Dataset.from_generator(chinese_dataset_generator, gen_kwargs={"file": evaluator.data.chinese_name}).shuffle(seed=evaluator.seed)
    english_squad = read_dataset_to_hf(evaluator.data.english_name)["train"]

    # we build a mapper that goes from id to index in english squad
    id_to_index = {}
    for i, row in enumerate(english_squad):
        id_to_index[row["id"]] = i

    assert (len(chinese_squad) >= evaluator.data.num_examples)
    demonstration_set = chinese_squad.select(range(len(chinese_squad) - evaluator.data.num_demonstrations, len(chinese_squad)))
    chinese_squad = chinese_squad.select(range(evaluator.data.num_examples))

    INSTRUCTION = SQUAD_INSTRUCTIOIN
    DEMONSTRATION_TEMPLATE = SQUAD_TEMPLATE_WITH_LABELS
    QUESTION_TEMPLATE = SQUAD_TEMPLATE_NO_LABELS

    demonstration = INSTRUCTION + " "
    for example in demonstration_set:
        english_answer = english_squad[id_to_index[example["id"]]]["answers"]["text"][0]
        demonstration += DEMONSTRATION_TEMPLATE.format(title=example["title"], context=example["context"], question=example["question"], answer=english_answer)
        demonstration += " " # we add a space between demonstrations

    # reformat the dataset such that it is in generation format
    def reformat_row(row, demonstration, template, english_squad, id_to_index):
        english_question = english_squad[id_to_index[row["id"]]]["question"]
        prompt = demonstration + template.format(title=row["title"], context=row["context"], question=english_question)
        english_label = english_squad[id_to_index[row["id"]]]["answers"]["text"][0]
        chinese_label = row["answers"]["text"][0]
        return {"prompt": prompt, "label": english_label}

    inference_dataset = chinese_squad.map(reformat_row, batched=False, fn_kwargs={"demonstration": demonstration, "template": QUESTION_TEMPLATE, "english_squad": english_squad, "id_to_index": id_to_index})

    save_evaluator_config_and_sample(evaluator, out_dir, inference_dataset[0]["prompt"])

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "generation_output.jsonl")
    print("saving to ", out_fn)

    run_inference_new("generate", hf_model, tokenizer, inference_dataset, out_fn, batch_size=evaluator.batch_size,
                      generation_kwargs=evaluator.generation_kwargs)

    # creates a processes that calls google perspective API and saves the output
    # progress_file = os.path.join(out_dir, "perspective_api_progress_includingprompt.json")
    # use_perspective_api(out_fn, PERSPECTIVE_API_KEY, progress_file)


def cad_hiddenstate_evaluator(hf_model, tokenizer, evaluator, out_dir):
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    eval_out_fn = os.path.join(out_dir, "hidden_states_eval.jsonl")
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")

    LABELS = ["IdentityDirectedAbuse", "CounterSpeech", "PersonDirectedAbuse", "Neutral", "AffiliationDirectedAbuse"]

    if not os.path.exists(train_out_fn) or not os.path.exists(eval_out_fn) or not os.path.exists(test_out_fn):

        #load the dataset and select balanced partitions
        train_dataset_fn = os.path.join(evaluator.data.name, "cad_v1_1_train.tsv")
        eval_dataset_fn = os.path.join(evaluator.data.name, "cad_v1_1_dev.tsv")
        test_dataset_fn = os.path.join(evaluator.data.name, "cad_v1_1_test.tsv")
        train_dataset = read_dataset_to_hf(train_dataset_fn)["train"].shuffle(seed=evaluator.seed)
        eval_dataset = read_dataset_to_hf(eval_dataset_fn)["train"].shuffle(seed=evaluator.seed)
        test_dataset = read_dataset_to_hf(test_dataset_fn)["train"].shuffle(seed=evaluator.seed)


        # reformat the dataset such that it is in generation format
        def reformat_row(row, prompt):
            final_instruction = prompt.format(input=row["text"], output="")
            label = [1 if LABELS[i] in row["labels"] else 0 for i in range(len(LABELS))]
            return {"prompt": final_instruction,
                    "label": label}

        if evaluator.use_prompt:
            prompt = TOXIC_CLASSIFICATION_WITH_PROMPT
        else:
            prompt = TOXIC_CLASSIFICATION_NO_PROMPT

        train_dataset = train_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})
        eval_dataset = eval_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})
        test_dataset = test_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

        save_evaluator_config_and_sample(evaluator, out_dir, train_dataset[0]["prompt"])

        # runs the hidden states and saves the output for both train and eval
        run_inference_new("hidden_state", hf_model, tokenizer, train_dataset, train_out_fn, batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, eval_dataset, eval_out_fn, batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, test_dataset, test_out_fn, batch_size=evaluator.batch_size)

    #trains the binary classifier and records the best accuracy
    hidden_size = hf_model.config.hidden_size

    # load the dataset into numpy format
    classifier_train_dataset = pd.read_json(train_out_fn, orient="records", lines=True)
    classifier_eval_dataset = pd.read_json(eval_out_fn, orient="records", lines=True)
    classifier_test_dataset = pd.read_json(test_out_fn, orient="records", lines=True)

    X_train = np.stack(classifier_train_dataset["hidden_state"])
    y_train = np.stack(classifier_train_dataset["label"])

    X_eval = np.stack(classifier_eval_dataset["hidden_state"])
    y_eval = np.stack(classifier_eval_dataset["label"])

    X_test = np.stack(classifier_test_dataset["hidden_state"])
    y_test = np.stack(classifier_test_dataset["label"])

    # shuffle the data
    indices_train = np.random.permutation(len(X_train))
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    indices_eval = np.random.permutation(len(X_eval))
    X_eval = X_eval[indices_eval]
    y_eval = y_eval[indices_eval]

    indices_test = np.random.permutation(len(X_test))
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]

    # Initialize dictionary to store f1-scores for each class
    f1_scores_eval = {}
    f1_scores_test = {}

    # Array to store predictions for all labels
    y_pred_all_eval = np.zeros_like(y_eval)
    y_pred_all_test = np.zeros_like(y_test)

    # Train a classifier for each label (binary relevance)
    for i in range(y_train.shape[1]):
        print(f"Training classifier for label {LABELS[i]}")

        # # Train the logistic regression model
        clf = LogisticRegression(class_weight="balanced", max_iter=5000)
        clf.fit(X_train, y_train[:, i])

        # Predict on the eval set for the current label
        y_pred = clf.predict(X_eval)
        y_pred_all_eval[:, i] = y_pred  # Store predictions for overall evaluation

        # Compute the F1 score for the current label
        f1 = f1_score(y_eval[:, i], y_pred)
        f1_scores_eval[f"Label {i}"] = f1
        print(f"dev F1 Score for label {LABELS[i]}: {f1:.4f}")

        # Predict on the test set for the current label
        y_pred = clf.predict(X_test)
        y_pred_all_test[:, i] = y_pred  # Store predictions for overall evaluation

        # Compute the F1 score for the current label
        f1 = f1_score(y_test[:, i], y_pred)
        f1_scores_test[f"Label {LABELS[i]}"] = f1
        print(f"test F1 Score for label {LABELS[i]}: {f1:.4f}")

    # Compute overall performance metrics
    macro_f1 = f1_score(y_test, y_pred_all_test, average='macro')
    micro_f1 = f1_score(y_test, y_pred_all_test, average='micro')
    accuracy = accuracy_score(y_test, y_pred_all_test)

    macro_f1_eval = f1_score(y_eval, y_pred_all_eval, average='macro')
    micro_f1_eval = f1_score(y_eval, y_pred_all_eval, average='micro')
    accuracy_eval = accuracy_score(y_eval, y_pred_all_eval)

    # Print out performance metrics for each label
    print("\nOverall Performance Metrics for Evaluation Set:")
    print(f"dev Macro F1 Score: {macro_f1_eval:.4f}")
    print(f"dev Micro F1 Score: {micro_f1_eval:.4f}")
    print(f"dev Accuracy: {accuracy_eval:.4f}")


    # Print out overall performance metrics
    print("\nOverall Performance Metrics:")
    print(f"test Macro F1 Score: {macro_f1:.4f}")
    print(f"test Micro F1 Score: {micro_f1:.4f}")
    print(f"test Accuracy: {accuracy:.4f}")

    # Save the performance metrics to a file
    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(f"Overall Performance Metrics for Evaluation Set:\n")
        f.write(f"dev Macro F1 Score: {macro_f1_eval:.4f}\n")
        f.write(f"dev Micro F1 Score: {micro_f1_eval:.4f}\n")
        f.write(f"dev Accuracy: {accuracy_eval:.4f}\n\n")
        f.write(f"Overall Performance Metrics:\n")
        f.write(f"test Macro F1 Score: {macro_f1:.4f}\n")
        f.write(f"test Micro F1 Score: {micro_f1:.4f}\n")
        f.write(f"test Accuracy: {accuracy:.4f}\n\n")
        f.write(f"F1 Scores for each label on the evaluation set:\n")
        f.write(json.dumps(f1_scores_eval, indent=4))
        f.write("\n\n")
        f.write(f"F1 Scores for each label on the test set:\n")
        f.write(json.dumps(f1_scores_test, indent=4))



def slurcorpus_hiddenstate_evaluator(hf_model, tokenizer, evaluator, out_dir):
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")

    LABELS = ['DEG', 'NDG', 'HOM', 'APR', 'CMP']

    if not os.path.exists(train_out_fn)  or not os.path.exists(test_out_fn):

        #load the dataset and select balanced partitions
        dataset = read_dataset_to_hf(evaluator.data.name)["train"].shuffle(seed=evaluator.seed)

        # filter for "none" in labels
        dataset = dataset.filter(lambda x: x["gold_label"] != None)

        # perform train test split
        dataset = dataset.train_test_split(test_size=0.15, shuffle=True, seed=42)

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # reformat the dataset such that it is in generation format
        def reformat_row(row, prompt):
            final_instruction = prompt.format(input=row["body"], output="")
            label = [1 if LABELS[i] in row["gold_label"] else 0 for i in range(len(LABELS))]
            return {"prompt": final_instruction,
                    "label": label}

        if evaluator.use_prompt:
            prompt = TOXIC_CLASSIFICATION_WITH_PROMPT
        else:
            prompt = TOXIC_CLASSIFICATION_NO_PROMPT

        train_dataset =  train_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt}, remove_columns=train_dataset.column_names)
        test_dataset = test_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt}, remove_columns=test_dataset.column_names)

        save_evaluator_config_and_sample(evaluator, out_dir, train_dataset[0]["prompt"])

        # runs the hidden states and saves the output for both train and eval
        run_inference_new("hidden_state", hf_model, tokenizer, train_dataset, train_out_fn, batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, test_dataset, test_out_fn, batch_size=evaluator.batch_size)


    # load the dataset into numpy format
    classifier_train_dataset = pd.read_json(train_out_fn, orient="records", lines=True)
    classifier_test_dataset = pd.read_json(test_out_fn, orient="records", lines=True)

    X_train = np.stack(classifier_train_dataset["hidden_state"])
    y_train = np.stack(classifier_train_dataset["label"])

    X_test = np.stack(classifier_test_dataset["hidden_state"])
    y_test = np.stack(classifier_test_dataset["label"])

    # shuffle the data
    indices_train = np.random.permutation(len(X_train))
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    indices_test = np.random.permutation(len(X_test))
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]

    # Initialize dictionary to store f1-scores for each class
    f1_scores_eval = {}
    f1_scores_test = {}

    # Array to store predictions for all labels
    y_pred_all_test = np.zeros_like(y_test)

    # Train a classifier for each label (binary relevance)
    for i in range(y_train.shape[1]):
        print(f"Training classifier for label {LABELS[i]}")

        # # Train the logistic regression model
        clf = LogisticRegression(class_weight="balanced", max_iter=5000)
        clf.fit(X_train, y_train[:, i])


        # Predict on the test set for the current label
        y_pred = clf.predict(X_test)
        y_pred_all_test[:, i] = y_pred  # Store predictions for overall evaluation

        # Compute the F1 score for the current label
        f1 = f1_score(y_test[:, i], y_pred)
        f1_scores_test[f"Label {LABELS[i]}"] = f1
        print(f"test F1 Score for label {LABELS[i]}: {f1:.4f}")

    # Compute overall performance metrics
    # we first remove the category CMP (dimension n_examples x categories)
    y_test = y_test[:, :-1]
    y_pred_all_test = y_pred_all_test[:, :-1]

    macro_f1 = f1_score(y_test, y_pred_all_test, average='macro')
    micro_f1 = f1_score(y_test, y_pred_all_test, average='micro')
    accuracy = accuracy_score(y_test, y_pred_all_test)


    # Print out overall performance metrics
    print("\nOverall Performance Metrics:")
    print(f"test Macro F1 Score: {macro_f1:.4f}")
    print(f"test Micro F1 Score: {micro_f1:.4f}")
    print(f"test Accuracy: {accuracy:.4f}")

    # Save the performance metrics to a file
    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(f"Overall Performance Metrics for Evaluation Set:\n")
        f.write(f"Overall Performance Metrics:\n")
        f.write(f"test Macro F1 Score: {macro_f1:.4f}\n")
        f.write(f"test Micro F1 Score: {micro_f1:.4f}\n")
        f.write(f"test Accuracy: {accuracy:.4f}\n\n")
        f.write(f"F1 Scores for each label on the evaluation set:\n")
        f.write(json.dumps(f1_scores_eval, indent=4))
        f.write("\n\n")
        f.write(f"F1 Scores for each label on the test set:\n")
        f.write(json.dumps(f1_scores_test, indent=4))


def NEW_generation_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
        generates the output for classification of toxic texts. Evaluate the model performance
        :param hf_model: the loaded model
        :param evaluator: the evaluator
        """

    dataset = read_dataset_to_hf(evaluator.data.name).shuffle(seed=evaluator.seed)
    test_dataset = dataset["test"]
    train_dataset = dataset["train"]
    total_demonstration_dataset = train_dataset

    def reformat_row(row, prompt):
        final_instruction = prompt.format(input=row["text"], output="")
        return {"prompt": final_instruction, "label": row["toxicity"] >= evaluator.data.toxicity_threshold}

    # Initialize lists to store metrics for multiple subsamples
    f1_scores, precision_scores, recall_scores, accuracy_scores, rocauc_scores = [], [], [], [], []

    for i in range(evaluator.data.num_samples):  # Generate different subsamples
        demonstration_dataset = select_binary_balanced_dataset(
            total_demonstration_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
            evaluator.seed + i, evaluator.data.num_demonstrations // 2, num_proc=evaluator.num_proc)

        prompt = TOXIC_CLASSIFICATION_WITH_PROMPT
        demonstration_dataset = demonstration_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

        demonstration_prefix = ""
        for example in demonstration_dataset:
            label = " Yes" if example["toxicity"] >= evaluator.data.toxicity_threshold else " No"
            demonstration = prompt.format(input=example["text"], output=label)
            demonstration_prefix += demonstration + "\n"

        new_prompt = demonstration_prefix + prompt

        test_dataset_balanced = select_binary_balanced_dataset(
            test_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
            evaluator.seed + i, evaluator.data.num_test // 2, num_proc=evaluator.num_proc)

        test_dataset_balanced = test_dataset_balanced.map(reformat_row, batched=False, fn_kwargs={"prompt": new_prompt})
        out_fn_balanced = os.path.join(out_dir, f"generation_output_balanced_{i}.jsonl")

        run_inference_new("logits", hf_model, tokenizer, test_dataset_balanced, out_fn_balanced,
                          batch_size=evaluator.batch_size, target_token_ids=evaluator.data.target_tokens)

        results_balanced = pd.read_json(out_fn_balanced, orient="records", lines=True)
        prediction_balanced = results_balanced["completion"]
        labels_balanced = test_dataset_balanced["label"]

        logits_balanced = results_balanced["logits"]
        probs_balanced = torch.softmax(torch.tensor(logits_balanced), dim=1)[:, 0].numpy()

        f1_scores.append(f1_score(labels_balanced, prediction_balanced))
        precision_scores.append(precision_score(labels_balanced, prediction_balanced))
        recall_scores.append(recall_score(labels_balanced, prediction_balanced))
        accuracy_scores.append(accuracy_score(labels_balanced, prediction_balanced))
        rocauc_scores.append(roc_auc_score(labels_balanced, probs_balanced))

    # Compute mean and standard error for each metric
    metrics = {
        "F1 Score": (np.mean(f1_scores), sem(f1_scores), f1_scores),
        "Precision": (np.mean(precision_scores), sem(precision_scores), precision_scores),
        "Recall": (np.mean(recall_scores), sem(recall_scores), recall_scores),
        "Accuracy": (np.mean(accuracy_scores), sem(accuracy_scores), accuracy_scores),
        "ROC AUC": (np.mean(rocauc_scores), sem(rocauc_scores), rocauc_scores)
    }

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write("Balanced Dataset Metrics (Mean ± StdErr):\n")
        for metric, (mean, stderr, raw) in metrics.items():
            f.write(f"{metric}: {mean:.4f} ± {stderr:.4f} | Raw: {raw}\n")

def NEW_CHAT_generation_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
        generates the output for classification of toxic texts. Evaluate the model performance
        :param hf_model: the loaded model
        :param evaluator: the evaluator
        """

    dataset = read_dataset_to_hf(evaluator.data.name).shuffle(seed=evaluator.seed)
    test_dataset = dataset["test"]
    train_dataset = dataset["train"]
    total_demonstration_dataset = train_dataset

    def reformat_row(row, prompt):
        final_instruction = prompt.format(input=row["text"], output="")
        return {"prompt": final_instruction, "label": row["toxicity"] >= evaluator.data.toxicity_threshold}

    def reformat_row_for_test(row, demonstration_conversations):
        final_instruction = prompt.format(input=row["text"], output="")
        prompt_conversation = demonstration_conversations + [{"role": "user", "content": final_instruction}]
        return {"prompt": prompt_conversation, "label": row["toxicity"] >= evaluator.data.toxicity_threshold}

    # Initialize lists to store metrics for multiple subsamples
    f1_scores, precision_scores, recall_scores, accuracy_scores, rocauc_scores = [], [], [], [], []

    for i in range(evaluator.data.num_samples):  # Generate different subsamples
        demonstration_dataset = select_binary_balanced_dataset(
            total_demonstration_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
            evaluator.seed + i, evaluator.data.num_demonstrations // 2, num_proc=evaluator.num_proc)

        prompt = TOXIC_CLASSIFICATION_WITH_PROMPT
        demonstration_dataset = demonstration_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

        demonstration_conversations = []
        for example in demonstration_dataset:
            label = "Yes" if example["toxicity"] >= evaluator.data.toxicity_threshold else "No"
            demonstration = prompt.format(input=example["text"], output="")
            demonstration_conversations += [{"role": "user", "content": demonstration}, {"role": "assistant", "content": label}]

        test_dataset_balanced = select_binary_balanced_dataset(
            test_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
            evaluator.seed + i, evaluator.data.num_test // 2, num_proc=evaluator.num_proc)

        test_dataset_balanced = test_dataset_balanced.map(reformat_row_for_test, batched=False, fn_kwargs={"demonstration_conversations": demonstration_conversations})
        out_fn_balanced = os.path.join(out_dir, f"generation_output_balanced_{i}.jsonl")

        run_inference_new("logits", hf_model, tokenizer, test_dataset_balanced, out_fn_balanced,
                          batch_size=evaluator.batch_size, target_token_ids=evaluator.data.target_tokens, use_chat_template=True)

        results_balanced = pd.read_json(out_fn_balanced, orient="records", lines=True)
        prediction_balanced = results_balanced["completion"]
        labels_balanced = test_dataset_balanced["label"]

        logits_balanced = results_balanced["logits"]
        probs_balanced = torch.softmax(torch.tensor(logits_balanced), dim=1)[:, 0].numpy()

        f1_scores.append(f1_score(labels_balanced, prediction_balanced))
        precision_scores.append(precision_score(labels_balanced, prediction_balanced))
        recall_scores.append(recall_score(labels_balanced, prediction_balanced))
        accuracy_scores.append(accuracy_score(labels_balanced, prediction_balanced))
        rocauc_scores.append(roc_auc_score(labels_balanced, probs_balanced))

    # Compute mean and standard error for each metric
    metrics = {
        "F1 Score": (np.mean(f1_scores), sem(f1_scores), f1_scores),
        "Precision": (np.mean(precision_scores), sem(precision_scores), precision_scores),
        "Recall": (np.mean(recall_scores), sem(recall_scores), recall_scores),
        "Accuracy": (np.mean(accuracy_scores), sem(accuracy_scores), accuracy_scores),
        "ROC AUC": (np.mean(rocauc_scores), sem(rocauc_scores), rocauc_scores)
    }

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write("Balanced Dataset Metrics (Mean ± StdErr):\n")
        for metric, (mean, stderr, raw) in metrics.items():
            f.write(f"{metric}: {mean:.4f} ± {stderr:.4f} | Raw: {raw}\n")

def generation_direct_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
        generates the output for classification of toxic texts. Evaluate the model performance
        :param hf_model: the loaded model
        :param evaluator: the evaluator
        """

    dataset = read_dataset_to_hf(evaluator.data.name).shuffle(seed=evaluator.seed)

    # we create two partitions of the dataset
    demonstration_dataset = dataset["train"]
    query_dataset = dataset["test"]

    # buffer = 1000 # this is for giving more data to demo dataset to get equal number of toxic and non-toxic examples
    # demonstration_datset = dataset.select(range(len(dataset) - evaluator.data.num_demonstrations - buffer, len(dataset)))
    # query_dataset = dataset.select(range(len(dataset) - evaluator.data.num_demonstrations - buffer))

    # select balanced dataset for the query and demonstration
    test_dataset_balanced = select_binary_balanced_dataset(query_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
                                             evaluator.seed, evaluator.data.num_test // 2, num_proc=evaluator.num_proc)
    test_dataset_orig = query_dataset.select(range(evaluator.data.num_test))
    demonstration_dataset = select_binary_balanced_dataset(demonstration_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
                                             evaluator.seed, evaluator.data.num_demonstrations // 2, num_proc=evaluator.num_proc)

    # reformat the dataset such that it is in generation format
    def reformat_row(row, prompt):
        final_instruction = prompt.format(input=row["text"], output="")
        return {"prompt": final_instruction,
                "label": row["toxicity"] >= evaluator.data.toxicity_threshold}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT

    demonstration_dataset = demonstration_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

    # we create a demonstration string
    demonstration_prefix = ""
    for example in demonstration_dataset:
        label = " no" if example["toxicity"] >= evaluator.data.toxicity_threshold else " no"
        demonstration = prompt.format(input=example["text"], output=label)
        demonstration_prefix += demonstration + "\n"

    # note: we don't provide demonstrations for zero-shot
    # new_prompt = demonstration_prefix + prompt
    new_prompt =  prompt

    test_dataset_balanced = test_dataset_balanced.map(reformat_row, batched=False, fn_kwargs={"prompt": new_prompt})
    test_dataset_unbalanced = test_dataset_orig.map(reformat_row, batched=False, fn_kwargs={"prompt": new_prompt})

    # test_dataset = test_dataset_balanced.select(range(1000))

    save_evaluator_config_and_sample(evaluator, out_dir, test_dataset_balanced[0]["prompt"])

    # runs the generation and saves the output
    out_fn_balanced = os.path.join(out_dir, "generation_output_balanced.jsonl")
    out_fn_unbalanced = os.path.join(out_dir, "generation_output_unbalanced.jsonl")
    print(f"saving to {out_fn_balanced} and {out_fn_unbalanced}")

    run_inference_new("logits", hf_model, tokenizer, test_dataset_balanced, out_fn_balanced, batch_size=evaluator.batch_size,
                      target_token_ids=evaluator.data.target_tokens)
    run_inference_new("logits", hf_model, tokenizer, test_dataset_unbalanced, out_fn_unbalanced, batch_size=evaluator.batch_size,
                      target_token_ids=evaluator.data.target_tokens)

    # loads the file and determine f1 score as well as rocauc score
    results_balanced = pd.read_json(out_fn_balanced, orient="records", lines=True)

    prediction_balanced = results_balanced["completion"]
    labels_balanced = test_dataset_balanced["label"]

    f1_balanced = f1_score(labels_balanced, prediction_balanced)
    precision_balanced = precision_score(labels_balanced, prediction_balanced)
    recall_balanced = recall_score(labels_balanced, prediction_balanced)
    accuracy_balanced = accuracy_score(labels_balanced, prediction_balanced)

    logits_balanced = results_balanced["logits"]
    # we choose the probability of the first class (which is the "yes" class)
    probs_balanced = torch.softmax(torch.tensor(logits_balanced), dim=1)[:, 0].numpy()

    rocauc_balanced = roc_auc_score(labels_balanced, probs_balanced)

    results_unbalanced = pd.read_json(out_fn_unbalanced, orient="records", lines=True)

    prediction_unbalanced = results_unbalanced["completion"]
    labels_unbalanced = test_dataset_unbalanced["label"]

    f1_unbalanced = f1_score(labels_unbalanced, prediction_unbalanced)
    precision_unbalanced = precision_score(labels_unbalanced, prediction_unbalanced)
    recall_unbalanced = recall_score(labels_unbalanced, prediction_unbalanced)
    accuracy_unbalanced = accuracy_score(labels_unbalanced, prediction_unbalanced)

    logits_unbalanced = results_unbalanced["logits"]
    # we choose the probability of the first class (which is the "yes" class)
    probs_unbalanced = torch.softmax(torch.tensor(logits_unbalanced), dim=1)[:, 0].numpy()

    rocauc_unbalanced = roc_auc_score(labels_unbalanced, probs_unbalanced)

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(f"Balanced Dataset Metrics:\n")
        f.write(f"F1 Score: {f1_balanced:.4f}\n")
        f.write(f"Precision: {precision_balanced:.4f}\n")
        f.write(f"Recall: {recall_balanced:.4f}\n")
        f.write(f"Accuracy: {accuracy_balanced:.4f}\n")
        f.write(f"ROC AUC: {rocauc_balanced:.4f}\n\n")

        f.write(f"Unbalanced Dataset Metrics:\n")
        f.write(f"F1 Score: {f1_unbalanced:.4f}\n")
        f.write(f"Precision: {precision_unbalanced:.4f}\n")
        f.write(f"Recall: {recall_unbalanced:.4f}\n")
        f.write(f"Accuracy: {accuracy_unbalanced:.4f}\n")
        f.write(f"ROC AUC: {rocauc_unbalanced:.4f}\n\n")

def NEW_generation_dynahate_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
        generates the output for classification of toxic texts. Evaluate the model performance
        :param hf_model: the loaded model
        :param evaluator: the evaluator
        """

    query_dataset = read_dataset_to_hf(evaluator.data.path)["train"].shuffle(seed=evaluator.seed)
    buffer = 1000  # buffer for extracting the few-shot examples from the end
    total_demonstration_dataset = query_dataset.select(
        range(len(query_dataset) - buffer, len(query_dataset)))
    query_dataset = query_dataset.select(range(len(query_dataset) - buffer))

    # Initialize lists to store metrics for multiple subsamples
    f1_scores, precision_scores, recall_scores, accuracy_scores, rocauc_scores = [], [], [], [], []

    for i in range(5):  # Generate 5 different subsamples
        test_dataset_balanced = select_binary_balanced_dataset(
            query_dataset, lambda x: x["label"] == "hate",
            evaluator.seed + i, evaluator.data.num_test // 2, num_proc=evaluator.num_proc)
        demonstration_dataset = select_binary_balanced_dataset(
            total_demonstration_dataset, lambda x: x["label"] == "hate",
            evaluator.seed + i, evaluator.data.num_demonstrations // 2, num_proc=evaluator.num_proc)

        def reformat_row(row, prompt):
            final_instruction = prompt.format(input=row["text"].strip(), output="")
            return {"prompt": final_instruction, "label": row["label"] == "hate"}

        prompt = TOXIC_CLASSIFICATION_WITH_PROMPT
        demonstration_prefix = "".join(
            prompt.format(input=example["text"].strip(), output=" Yes" if example["label"] == "hate" else " No") + "\n"
            for example in demonstration_dataset)

        new_prompt = demonstration_prefix + prompt
        test_dataset_balanced = test_dataset_balanced.map(reformat_row, batched=False, fn_kwargs={"prompt": new_prompt})

        out_fn_balanced = os.path.join(out_dir, f"generation_output_balanced_{i}.jsonl")
        run_inference_new("logits", hf_model, tokenizer, test_dataset_balanced, out_fn_balanced,
                          batch_size=evaluator.batch_size, target_token_ids=evaluator.data.target_tokens)

        results_balanced = pd.read_json(out_fn_balanced, orient="records", lines=True)
        prediction_balanced = results_balanced["completion"]
        labels_balanced = test_dataset_balanced["label"]
        logits_balanced = results_balanced["logits"]
        probs_balanced = torch.softmax(torch.tensor(logits_balanced), dim=1)[:, 0].numpy()

        f1_scores.append(f1_score(labels_balanced, prediction_balanced))
        precision_scores.append(precision_score(labels_balanced, prediction_balanced))
        recall_scores.append(recall_score(labels_balanced, prediction_balanced))
        accuracy_scores.append(accuracy_score(labels_balanced, prediction_balanced))
        rocauc_scores.append(roc_auc_score(labels_balanced, probs_balanced))

    # Compute mean and standard error for each metric
    metrics = {
        "F1 Score": (np.mean(f1_scores), sem(f1_scores), f1_scores),
        "Precision": (np.mean(precision_scores), sem(precision_scores), precision_scores),
        "Recall": (np.mean(recall_scores), sem(recall_scores), recall_scores),
        "Accuracy": (np.mean(accuracy_scores), sem(accuracy_scores), accuracy_scores),
        "ROC AUC": (np.mean(rocauc_scores), sem(rocauc_scores), rocauc_scores)
    }

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write("Balanced Dataset Metrics (Mean ± StdErr):\n")
        for metric, (mean, stderr, raw) in metrics.items():
            f.write(f"{metric}: {mean:.4f} ± {stderr:.4f} | Raw: {raw}\n")


def NEW_hidden_state_dynahate_evaluator(hf_model, tokenizer, evaluator, out_dir):
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")

    dataset = read_dataset_to_hf(evaluator.data.path)["train"].shuffle(seed=evaluator.seed)

    test_dataset = dataset.filter(lambda x: x["split"] == "test")
    train_dataset = dataset.filter(lambda x: x["split"] == "train")

    accuracy_scores, f1_scores, precision_scores, recall_scores, roc_auc_scores, pr_auc_scores = [], [], [], [], [], []

    for i in range(evaluator.data.num_samples):  # Generate 5 different subsamples
        train_subsample = select_binary_balanced_dataset(
            train_dataset, lambda x: x["label"] == "hate",
            evaluator.seed + i, evaluator.data.num_train // 2)

        test_subsample = select_binary_balanced_dataset(
            test_dataset, lambda x: x["label"] == "hate",
            evaluator.seed + i, evaluator.data.num_test // 2)

        def reformat_row(row, prompt):
            final_instruction = prompt.format(input=row["text"], output="")
            return {"prompt": final_instruction, "label": row["label"] == "hate"}

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

        accuracy_scores.append(accuracy_score(y_test, y_pred_test))
        f1_scores.append(f1_score(y_test, y_pred_test))
        precision_scores.append(precision_score(y_test, y_pred_test))
        recall_scores.append(recall_score(y_test, y_pred_test))
        roc_auc_scores.append(roc_auc_score(y_test, y_prob_test))
        pr_auc_scores.append(average_precision_score(y_test, y_prob_test))

    metrics = {
        "Test Accuracy": (np.mean(accuracy_scores), sem(accuracy_scores), accuracy_scores),
        "Test F1 Score": (np.mean(f1_scores), sem(f1_scores), f1_scores),
        "Test Precision": (np.mean(precision_scores), sem(precision_scores), precision_scores),
        "Test Recall": (np.mean(recall_scores), sem(recall_scores), recall_scores),
        "Test ROC AUC": (np.mean(roc_auc_scores), sem(roc_auc_scores), roc_auc_scores),
        "Test PR AUC": (np.mean(pr_auc_scores), sem(pr_auc_scores), pr_auc_scores),
    }

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write("Test Metrics (Mean ± StdErr):\n")
        for metric, (mean, stderr, raw) in metrics.items():
            f.write(f"{metric}: {mean:.4f} ± {stderr:.4f} | {raw}\n")

def NEW_EASY_generation_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
        generates the output for classification of toxic texts. Evaluate the model performance
        :param hf_model: the loaded model
        :param evaluator: the evaluator
        """
    demonstration_dataset = read_dataset_to_hf(evaluator.data.name, name="annotated")["train"].shuffle(seed=evaluator.seed)
    test_dataset = read_dataset_to_hf(evaluator.data.name, name="annotated")["test"].shuffle(seed=evaluator.seed)

    # we first filter the datasetes to only include extreme examples (which makes the task easier
    demonstration_dataset = demonstration_dataset.filter(lambda x: x["toxicity_human"] >= 4.5 or x["toxicity_human"] <= 1.5)
    test_dataset = test_dataset.filter(lambda x: x["toxicity_human"] >= 4.5 or x["toxicity_human"] <= 1.5)

    # select balanced dataset for the demonstration
    demonstration_dataset = select_binary_balanced_dataset(demonstration_dataset, lambda x: x["toxicity_human"] >= 3,
                                                           evaluator.seed, evaluator.data.num_demonstrations // 2,
                                                           num_proc=evaluator.num_proc)

    test_dataset = select_binary_balanced_dataset(test_dataset, lambda x: x["toxicity_human"] >= 3,
                                                  evaluator.seed,
                                                  374 // 2)  # 374 is max number supported for balanced dataset

    # reformat the dataset such that it is in generation format
    def reformat_row(row, prompt):
        final_instruction = prompt.format(input=row["text"].strip(), output="")
        return {"prompt": final_instruction,
                "label": row["toxicity_human"] >= 3}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT

    # we create a demonstration string
    demonstration_prefix = ""
    for example in demonstration_dataset:
        label = " Yes" if example["toxicity_human"] >= 3 else " No"
        demonstration = prompt.format(input=example["text"].strip(), output=label)
        demonstration_prefix += demonstration + "\n"

    new_prompt = demonstration_prefix + prompt

    test_dataset = test_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": new_prompt})

    save_evaluator_config_and_sample(evaluator, out_dir, test_dataset[0]["prompt"])

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "generation_output.jsonl")
    print(f"saving to {out_fn}")

    run_inference_new("logits", hf_model, tokenizer, test_dataset, out_fn, batch_size=evaluator.batch_size,
                      target_token_ids=evaluator.data.target_tokens)

    # loads the file and determine f1 score as well as rocauc score
    results = pd.read_json(out_fn, orient="records", lines=True)

    prediction = results["completion"]
    labels = test_dataset["label"]

    f1 = f1_score(labels, prediction)
    precision = precision_score(labels, prediction)
    recall = recall_score(labels, prediction)
    accuracy = accuracy_score(labels, prediction)

    logits = results["logits"]
    # we choose the probability of the first class (which is the "yes" class)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 0].numpy()

    rocauc = roc_auc_score(labels, probs)

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(f"Dataset Metrics:\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {rocauc:.4f}\n\n")


def NEW_generation_balanced_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
        generates the output for classification of toxic texts. Evaluate the model performance
        :param hf_model: the loaded model
        :param evaluator: the evaluator
        """

    demonstration_dataset = read_dataset_to_hf(evaluator.data.name, name="annotated")["train"].shuffle(
        seed=evaluator.seed)
    test_dataset = read_dataset_to_hf(evaluator.data.name, name="annotated")["test"].shuffle(seed=evaluator.seed)

    # select balanced dataset for the demonstration
    demonstration_dataset = select_binary_balanced_dataset(demonstration_dataset, lambda x: x["toxicity_human"] >= 3,
                                                           evaluator.seed, evaluator.data.num_demonstrations // 2,
                                                           num_proc=evaluator.num_proc)

    test_dataset = select_binary_balanced_dataset(test_dataset, lambda x: x["toxicity_human"] >= 3,
                                                  evaluator.seed,
                                                  804 // 2)  # 804 is max number supported for balanced dataset

    # reformat the dataset such that it is in generation format
    def reformat_row(row, prompt):
        final_instruction = prompt.format(input=row["text"].strip(), output="")
        return {"prompt": final_instruction,
                "label": row["toxicity_human"] >= 3}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT

    # we create a demonstration string
    demonstration_prefix = ""
    for example in demonstration_dataset:
        label = " Yes" if example["toxicity_human"] >= 3 else " No"
        demonstration = prompt.format(input=example["text"].strip(), output=label)
        demonstration_prefix += demonstration + "\n"

    new_prompt = demonstration_prefix + prompt

    test_dataset = test_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": new_prompt})

    save_evaluator_config_and_sample(evaluator, out_dir, test_dataset[0]["prompt"])

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "generation_output.jsonl")
    print(f"saving to {out_fn}")

    run_inference_new("logits", hf_model, tokenizer, test_dataset, out_fn, batch_size=evaluator.batch_size,
                      target_token_ids=evaluator.data.target_tokens)

    # loads the file and determine f1 score as well as rocauc score
    results = pd.read_json(out_fn, orient="records", lines=True)

    prediction = results["completion"]
    labels = test_dataset["label"]

    f1 = f1_score(labels, prediction)
    precision = precision_score(labels, prediction)
    recall = recall_score(labels, prediction)
    accuracy = accuracy_score(labels, prediction)

    logits = results["logits"]
    # we choose the probability of the first class (which is the "yes" class)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 0].numpy()

    rocauc = roc_auc_score(labels, probs)

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(f"Dataset Metrics:\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {rocauc:.4f}\n\n")

def NEW_generation_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
        generates the output for classification of toxic texts. Evaluate the model performance
        :param hf_model: the loaded model
        :param evaluator: the evaluator
        """
    test_dataset = read_dataset_to_hf(evaluator.data.name, name="annotated")["test"].shuffle(seed=evaluator.seed)

    demonstration_dataset = read_dataset_to_hf(evaluator.data.name, name="annotated")["train"].shuffle(seed=evaluator.seed)

    # select balanced dataset for the demonstration
    demonstration_dataset = select_binary_balanced_dataset(demonstration_dataset, lambda x: x["toxicity_human"] >= 3,
                                             evaluator.seed, evaluator.data.num_demonstrations // 2, num_proc=evaluator.num_proc)

    # reformat the dataset such that it is in generation format
    def reformat_row(row, prompt):
        final_instruction = prompt.format(input=row["text"].strip(), output="")
        return {"prompt": final_instruction,
                "label": row["toxicity_human"] >= 3}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT

    # we create a demonstration string
    demonstration_prefix = ""
    for example in demonstration_dataset:
        label = " Yes" if example["toxicity_human"] >= 3 else " No"
        demonstration = prompt.format(input=example["text"].strip(), output=label)
        demonstration_prefix += demonstration + "\n"

    new_prompt = demonstration_prefix + prompt

    test_dataset = test_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": new_prompt})

    save_evaluator_config_and_sample(evaluator, out_dir, test_dataset[0]["prompt"])

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "generation_output.jsonl")
    print(f"saving to {out_fn}")

    run_inference_new("logits", hf_model, tokenizer, test_dataset, out_fn, batch_size=evaluator.batch_size,
                      target_token_ids=evaluator.data.target_tokens)

    # loads the file and determine f1 score as well as rocauc score
    results = pd.read_json(out_fn, orient="records", lines=True)

    prediction = results["completion"]
    labels = test_dataset["label"]

    f1 = f1_score(labels, prediction)
    precision = precision_score(labels, prediction)
    recall = recall_score(labels, prediction)
    accuracy = accuracy_score(labels, prediction)

    logits = results["logits"]
    # we choose the probability of the first class (which is the "yes" class)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 0].numpy()

    rocauc = roc_auc_score(labels, probs)

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(f"Dataset Metrics:\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {rocauc:.4f}\n\n")


def generation_direct_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
        generates the output for classification of toxic texts. Evaluate the model performance
        :param hf_model: the loaded model
        :param evaluator: the evaluator
        """

    # dataset = read_dataset_to_hf(evaluator.data.name).shuffle(seed=evaluator.seed)

    # we create two partitions of the dataset
    # query_dataset = dataset["test"]
    query_dataset = read_dataset_to_hf(evaluator.data.name, name="train")["train"].shuffle(seed=evaluator.seed)

    # buffer = 1000 # this is for giving more data to demo dataset to get equal number of toxic and non-toxic examples
    # demonstration_datset = dataset.select(range(len(dataset) - evaluator.data.num_demonstrations - buffer, len(dataset)))
    # query_dataset = dataset.select(range(len(dataset) - evaluator.data.num_demonstrations - buffer))

    # select balanced dataset for the query and demonstration
    test_dataset_balanced = select_binary_balanced_dataset(query_dataset, lambda x: x["prompt_label"] == 1,
                                             evaluator.seed, evaluator.data.num_test // 2, num_proc=evaluator.num_proc)
    test_dataset_orig = query_dataset.select(range(evaluator.data.num_test))
    # demonstration_dataset = select_binary_balanced_dataset(demonstration_dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold,
    #                                          evaluator.seed, evaluator.data.num_demonstrations // 2, num_proc=evaluator.num_proc)

    # reformat the dataset such that it is in generation format
    def reformat_row(row, prompt):
        final_instruction = prompt.format(input=row["generation"], output="")
        return {"prompt": final_instruction,
                "label": row["prompt_label"] == 1}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT

    # demonstration_dataset = demonstration_dataset.map(reformat_row, batched=False, fn_kwargs={"prompt": prompt})

    # we create a demonstration string
    # demonstration_prefix = ""
    # for example in demonstration_dataset:
    #     label = " yes" if example["toxicity"] >= evaluator.data.toxicity_threshold else " no"
    #     demonstration = prompt.format(input=example["text"], output=label)
    #     demonstration_prefix += demonstration + "\n"

    # note: we don't provide demonstrations for zero-shot
    # new_prompt = demonstration_prefix + prompt
    new_prompt =  prompt

    test_dataset_balanced = test_dataset_balanced.map(reformat_row, batched=False, fn_kwargs={"prompt": new_prompt})
    test_dataset_unbalanced = test_dataset_orig.map(reformat_row, batched=False, fn_kwargs={"prompt": new_prompt})

    # test_dataset = test_dataset_balanced.select(range(1000))

    save_evaluator_config_and_sample(evaluator, out_dir, test_dataset_balanced[0]["prompt"])

    # runs the generation and saves the output
    out_fn_balanced = os.path.join(out_dir, "generation_output_balanced.jsonl")
    out_fn_unbalanced = os.path.join(out_dir, "generation_output_unbalanced.jsonl")
    print(f"saving to {out_fn_balanced} and {out_fn_unbalanced}")

    run_inference_new("logits", hf_model, tokenizer, test_dataset_balanced, out_fn_balanced, batch_size=evaluator.batch_size,
                      target_token_ids=evaluator.data.target_tokens)
    run_inference_new("logits", hf_model, tokenizer, test_dataset_unbalanced, out_fn_unbalanced, batch_size=evaluator.batch_size,
                      target_token_ids=evaluator.data.target_tokens)

    # loads the file and determine f1 score as well as rocauc score
    results_balanced = pd.read_json(out_fn_balanced, orient="records", lines=True)

    prediction_balanced = results_balanced["completion"]
    labels_balanced = test_dataset_balanced["label"]

    f1_balanced = f1_score(labels_balanced, prediction_balanced)
    precision_balanced = precision_score(labels_balanced, prediction_balanced)
    recall_balanced = recall_score(labels_balanced, prediction_balanced)
    accuracy_balanced = accuracy_score(labels_balanced, prediction_balanced)

    logits_balanced = results_balanced["logits"]
    # we choose the probability of the first class (which is the "yes" class)
    probs_balanced = torch.softmax(torch.tensor(logits_balanced), dim=1)[:, 0].numpy()

    rocauc_balanced = roc_auc_score(labels_balanced, probs_balanced)

    results_unbalanced = pd.read_json(out_fn_unbalanced, orient="records", lines=True)

    prediction_unbalanced = results_unbalanced["completion"]
    labels_unbalanced = test_dataset_unbalanced["label"]

    f1_unbalanced = f1_score(labels_unbalanced, prediction_unbalanced)
    precision_unbalanced = precision_score(labels_unbalanced, prediction_unbalanced)
    recall_unbalanced = recall_score(labels_unbalanced, prediction_unbalanced)
    accuracy_unbalanced = accuracy_score(labels_unbalanced, prediction_unbalanced)

    logits_unbalanced = results_unbalanced["logits"]
    # we choose the probability of the first class (which is the "yes" class)
    probs_unbalanced = torch.softmax(torch.tensor(logits_unbalanced), dim=1)[:, 0].numpy()

    rocauc_unbalanced = roc_auc_score(labels_unbalanced, probs_unbalanced)

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(f"Balanced Dataset Metrics:\n")
        f.write(f"F1 Score: {f1_balanced:.4f}\n")
        f.write(f"Precision: {precision_balanced:.4f}\n")
        f.write(f"Recall: {recall_balanced:.4f}\n")
        f.write(f"Accuracy: {accuracy_balanced:.4f}\n")
        f.write(f"ROC AUC: {rocauc_balanced:.4f}\n\n")

        f.write(f"Unbalanced Dataset Metrics:\n")
        f.write(f"F1 Score: {f1_unbalanced:.4f}\n")
        f.write(f"Precision: {precision_unbalanced:.4f}\n")
        f.write(f"Recall: {recall_unbalanced:.4f}\n")
        f.write(f"Accuracy: {accuracy_unbalanced:.4f}\n")
        f.write(f"ROC AUC: {rocauc_unbalanced:.4f}\n\n")


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
    elif "NEW_dynahate_generation" in evaluator.label:
        NEW_generation_dynahate_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_dynahate_hiddenstate" in evaluator.label:
        NEW_hidden_state_dynahate_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_civilcomments_hiddenstate" in evaluator.label:
        NEW_hidden_state_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_civilcomments_noisychannel" in evaluator.label:
        NEW_noisychannel_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_civilcomments_generation" in evaluator.label:
        NEW_generation_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_CHAT_civilcomments_hiddenstate" in evaluator.label:
        NEW_CHAT_hidden_state_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_CHAT_civilcomments_generation" in evaluator.label:
        NEW_CHAT_generation_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "civilcomments_generation_direct" in evaluator.label:
        generation_direct_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_toxigen_hiddenstate" in evaluator.label:
        NEW_hidden_state_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_EASY_toxigen_hiddenstate" in evaluator.label:
        NEW_EASY_hidden_state_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_toxigen_balanced_generation" in evaluator.label:
        NEW_generation_balanced_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "NEW_EASY_toxigen_generation" in evaluator.label:
        NEW_EASY_generation_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "tofu_custom" in evaluator.label:
        tofu_custom_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "in_distribution_perplexity_dolma" in evaluator.label:
        in_distribution_perplexity_evaluator_dolma(hf_model, tokenizer, evaluator, out_dir)
    elif "in_distribution_perplexity_nontoxicreddit" in evaluator.label:
        in_distribution_perplexity_evaluator_nontoxicreddit(hf_model, tokenizer, evaluator, out_dir)
    elif "in_distribution_perplexity_nontoxicdocumentreddit" in evaluator.label:
        in_distribution_perplexity_evaluator_nontoxicdocumentreddit(hf_model, tokenizer, evaluator, out_dir)
    # elif "in_distribution_perplexity" in evaluator.label:
    #     in_distribution_perplexity_evaluator(hf_model, tokenizer, evaluator, out_dir)
    # elif "CHAT_realtoxicityprompts_generation" in evaluator.label:
    #     real_toxicity_prompt_chat_generation_evaluator(hf_model, tokenizer, evaluator, out_dir)
    # elif "NEW_toxigen_generation" in evaluator.label:
    #     NEW_generation_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir)
    # elif "xnli_hiddenstate" in evaluator.label:
    #     hidden_state_xnli_evaluator(hf_model, tokenizer, evaluator, out_dir)
    # elif "squad_generation" in evaluator.label:
    #     squad_generation_evaluator(hf_model, tokenizer, evaluator, out_dir)
    # elif "cad_hiddenstate" in evaluator.label:
    #     cad_hiddenstate_evaluator(hf_model, tokenizer, evaluator, out_dir)
    # elif "slurcorpus_hiddenstate" in evaluator.label:
    #     slurcorpus_hiddenstate_evaluator(hf_model, tokenizer, evaluator, out_dir)
    # elif evaluator.label == "civilcomments_finegrained_hiddenstate" or evaluator.label == "civilcomments_finegrained_hiddenstate_classification":
    #     hidden_state_civilcomments_finegrained_evaluator(hf_model, tokenizer, evaluator, out_dir)
    # elif "civilcomments_generation_subtle" in evaluator.label:
    #     generation_subtle_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)
    # elif "civilcomments_hiddenstate_insult" in evaluator.label:
    #     hidden_state_civilcomments_insult_evaluator(hf_model, tokenizer, evaluator, out_dir)
    # elif "toxigen_generation" in evaluator.label:
    #     generation_direct_toxigen_evaluator(hf_model, tokenizer, evaluator, out_dir)


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

