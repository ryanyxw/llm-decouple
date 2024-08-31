import multiprocessing
import os

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from API_KEYS import PERSPECTIVE_API_KEY
from src.modules.data.datasets.PandasDataset import PandasDataset
from src.modules.data.format_utils import select_binary_balanced_dataset
from src.modules.data.load import read_dataset_to_hf
from src.modules.modeling.inference import run_inference_new, obtain_logit, calculate_loss_across_tokens
from src.modules.modeling.models.LogisticRegression import BinaryClassifier
from src.modules.templates import CIVIL_COMMENTS_TEMPLATE_NO_LABELS, CIVIL_COMMENTS_FINEGRAINED_TEMPLATE_NO_LABELS, \
    CIVIL_COMMENTS_FINEGRAINED_LABELS
from src.modules.utils import use_perspective_api, seed_all
from src.training.run_train_torch import train_classifier, train_binaryclassifier_multi


def save_evaluator_config_and_sample(evaluator, out_dir, sample):
    # saves a sample of the prompt to a parallel file along with configs
    print("sample of example fed into model: \n" + sample)
    with open(os.path.join(out_dir, "template.jsonl"), "w") as f:
        f.write(sample + "\n")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        OmegaConf.save(evaluator, f, resolve=True)

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


def hidden_state_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir):
    train_out_fn = os.path.join(out_dir, "hidden_states_train.jsonl")
    eval_out_fn = os.path.join(out_dir, "hidden_states_eval.jsonl")
    test_out_fn = os.path.join(out_dir, "hidden_states_test.jsonl")

    if not os.path.exists(train_out_fn) or not os.path.exists(eval_out_fn) or not os.path.exists(test_out_fn):
        #load the dataset and select balanced partitions
        dataset = read_dataset_to_hf(evaluator.data.name)["train"].shuffle(seed=evaluator.seed)
        tot_examples = evaluator.data.num_train + evaluator.data.num_eval + evaluator.data.num_test

        assert (len(dataset) >= tot_examples)

        dataset = select_binary_balanced_dataset(dataset, lambda x: x["toxicity"] >= evaluator.data.toxicity_threshold, evaluator.seed, tot_examples // 2)

        train_dataset = dataset.select(range(evaluator.data.num_train))
        eval_dataset = dataset.select(range(evaluator.data.num_train, evaluator.data.num_train + evaluator.data.num_eval))
        test_dataset = dataset.select(range(evaluator.data.num_train + evaluator.data.num_eval, \
                                            evaluator.data.num_train + evaluator.data.num_eval + evaluator.data.num_test))

        # reformat the dataset such that it is in generation format
        def reformat_row(row):
            final_instruction = CIVIL_COMMENTS_TEMPLATE_NO_LABELS.format(input=row["text"], output="")
            return {"prompt": final_instruction,
                    "label": row["toxicity"] >= evaluator.data.toxicity_threshold}
        train_dataset = train_dataset.map(reformat_row, batched=False)
        eval_dataset = eval_dataset.map(reformat_row, batched=False)
        test_dataset = test_dataset.map(reformat_row, batched=False)

        save_evaluator_config_and_sample(evaluator, out_dir, train_dataset[0]["prompt"])

        # runs the hidden states and saves the output for both train and eval
        run_inference_new("hidden_state", hf_model, tokenizer, train_dataset, train_out_fn, batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, eval_dataset, eval_out_fn, batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, test_dataset, test_out_fn, batch_size=evaluator.batch_size)


    #trains the binary classifier and records the best accuracy
    hidden_size = hf_model.config.hidden_size


    classifier_train_dataset = PandasDataset(pd.read_json(train_out_fn, orient="records", lines=True))
    classifier_eval_dataset = PandasDataset(pd.read_json(eval_out_fn, orient="records", lines=True))
    classifier_test_dataset = PandasDataset(pd.read_json(test_out_fn, orient="records", lines=True))

    if evaluator.use_acc:
        acc_out_dir = os.path.join(out_dir, "acc")
        os.makedirs(acc_out_dir, exist_ok=True)
        classifier_model = BinaryClassifier(input_dim=hidden_size).to("cuda")

        def accuracy_metric(logits, y):
            preds = torch.argmax(logits, dim=1)
            return torch.sum(preds == y).item() / len(y)

        best_dev_acc = train_classifier(classifier_model, classifier_train_dataset, classifier_eval_dataset, evaluator.binary_classifier.epochs,\
                                        evaluator.binary_classifier.batch_size, acc_out_dir, metric_func=accuracy_metric)

        # load the best model
        best_model = BinaryClassifier(hidden_size)
        best_model.load_state_dict(torch.load(os.path.join(acc_out_dir, "tmp_best_model.pth")))
        best_model.to("cuda")

        # test the best model
        best_model.eval()
        with torch.no_grad():
            # num_correct = 0
            total_logits = []
            total_labels = []
            for X_test, y_test in DataLoader(classifier_test_dataset, batch_size=evaluator.binary_classifier.batch_size):
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

    if evaluator.use_rocauc:
        rocauc_out_dir = os.path.join(out_dir, "rocauc")
        os.makedirs(rocauc_out_dir, exist_ok=True)
        classifier_model = BinaryClassifier(input_dim=hidden_size).to("cuda")

        def rocauc_metric(logits, y):
            from sklearn.metrics import roc_auc_score
            preds = torch.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()
            return roc_auc_score(y.cpu().detach().numpy(), preds)

        best_dev_rocauc = train_classifier(classifier_model, classifier_train_dataset, classifier_eval_dataset,
                                        evaluator.binary_classifier.epochs,
                                        evaluator.binary_classifier.batch_size, rocauc_out_dir, metric_func=rocauc_metric)

        # load the best model
        best_model = BinaryClassifier(hidden_size)
        best_model.load_state_dict(torch.load(os.path.join(rocauc_out_dir, "tmp_best_model.pth")))
        best_model.to("cuda")

        # test the best model
        best_model.eval()
        with torch.no_grad():
            total_logits = []
            total_labels = []
            for X_test, y_test in DataLoader(classifier_test_dataset,
                                             batch_size=evaluator.binary_classifier.batch_size):
                test_logits = best_model(X_test.to("cuda")).to("cpu")
                total_logits.append(test_logits)
                total_labels.append(y_test.to("cpu"))

            test_rocauc = rocauc_metric(torch.cat(total_logits), torch.cat(total_labels))
            print(f'Test accuracy: {test_rocauc:.5f}')

        with open(os.path.join(rocauc_out_dir, "rocauc_stats.txt"), "w") as f:
            f.write(f"Best dev rocauc: {best_dev_rocauc}, test rocauc: {test_rocauc}")

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
        def reformat_row(row):
            final_instruction = CIVIL_COMMENTS_TEMPLATE_NO_LABELS.format(input=row["text"], output="")
            return {"prompt": final_instruction,
                    "label": labeler(row)}

        train_dataset = train_dataset.map(reformat_row, batched=False)
        eval_dataset = eval_dataset.map(reformat_row, batched=False)
        test_dataset = test_dataset.map(reformat_row, batched=False)

        save_evaluator_config_and_sample(evaluator, out_dir, train_dataset[0]["prompt"])

        # runs the hidden states and saves the output for both train and eval
        run_inference_new("hidden_state", hf_model, tokenizer, train_dataset, train_out_fn, batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, eval_dataset, eval_out_fn, batch_size=evaluator.batch_size)
        run_inference_new("hidden_state", hf_model, tokenizer, test_dataset, test_out_fn, batch_size=evaluator.batch_size)


    #trains the binary classifier and records the best accuracy
    hidden_size = hf_model.config.hidden_size


    classifier_train_dataset = PandasDataset(pd.read_json(train_out_fn, orient="records", lines=True))
    classifier_eval_dataset = PandasDataset(pd.read_json(eval_out_fn, orient="records", lines=True))
    classifier_test_dataset = PandasDataset(pd.read_json(test_out_fn, orient="records", lines=True))

    if evaluator.use_acc:
        acc_out_dir = os.path.join(out_dir, "acc")
        os.makedirs(acc_out_dir, exist_ok=True)
        classifier_model = BinaryClassifier(input_dim=hidden_size).to("cuda")

        def accuracy_metric(logits, y):
            preds = torch.argmax(logits, dim=1)
            return torch.sum(preds == y).item() / len(y)

        best_dev_acc = train_classifier(classifier_model, classifier_train_dataset, classifier_eval_dataset, evaluator.binary_classifier.epochs,\
                                    evaluator.binary_classifier.batch_size, acc_out_dir, metric_func=accuracy_metric)

        # load the best model
        best_model = BinaryClassifier(hidden_size)
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

    if evaluator.use_rocauc:
        rocauc_out_dir = os.path.join(out_dir, "rocauc")
        os.makedirs(rocauc_out_dir, exist_ok=True)
        classifier_model = BinaryClassifier(input_dim=hidden_size).to("cuda")

        def rocauc_metric(logits, y):
            from sklearn.metrics import roc_auc_score
            preds = torch.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()
            return roc_auc_score(y.cpu().detach().numpy(), preds)

        best_dev_rocauc = train_classifier(classifier_model, classifier_train_dataset, classifier_eval_dataset,
                                           evaluator.binary_classifier.epochs,
                                           evaluator.binary_classifier.batch_size, rocauc_out_dir,
                                           metric_func=rocauc_metric)

        # load the best model
        best_model = BinaryClassifier(hidden_size)
        best_model.load_state_dict(torch.load(os.path.join(rocauc_out_dir, "tmp_best_model.pth")))
        best_model.to("cuda")

        # test the best model
        best_model.eval()
        with torch.no_grad():
            total_logits = []
            total_labels = []
            for X_test, y_test in DataLoader(classifier_test_dataset,
                                             batch_size=evaluator.binary_classifier.batch_size):
                test_logits = best_model(X_test.to("cuda")).to("cpu")
                total_logits.append(test_logits)
                total_labels.append(y_test.to("cpu"))

            test_rocauc = rocauc_metric(torch.cat(total_logits), torch.cat(total_labels))
            print(f'Test accuracy: {test_rocauc:.5f}')

        with open(os.path.join(rocauc_out_dir, "rocauc_stats.txt"), "w") as f:
            f.write(f"Best dev rocauc: {best_dev_rocauc}, test rocauc: {test_rocauc}")

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

def in_distribution_perplexity_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
    evaluates the model on in-distribution toxic data. Four different types of perplexity analysis
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """

    # load the dataset and select the necessary ones
    dataset_names = ["nontoxic_only", "nontoxic_toxic", "toxic_nontoxic", "toxic_only"]
    dataset_arr = [read_dataset_to_hf(os.path.join(evaluator.data.name, name, "data.jsonl"))["train"].select(range(evaluator.data.num_examples)) for name in dataset_names]

    # we assume we use all examples in the test data

    # loop over each type of evaluation data
    for dataset_ind in range(len(dataset_arr)):
        dataset = dataset_arr[dataset_ind]
        dataset_name = dataset_names[dataset_ind]

        current_out_dir = os.path.join(out_dir, dataset_name)
        os.makedirs(current_out_dir, exist_ok=True)

        save_evaluator_config_and_sample(evaluator, current_out_dir, tokenizer.decode(dataset[0]["input_ids"]))

        # runs the generation and saves the output
        out_fn = os.path.join(current_out_dir, "stats.txt")
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
    elif "civilcomments_hiddenstate_noprompt" in evaluator.label:
        hidden_state_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif evaluator.label == "civilcomments_finegrained_hiddenstate" or evaluator.label == "civilcomments_finegrained_hiddenstate_classification":
        hidden_state_civilcomments_finegrained_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "civilcomments_hiddenstate_insult" in evaluator.label:
        hidden_state_civilcomments_insult_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif "in_distribution_perplexity" in evaluator.label:
        in_distribution_perplexity_evaluator(hf_model, tokenizer, evaluator, out_dir)


def evaluate_model_with_multiple_evaluators(hf_model, tokenizer, evaluators, model_dir):
    """
    Evaluates the model using a list of evaluators.
    :param hf_model: the loaded model
    :param tokenizer: the tokenizer
    :param evaluators: the list of evaluators
    :param out_dir: the directory of the model that we will output our directories into
    :return: nothing
    """


    for evaluator in evaluators:
        out_dir = os.path.join(model_dir, evaluator.label)
        os.makedirs(out_dir, exist_ok=True)
        evaluate_model_with_single_evaluators(hf_model, tokenizer, evaluator, out_dir)

