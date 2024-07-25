import argparse
import json
import os

import pandas as pd
from transformers import DefaultDataCollator, TrainingArguments

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

from tqdm import tqdm

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.datasets.PandasDataset import PandasDataset
from src.modules.data.load import read_dataset_to_hf, read_lines_from_file


from sklearn.model_selection import train_test_split

from src.modules.modeling.modeling_utils import setup_model, free_gpus, setup_model_torch
from src.modules.modeling.models.LogisticRegression import BinaryClassifier
from src.modules.utils import load_config,  validate_inputs

def default_accuracy_metric(logits, y):
    preds = torch.argmax(logits, dim=1)
    return torch.sum(preds == y).item() / len(y)

def train_classifier(model, train_dataset, test_dataset, num_train_epochs, batch_size, output_dir, metric_func=default_accuracy_metric):
    """
    Trains a classifier model_module on the train_dataset and evaluates on the test_dataset
    Assumes that train_dataset and test_dataset use CrossEntropyLoss
    """

    loss_func = nn.CrossEntropyLoss()
    # Cross-entropy loss is just softmax regression loss
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.1)
    # Stochastic gradient descent optimizer

    # Simple version of early stopping: save the best model_module checkpoint based on dev accuracy
    best_dev_acc = -1
    best_epoch = -1

    # prepare output file
    out_fn = open(os.path.join(output_dir, "train_log.txt"), "w")

    for t in range(num_train_epochs):
        aggregate_metrics_train = []
        weight_for_each_batch_train = []

        # Training loop
        model.train()
        # Set model_module to "training mode", e.g. turns dropout on if you have dropout layers
        for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):

            x_batch, y_batch = batch
            optimizer.zero_grad()

            logits = model(x_batch.to("cuda")).to("cpu")


            loss = loss_func(logits, y_batch.long())

            loss.backward()

            optimizer.step()

            aggregate_metrics_train.append(metric_func(logits, y_batch))
            weight_for_each_batch_train.append(len(y_batch))

        # Evaluate train and dev accuracy at the end of each epoch
        normalized_weights_train = torch.tensor(weight_for_each_batch_train) / sum(weight_for_each_batch_train)
        train_acc = torch.tensor(aggregate_metrics_train).dot(normalized_weights_train).item()

        model.eval()
        with torch.no_grad():
            aggregate_metrics_eval = []
            weight_for_each_batch_eval = []
            for X_dev, y_dev in DataLoader(test_dataset, batch_size=batch_size):
                dev_logits = model(X_dev.to("cuda")).to("cpu")
                aggregate_metrics_eval.append(metric_func(dev_logits, y_dev))
                weight_for_each_batch_eval.append(len(y_dev))

            normalized_weights_eval = torch.tensor(weight_for_each_batch_eval) / sum(weight_for_each_batch_eval)
            dev_acc = torch.tensor(aggregate_metrics_eval).dot(normalized_weights_eval).item()
        print(f' Epoch {t: <2}: train_acc={train_acc:.5f}, dev_acc={dev_acc:.5f}')
        out_fn.write(f' Epoch {t: <2}: train_acc={train_acc:.5f}, dev_acc={dev_acc:.5f}\n')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), f"{output_dir}/tmp_best_model.pth")
            best_epoch = t
    print(f"Best dev accuracy: {best_dev_acc} at epoch {best_epoch}")
    out_fn.write(f"Best dev accuracy: {best_dev_acc} at epoch {best_epoch}")
    return best_dev_acc


def train_binaryclassifier_multi(train_dataset, eval_dataset, test_dataset, num_train_epochs, batch_size, output_dir_global,
                     metric_func=default_accuracy_metric):
    """
    Trains multiple classifier model_module on the train_dataset and evaluates on the test_dataset
    Assumes that train_dataset and test_dataset use CrossEntropyLoss
    The number of trained classifiers depend on how many label inputs are given. Assumes each entry in label is a probability
    """

    # checks to make sure train_dataset label is a list
    assert hasattr(train_dataset[0][1], "__len__")

    num_classifiers = len(train_dataset[0][1])
    hidden_size = train_dataset[0][0].shape[0]

    for classifier_num in range(num_classifiers):
        model = BinaryClassifier(input_dim=hidden_size).to("cuda")

        output_dir = os.path.join(output_dir_global, f"classifier_{classifier_num}")
        os.makedirs(output_dir, exist_ok=True)

        loss_func = nn.CrossEntropyLoss()
        # Cross-entropy loss is just softmax regression loss
        optimizer = optim.AdamW(model.parameters(), weight_decay=0.1)
        # Stochastic gradient descent optimizer

        # Simple version of early stopping: save the best model_module checkpoint based on dev accuracy
        best_dev_metric = -1
        best_epoch = -1

        # prepare output file
        out_fn = open(os.path.join(output_dir, "train_log.txt"), "w")

        for t in range(num_train_epochs):
            aggregate_metrics_train = []
            weight_for_each_batch_train = []

            # Training loop
            model.train()
            # Set model_module to "training mode", e.g. turns dropout on if you have dropout layers
            for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
                x_batch, y_batch = batch

                optimizer.zero_grad()

                logits = model(x_batch.to("cuda")).to("cpu")

                current_label = y_batch[:, classifier_num]
                current_label = torch.stack([1-current_label, current_label], dim=1)

                loss = loss_func(logits, current_label)

                loss.backward()

                optimizer.step()

                aggregate_metrics_train.append(metric_func(logits, current_label))
                weight_for_each_batch_train.append(len(y_batch))

            # Evaluate train and dev accuracy at the end of each epoch
            normalized_weights_train = torch.tensor(weight_for_each_batch_train) / sum(weight_for_each_batch_train)
            train_metric = torch.tensor(aggregate_metrics_train).dot(normalized_weights_train).item()

            model.eval()
            with torch.no_grad():
                aggregate_metrics_eval = []
                weight_for_each_batch_eval = []
                for X_dev, y_dev in DataLoader(eval_dataset, batch_size=batch_size):
                    dev_logits = model(X_dev.to("cuda")).to("cpu")
                    dev_label = y_dev[:, classifier_num]
                    dev_label = torch.stack([1 - dev_label, dev_label], dim=1)
                    aggregate_metrics_eval.append(metric_func(dev_logits, dev_label))
                    weight_for_each_batch_eval.append(len(y_dev))

                normalized_weights_eval = torch.tensor(weight_for_each_batch_eval) / sum(weight_for_each_batch_eval)
                dev_metric = torch.tensor(aggregate_metrics_eval).dot(normalized_weights_eval).item()
            print(f' Epoch {t: <2}: train_metric={train_metric:.5f}, dev_metric={dev_metric:.5f}')
            out_fn.write(f' Epoch {t: <2}: train_metric={train_metric:.5f}, dev_metric={dev_metric:.5f}\n')
            if dev_metric > best_dev_metric:
                best_dev_metric = dev_metric
                torch.save(model.state_dict(), f"{output_dir}/tmp_best_model.pth")
                best_epoch = t
        print(f"Best dev metric: {best_dev_metric} at epoch {best_epoch}")
        out_fn.write(f"Best dev metric: {best_dev_metric} at epoch {best_epoch}")

        # load the best model
        best_model = BinaryClassifier(hidden_size)
        best_model.load_state_dict(torch.load(os.path.join(output_dir, "tmp_best_model.pth")))
        best_model.to("cuda")

        # test the best model
        best_model.eval()
        with torch.no_grad():
            aggregate_metrics_test = []
            weight_for_each_batch_test = []
            for X_test, y_test in DataLoader(test_dataset,
                                             batch_size=batch_size):
                test_logits = best_model(X_test.to("cuda")).to("cpu")
                test_label = y_test[:, classifier_num]
                test_label = torch.stack([1 - test_label, test_label], dim=1)
                aggregate_metrics_test.append(metric_func(test_logits, test_label))
                weight_for_each_batch_test.append(len(y_test))


            normalized_weights_test = torch.tensor(weight_for_each_batch_test) / sum(weight_for_each_batch_test)
            test_metric = torch.tensor(aggregate_metrics_test).dot(normalized_weights_test).item()
            print(f'Test metric: {test_metric:.5f}')

        with open(os.path.join(output_dir, "stats.txt"), "w") as f:
            f.write(f"Best dev metric: {best_dev_metric}, test metric: {test_metric}")


def test(model, test_dataset, batch_size, metric_func=default_accuracy_metric):
    model.eval()
    with torch.no_grad():
        aggregate_metrics_test = []
        weight_for_each_batch_test = []
        for X_test, y_test in DataLoader(test_dataset, batch_size=batch_size):
            test_logits = model(X_test.to("cuda")).to("cpu")
            aggregate_metrics_test.append(metric_func(test_logits, y_test))
            weight_for_each_batch_test.append(len(y_test))

        normalized_weights_test = torch.tensor(weight_for_each_batch_test) / sum(weight_for_each_batch_test)
        test_acc = torch.tensor(aggregate_metrics_test).dot(normalized_weights_test).item()
        print(f'Test accuracy: {test_acc:.5f}')
    return test_acc

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

    model = setup_model_torch(configs.model_path_or_name, in_dim=configs.in_dim)

    ### setup the data, tokenizer, and preprocessing
    def prepare_line(line):
        json_line = json.loads(line)
        return [json_line["hidden_state"], json_line["label"]]
    raw_data = []

    for i, line in tqdm(enumerate(read_lines_from_file(configs.input_dataset_file, prepare_line)), total=configs.tot_num):
        if i < configs.tot_num:
            raw_data.append(line)
        else:
            break
    # raw_data = [line for i, line in enumerate(read_lines_from_file(configs.input_dataset_file, prepare_line)) if i < configs.tot_num]
    df = pd.DataFrame(raw_data, columns=["hidden_state", "label"])

    train_and_eval_dataset, test_dataset = train_test_split(df, test_size=configs.test_split, random_state=configs.seed)
    train_dataset, eval_dataset = train_test_split(train_and_eval_dataset, test_size=configs.eval_split,
                                                   random_state=configs.seed)

    print(f"length of train dataset: {len(train_dataset)}")
    print(f"length of eval dataset: {len(eval_dataset)}")
    print(f"length of test dataset: {len(test_dataset)}")

    train_dataset = PandasDataset(train_dataset)
    eval_dataset = PandasDataset(eval_dataset)
    test_dataset = PandasDataset(test_dataset)

    ### Performs the training and saving
    if configs.train.do:
        print("train output directory: ", configs.train.output_dir)

        # prepare_wandb(configs.exp_name)


        best_dev_acc = train_classifier(model, train_dataset, eval_dataset, configs.train.num_train_epochs, configs.train.batch_size, configs.train.output_dir)

        #load the best model
        best_model = BinaryClassifier(configs.in_dim)
        best_model.load_state_dict(torch.load(f"{configs.train.output_dir}/tmp_best_model.pth"))
        best_model.to("cuda")

        test_acc = test(best_model, test_dataset, configs.train.batch_size)
        with open(f"{configs.train.output_dir}/stats.txt", "w") as f:
            f.write(f"train size: {len(train_dataset)}, eval size: {len(eval_dataset)}, test size: {len(test_dataset)}\n")
            f.write(f"Best dev accuracy: {best_dev_acc}, test acc: {test_acc}")

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