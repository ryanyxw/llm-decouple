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

def train_classifier(model, train_dataset, test_dataset, num_train_epochs, batch_size, output_dir):
    loss_func = nn.CrossEntropyLoss()
    # Cross-entropy loss is just softmax regression loss
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.1)
    # Stochastic gradient descent optimizer

    # Simple version of early stopping: save the best model_module checkpoint based on dev accuracy
    best_dev_acc = -1
    best_epoch = -1

    for t in range(num_train_epochs):
        train_num_correct = 0

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

            # Compute running count of number of training examples correct
            preds = torch.argmax(logits, dim=1)
            # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
            train_num_correct += torch.sum(preds == y_batch).item()

        # Evaluate train and dev accuracy at the end of each epoch
        train_acc = train_num_correct / len(train_dataset)
        model.eval()
        with torch.no_grad():
            num_correct = 0
            for X_dev, y_dev in DataLoader(test_dataset, batch_size=batch_size):
                dev_logits = model(X_dev.to("cuda")).to("cpu")
                dev_preds = torch.argmax(dev_logits, dim=1)
                num_correct += torch.sum(dev_preds == y_dev).item()

            dev_acc = num_correct / len(test_dataset)
        print(f' Epoch {t: <2}: train_acc={train_acc:.5f}, dev_acc={dev_acc:.5f}')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), f"{output_dir}/tmp_best_model.pth")
            best_epoch = t
    print(f"Best dev accuracy: {best_dev_acc} at epoch {best_epoch}")
    return best_dev_acc


def test(model, test_dataset, batch_size):
    model.eval()
    with torch.no_grad():
        num_correct = 0
        for X_test, y_test in DataLoader(test_dataset, batch_size=batch_size):
            test_logits = model(X_test.to("cuda")).to("cpu")
            test_preds = torch.argmax(test_logits, dim=1)
            num_correct += torch.sum(test_preds == y_test).item()

        test_acc = num_correct / len(test_dataset)  # (QUESTION 4a: line 26)
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