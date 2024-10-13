import argparse
import os

from tqdm import tqdm
from transformers import DefaultDataCollator, TrainingArguments

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import load_and_reformat_dataset
from src.modules.data.format_utils import format_to_pretraining
from src.modules.data.load import read_dataset_to_hf
from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.modeling.inference import run_inference, obtain_logit
from src.modules.modeling.modeling_utils import setup_model, free_gpus
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config


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

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    ### Performs the training and saving
    if configs.train.do:
        exp_configs = configs.train
        print("train output directory: ", exp_configs.out_directory)

        model = setup_model(exp_configs.model_path_or_name, trust_remote_code=True)
        max_len = min(model.config.max_position_embeddings, exp_configs.max_seq_len)
        tokenizer = load_tokenizer(exp_configs.tokenizer_name, max_len)

        if configs.wandb.do:
            prepare_wandb(configs.wandb)
        else:
            os.environ["WANDB_MODE"] = "dryrun"
            
        chinese_dataset = read_dataset_to_hf(exp_configs.chinese_dataset)["train"] # note we don't shuffle since we want to select held out set
        english_dataset = read_dataset_to_hf(exp_configs.english_dataset)["train"] # note we don't shuffle since we want to select held out set

        # partition out the training dataset
        chinese_dataset = chinese_dataset.select(range(0, len(chinese_dataset) - exp_configs.held_out_size))
        english_dataset = english_dataset.select(range(0, len(english_dataset) - exp_configs.held_out_size))

        # make sure that the two datasets are the same since they should be exact translations of one another
        assert (len(chinese_dataset) == len(english_dataset))

        # we shuffle the two datasets together
        chinese_dataset = chinese_dataset.shuffle(seed=configs.seed)
        english_dataset = english_dataset.shuffle(seed=configs.seed)

        # filter both to only extract entailmant sequences (label of 0)
        chinese_dataset = chinese_dataset.filter(lambda x: x["label"] == 0)
        english_dataset = english_dataset.filter(lambda x: x["label"] == 0)

        # we merge the chinese and english datasets together

        def merge_function_and_tokenize(chinese_row, idx, english_dataset, tokenizer):
            """
            We merge and tokenize the chinese and english datasets.
            loss_mask of 0 means chinese, loss_mask of 1 means english
            """
            english_row = english_dataset[idx]
            chinese_premise = tokenizer.encode(chinese_row["premise"], add_special_tokens=False)
            english_premise = tokenizer.encode(english_row["premise"], add_special_tokens=False)
            chinese_hypothesis = tokenizer.encode(chinese_row["hypothesis"], add_special_tokens=False)
            english_hypothesis = tokenizer.encode(english_row["hypothesis"], add_special_tokens=False)

            total_concat = tokenizer.encode(chinese_row["premise"] + english_row["premise"] + chinese_row["hypothesis"] + english_row["hypothesis"], add_special_tokens=False)

            total_ids = chinese_premise + english_premise + chinese_hypothesis + english_hypothesis
            # if not total_ids == total_concat:
            #     import pdb
            #     pdb.set_trace()

            # for masked
            # input_ids = chinese_premise + english_hypothesis
            # loss_mask = [0] * len(chinese_premise) + [1] * len(english_hypothesis)
            # attention_mask = [1] * len(input_ids)

            # for vanilla
            # input_ids = chinese_premise + english_hypothesis
            # loss_mask = [1] * len(chinese_premise) + [1] * len(english_hypothesis)
            # attention_mask = [1] * len(input_ids)

            # for chinese only
            # input_ids = chinese_premise
            # loss_mask = [1] * len(chinese_premise)
            # attention_mask = [1] * len(input_ids)

            # for english only
            input_ids = english_hypothesis
            loss_mask = [1] * len(english_hypothesis)
            attention_mask = [1] * len(input_ids)

            return {"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}

        dataset = chinese_dataset.map(merge_function_and_tokenize,
                                      with_indices=True,
                                      batched=False,
                                      remove_columns=chinese_dataset.column_names,
                                      fn_kwargs={"english_dataset": english_dataset,
                                                 "tokenizer": tokenizer})

        # reformat to pretraining
        # NOTE: DO NOT SHUFFLE HERE - we want to keep the order of the dataset consistent
        dataset_formatted = format_to_pretraining(dataset, tokenizer, max_len)

        batch_size = exp_configs.per_device_train_batch_size * exp_configs.gradient_accumulation_steps


        # we truncate the dataset to have enough tokens with loss
        if exp_configs.count_with_backprp_tokens:
            total_loss_tokens = exp_configs.max_steps * batch_size * exp_configs.max_seq_len

            def filter_sequence_until_k_loss_tokens(input):
                """
                accept the sequences of the dataset until we have k loss tokens
                """
                nonlocal total_loss_tokens

                # we count the total number of tokens with loss_mask != 0
                num_zeros = sum([1 for x in input["loss_mask"] if x == 0])
                num_nonzero = len(input["loss_mask"]) - num_zeros

                total_loss_tokens -= num_nonzero

                if total_loss_tokens < 0:
                    return False
                return True

            dataset_formatted = dataset_formatted.filter(filter_sequence_until_k_loss_tokens)
        else:
            # else we just truncate the dataset until max_steps
            dataset_formatted = dataset_formatted.select(range(exp_configs.max_steps * batch_size))

        print(dataset_formatted)

        ### setup the training arguments
        # This only helps with batching - we assume that our data is already padded
        data_collator = DefaultDataCollator()
        #return the trained model
        training_args = TrainingArguments(
            output_dir=exp_configs.out_directory,
            # overwrite_output_dir=True,
            per_device_train_batch_size=exp_configs.per_device_train_batch_size,
            gradient_accumulation_steps=exp_configs.gradient_accumulation_steps,
            num_train_epochs=exp_configs.num_train_epochs,
            # max_steps=exp_configs.max_steps, # we don't use max_steps since we want to train over entire dataset
            eval_strategy="no",
            logging_steps=5,
            seed=configs.seed,
            fp16=exp_configs.fp16,
            report_to="wandb",
            run_name=configs.exp_name,
            save_strategy="epoch",
            save_total_limit=1,
            remove_unused_columns=False,
            warmup_ratio=0.1
        )

        ### setup the trainer
        trainer = SelectiveLossTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_formatted,
            tokenizer=tokenizer,
        )

        ### train the model
        trainer.train()

        save_config(configs, os.path.join(exp_configs.out_directory, "config.yaml"))

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