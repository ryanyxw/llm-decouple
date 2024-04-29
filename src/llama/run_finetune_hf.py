import argparse
import os
from transformers import DefaultDataCollator, TrainingArguments

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.load import read_dataset_to_hf
from src.modules.data.preprocess import preprocess_conversation
from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.modeling.modeling_utils import setup_model
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

    model = setup_model(configs.model_path_or_name)
    prepare_wandb()


    ### Performs the training and saving
    if configs.train.do:
        ### setup the data, tokenizer, and preprocessing
        raw_dataset = read_dataset_to_hf(configs.train.input_dataset_file)["train"]
        tokenizer = load_tokenizer(configs.tokenizer_name)
        preprocessed_dataset = preprocess_conversation(raw_dataset, tokenizer, configs.train.max_seq_len, seed=configs.seed, num_proc=configs.num_proc, use_loss_mask=configs.train.use_loss_mask)
        preprocessed_dataset.select(range(configs.train.num_train_examples))

        ### setup the lora model
        peft_config = LoraConfig(
            target_modules = list(configs.train.lora_modules)
        )
        peft_model = get_peft_model(model, peft_config)

        ### setup the training arguments
        # This only helps with batching - we assume that our data is already padded
        data_collator = DefaultDataCollator()

        training_args = TrainingArguments(
            output_dir=configs.out_directory,
            overwrite_output_dir=True,
            per_device_train_batch_size=configs.train.per_device_train_batch_size,
            gradient_accumulation_steps=configs.train.gradient_accumulation_steps,
            num_train_epochs=configs.train.num_train_epochs,
            logging_steps=5,
            seed=configs.seed,
            fp16=configs.train.fp16,
            report_to="wandb",
            run_name=configs.exp_name,
            save_strategy="epoch",
            save_total_limit=1,
            remove_unused_columns=False,
        )

        ### setup the trainer
        trainer = SelectiveLossTrainer(
            model=peft_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=preprocessed_dataset,
        )

        ### train the model
        trainer.train()

        save_config(configs, os.path.join(configs.out_directory, "config.yaml"))

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