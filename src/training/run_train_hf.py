import argparse
import os

from tqdm import tqdm
from transformers import DefaultDataCollator, TrainingArguments, DataCollatorWithPadding

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import load_and_reformat_dataset, prepare_dataset_for_training
from src.modules.data.load import read_dataset_to_hf
from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.modeling.inference import run_inference, obtain_logit, TokenizerConfig
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


    ### Performs the training and saving
    if configs.train.do:
        exp_configs = configs.train

        # we set the out_directory according to the model and dataset used
        # out_directory = exp_configs.out_directory
        out_directory = os.path.join(exp_configs.model_path_or_name, exp_configs.exp_name)
        os.makedirs(out_directory, exist_ok=True)

        save_config(configs, os.path.join(out_directory, "config.yaml"))

        print("train output directory: ", out_directory)

        model = setup_model(exp_configs.model_path_or_name, trust_remote_code=True)
        max_len = min(model.config.max_position_embeddings, exp_configs.max_seq_len)
        exp_configs["max_seq_len"] = max_len # update the max_seq_len in the config
        # max_len = 1024
        # model = None
        tokenizer = load_tokenizer(exp_configs.tokenizer_name, max_len)
        print("loaded model and tokenizer! ")

        # prepare for padding from the beginning
        # tokenizer_config = TokenizerConfig()
        # tokenizer_config.prepare_generation(tokenizer)

        if exp_configs.wandb.do:
            prepare_wandb(exp_configs.wandb)

        train_dataset, eval_datasets = prepare_dataset_for_training(tokenizer,
                                                                    configs.seed,
                                                                    configs.num_proc,
                                                                    **exp_configs)


        assert isinstance(eval_datasets, dict), "eval_datasets should be a dictionary"

        # check if we even need to do eval
        if len(eval_datasets) == 0:
            use_eval = False
        else:
            use_eval = True

        ### setup lora
        if exp_configs.lora.do:
            print("using lora")
            peft_config = LoraConfig(
                target_modules = list(exp_configs.lora.lora_modules)
            )
            model = get_peft_model(model, peft_config)
            #print trainable parameters
            print("trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        ### setup the training arguments
        # This only helps with batching - we assume that our data is already padded
        data_collator = DefaultDataCollator()
        #return the trained model
        training_args = TrainingArguments(
            output_dir=out_directory,
            overwrite_output_dir=True,
            eval_strategy="steps" if use_eval else "no",
            per_device_eval_batch_size=exp_configs.eval.per_device_eval_batch_size,
            eval_steps=exp_configs.eval.eval_steps,
            seed=configs.seed,
            report_to="wandb" if exp_configs.wandb.do else "none",
            save_strategy="epoch" if exp_configs.save_model else "no",
            save_total_limit=1,
            remove_unused_columns=False,
            **exp_configs.training_args
        )

        ### setup the trainer
        trainer = SelectiveLossTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets if use_eval else None,
            tokenizer=tokenizer,
        )

        ### train the model
        trainer.train()


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