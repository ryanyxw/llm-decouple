import argparse
import os
from transformers import DefaultDataCollator, TrainingArguments

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import load_and_reformat_dataset_for_inference
from src.modules.data.load import read_dataset_to_hf
from src.modules.data.preprocess import preprocess_conversation
from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.modeling.inference import run_inference
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
        print("train output directory: ", configs.train.out_directory)

        model = setup_model(configs.train.model_path_or_name, trust_remote_code=True)
        max_len = min(model.config.max_position_embeddings, configs.max_seq_len)
        tokenizer = load_tokenizer(configs.train.tokenizer_name, max_len)

        prepare_wandb(configs.exp_name)

        ### setup the data, tokenizer, and preprocessing
        raw_dataset = read_dataset_to_hf(configs.train.input_dataset_file)["train"]
        preprocessed_dataset = preprocess_conversation(raw_dataset, tokenizer, configs.max_seq_len, seed=configs.seed, num_proc=configs.num_proc, use_loss_mask=configs.train.use_loss_mask)
        preprocessed_dataset = preprocessed_dataset.select(range(configs.train.num_train_examples))

        ### setup the lora model
        peft_config = LoraConfig(
            target_modules = list(configs.train.lora_modules)
        )
        peft_model = get_peft_model(model, peft_config)

        ### setup the training arguments
        # This only helps with batching - we assume that our data is already padded
        data_collator = DefaultDataCollator()
        #return the trained model
        training_args = TrainingArguments(
            output_dir=configs.train.out_directory,
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

        save_config(configs, os.path.join(configs.train.out_directory, "config.yaml"))

        # free gpu from the model
        import gc
        del model
        del peft_model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()


    ### Performs the inference
    if configs.generate.do:
        print("doing generation!")
        print("generate with model ", configs.generate.inferencemodel_path_or_name)
        model = setup_model(configs.generate.inferencemodel_path_or_name)
        max_len = min(model.config.max_position_embeddings, configs.max_seq_len)
        tokenizer = load_tokenizer(configs.generate.inferencetokenizer_name, max_len)

        reformatted_dataset = load_and_reformat_dataset_for_inference(configs.generate.in_dataset_name,
                                                                      configs.generate.input_dataset_file,
                                                                      configs.generate.num_generate_examples,
                                                                      configs.seed,
                                                                      configs.generate.num_demonstrations,
                                                                      **configs.generate.kwargs)

        # saves a sample of the prompt to a parallel file
        print("sample of example fed into model: \n" + reformatted_dataset[0]["prompt"])
        parent_of_output = os.path.dirname(configs.generate.output_filename)
        orig_name = os.path.basename(configs.generate.output_filename).split(".")
        template_fn = os.path.join(parent_of_output, orig_name[0] + "_template." + orig_name[1])
        with open(template_fn, "w") as f:
            f.write(reformatted_dataset[0]["prompt"])

        ### runs the generation
        out_fn = configs.generate.output_filename
        print("saving to ", out_fn)

        run_inference(model, tokenizer, reformatted_dataset, out_fn, batch_size=configs.generate.batch_size, **configs.generate.kwargs)


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