import argparse
import os
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, \
    DefaultDataCollator, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.utils import confirm_with_user, load_config, prepare_folder


def validate_inputs(configs):
    if (configs.mode == "finetune_llama"):
        prepare_folder(configs.output_dir)
    if (configs.mode == "finetune_olmo"):
        pass



def finetune_llama(configs):
    model = AutoModelForCausalLM.from_pretrained(configs.model_name, trust_remote_code=True)

    peft_config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # max_length = model.config.max_position_embeddings
    max_length=512

    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(configs.model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_length


    # We want to tokenize each example such that the label is -100
    def prepare_dataset(dataset):
        #tokenize the dataset with truncation only. leave padding for data colaltor
        def tokenize_function(examples):
            tokenized_inputs = tokenizer(examples["text"])

            toxic_length = max_length - len(appended_tokenized)

            if (toxic_length > len(tokenized_inputs["input_ids"])):
                pad_tokens = toxic_length - len(tokenized_inputs["input_ids"])
                toxic_tokens = len(tokenized_inputs["input_ids"])
            else:
                pad_tokens = 0
                toxic_tokens = toxic_length

            return_input_ids = tokenized_inputs["input_ids"][:toxic_tokens] + appended_tokenized + [tokenizer.pad_token_id] * pad_tokens
            return_attention_mask = tokenized_inputs["attention_mask"][:toxic_tokens] + [1] * len(appended_tokenized) + [0] * pad_tokens
            #stores the loss_mask
            return_labels = [0] * toxic_tokens + [1] * len(appended_tokenized) + [0] * pad_tokens

            if (configs.use_loss_mask == False):
                return_labels = [1 for _ in range(len(return_labels) - pad_tokens)] + [0] * pad_tokens

            if (len(return_input_ids) != len(return_labels) or len(return_input_ids) != len(return_attention_mask) or len(return_labels) != len(return_attention_mask)):
                import pdb
                pdb.set_trace()

            return {
                "input_ids": torch.tensor(return_input_ids),
                "attention_mask": torch.tensor(return_attention_mask),
                "loss_mask": torch.tensor(return_labels)
            }
        return dataset.map(tokenize_function, batched=False, remove_columns=dataset.column_names)

    if (configs.is_conversational):
        # import pdb
        tokenized_dataset = load_from_disk(configs.dataset_path)

        if (not configs.use_loss_mask):
            def add_loss_to_all_except_pad(example):
                example["loss_mask"] = [1 if x != tokenizer.pad_token_id else 0 for x in example["input_ids"]]
                return example

            tokenized_dataset = tokenized_dataset.map(add_loss_to_all_except_pad, batched=False, num_proc=configs.num_proc)


        # def convert_to_torch(example):
        #     return {
        #         "input_ids": torch.tensor(example["input_ids"]),
        #         "attention_mask": torch.tensor(example["attention_mask"]),
        #         "loss_mask": torch.tensor(example["loss_mask"])
        #     }
        #
        # tokenized_dataset = tokenized_dataset.map(convert_to_torch, batched=False, num_proc=configs.num_proc)

    if (configs.is_puretext):
        # load the dataset
        dataset = load_dataset("json", data_files=configs.dataset_path)["train"]



        # tokenize the appended string
        appended_tokenized = tokenizer(configs.appended_string)["input_ids"]

        tokenized_dataset = prepare_dataset(dataset)


    tokenized_dataset = tokenized_dataset.shuffle(seed=configs.seed)
    #
    # import pdb
    # pdb.set_trace()

    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir=configs.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=configs.per_device_train_batch_size,
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        num_train_epochs=configs.num_train_epochs,
        logging_steps=5,
        seed=configs.seed,
        fp16=configs.fp16,
        report_to="wandb",
        run_name=configs.run_name,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = SelectiveLossTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    trainer.train()



def main(args):
    print("yay!")
    #load the config file
    print("loading config file...")
    configs = load_config(args.config_file)

    #set the args to be the configs
    for key, value in args.__dict__.items():
        configs.__setattr__(key, value)

    # target exists and destination does not exist, creating output directories
    validate_inputs(configs)

    print("executing command...")



    if (configs.mode == "finetune_llama"):
        finetune_llama(configs)


    if not os.path.exists(os.path.join(configs.output_dir, "configs")):
        os.makedirs(os.path.join(configs.output_dir, "configs"))
    with open(os.path.join(configs.output_dir, "configs", "config.json"), "w") as f:
        OmegaConf.save(configs, f)



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