import argparse
import os

from tqdm import tqdm
from transformers import DefaultDataCollator, TrainingArguments

from src.modules.data.data_utils import load_tokenizer
from src.modules.data.format_datasets import load_and_reformat_dataset
from src.modules.data.load import read_dataset_to_hf
from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
from peft import get_peft_model, LoraConfig
import torch
from omegaconf import OmegaConf

from src.modules.modeling.evaluate import evaluate_model_with_multiple_evaluators
from src.modules.modeling.inference import run_inference, obtain_logit
from src.modules.modeling.modeling_utils import setup_model, free_gpus
from src.modules.modeling.models.modeling_olmo_custom import CustomOlmoForCausalLM
from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
    save_config, execute_shell_command


def evaluate_model_before_hf_conversion(model_path, evaluators, OLMO_DIR, olmo_type, out_dir=None):
    print(f"model path of {model_path} entered! ")

    # begin conversion of the checkpoint to hf
    print("converting model to hf...")

    # if the current folder is an olmo model , we convert it to hf
    if os.path.exists(os.path.join(model_path, "config.yaml")):
        hf_model_path = os.path.join(model_path, "hf")
    else:
        hf_model_path = model_path

    if olmo_type == "olmo_standard":
        # convert the model if it hasn't been converted yet
        if not os.path.exists(hf_model_path):

            command = f"python {OLMO_DIR}/scripts/convert_olmo_to_hf_new.py --input_dir {model_path} --output_dir {hf_model_path} --tokenizer_json_path {OLMO_DIR}/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"
            execute_shell_command(command)

        hf_model = setup_model(hf_model_path)
    elif olmo_type == "olmo_custom":
        # convert the model if it hasn't been converted yet
        if not os.path.exists(hf_model_path):
            command = f"python {OLMO_DIR}/scripts/convert_custom_olmo_to_hf_new.py --input_dir {model_path} --output_dir {hf_model_path} --tokenizer_json_path {OLMO_DIR}/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"
            execute_shell_command(command)

        hf_model = CustomOlmoForCausalLM.from_pretrained(hf_model_path).to("cuda")
    else:
        raise ValueError(f"olmo_type {olmo_type} not recognized")

    max_len = hf_model.config.max_position_embeddings
    print(f"max_len: {max_len}")
    tokenizer_path_temp = "/home/ryan/decouple/models/olmo_ckpt/contpretrain/exp_9_3epoch/unfiltered_exp9_3epoch/step3000-unsharded/hf"
    tokenizer = load_tokenizer(tokenizer_path_temp, max_len)

    # tokenizer = load_tokenizer(hf_model_path, max_len)
    # import pdb
    # pdb.set_trace()
    # hf_model = None
    # tokenizer = None

    # run the evaluation
    evaluate_model_with_multiple_evaluators(hf_model, tokenizer, evaluators, hf_model_path, out_dir)


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

    # we loop through all the models we have to evaluate

    for model_run_path in configs.model_paths:
        print(f"evaluating model at {model_run_path}")

        # if out_dir is not none, add the current model name to out_dir path
        if "out_dir" in configs:
            out_dir = os.path.join(configs.out_dir, os.path.basename(model_run_path))
        else:
            out_dir = None

        #if there are no checkpoints, we evaluate the model from model_paths directly
        if "checkpoint_names" in configs:
            for checkpoint in configs.checkpoint_names:
                print(f"evaluating checkpoint {checkpoint}")
                model_path = os.path.join(model_run_path, checkpoint)
                if out_dir is not None:
                    out_dir = os.path.join(out_dir, checkpoint)
                evaluate_model_before_hf_conversion(model_path, configs.evaluators, configs.OLMO_DIR, configs.model_type, out_dir)
        else:
            model_path = model_run_path
            evaluate_model_before_hf_conversion(model_path, configs.evaluators, configs.OLMO_DIR, configs.model_type, out_dir)

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


if __name__ == "__main__":
    args = parse_args()
    main(args)