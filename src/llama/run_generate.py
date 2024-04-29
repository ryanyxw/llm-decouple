import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
import json
import tqdm as tqdm


from src.modules.utils import confirm_with_user, load_config, prepare_folder


def validate_inputs(configs):
    if (configs.mode == "llama"):
        # if not os.path.exists(configs.model_path):
        #     raise ValueError(f"input directory {configs.model_path} does not exist")
        # if os.path.exists(configs.output_file):
        #     if not confirm_with_user(f"output file {configs.output_file} already exists. Do you want to overwrite?"):
        #         raise ValueError("output file already exists")
        pass




def inference_llama(configs):


    dataset = load_dataset(configs.dataset_name)["train"]

    small_dataset = dataset.shuffle(seed=configs.seed).select(range(configs.num_examples))

    model = AutoModelForCausalLM.from_pretrained(configs.model_path, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(configs.tokenizer)

    with open(configs.output_file, "w") as f:
        progress_bar = tqdm.tqdm(total=configs.num_examples)
        for example in small_dataset:
            prompt = example["prompt"]["text"]
            model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            generated_ids = model.generate(**model_inputs, max_length=configs.max_length)
            final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            score = example["prompt"]["toxicity"]

            f.write(json.dumps({"prompt": prompt, "completion": final, "prompt_toxicity": score}) + "\n")
            progress_bar.update(1)



    # model = AutoModelForCausalLM.from_pretrained(configs.model_path, device_map="auto")
    #
    # tokenizer = AutoTokenizer.from_pretrained(configs.tokenizer)
    #
    # model_inputs = tokenizer(["what a great day to"], return_tensors="pt").to("cuda")
    #
    # generated_ids = model.generate(**model_inputs, max_length=50)
    # final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


    # import pdb
    # pdb.set_trace()


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

    if (configs.mode == "generate"):
        inference_llama(configs)


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