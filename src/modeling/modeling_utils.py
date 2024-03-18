from transformers import AutoModelForCausalLM
import torch


def setup_model(path_to_model):
    model = AutoModelForCausalLM.from_pretrained(path_to_model, return_dict=True)
    print(f"imported model from {path_to_model}")
    return model