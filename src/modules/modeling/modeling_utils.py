from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch


def setup_model(path_to_model, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(path_to_model, **kwargs)
    print(f"imported model from {path_to_model}")
    return model

def use_peft_model(model, peft_config):
    peft_dataset = get_peft_model(model, peft_config)
    peft_dataset.print_trainable_parameters()
    return peft_dataset