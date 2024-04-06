from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch


def setup_model(path_to_model, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(path_to_model, return_dict=True, **kwargs)
    print(f"imported model from {path_to_model}")
    return model

#sets up tokenizer
def setup_tokenizer(path_to_tokenizer):
    return AutoTokenizer.from_pretrained(path_to_tokenizer)