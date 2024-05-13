import os

from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

from src.modules.modeling.models.LogisticRegression import BinaryClassifier


def setup_model(path_to_model, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(path_to_model, **kwargs)
    print(f"imported model from {path_to_model}")
    return model.to("cuda")

def setup_model_torch(path_to_model, **kwargs):
    if (path_to_model == "logistic_reg"):
        model = BinaryClassifier(kwargs["in_dim"]).to("cuda")
        return model
    elif (os.path.exists(path_to_model)):
        model = torch.load(path_to_model).to("cuda")
        print(f"imported model from {path_to_model}")
        return model
    else:
        raise FileNotFoundError(f"unknown model name at {path_to_model}")

def use_peft_model(model, peft_config):
    peft_dataset = get_peft_model(model, peft_config)
    peft_dataset.print_trainable_parameters()
    return peft_dataset

#will call the garbage collector on the indefinite list of pointers given
def free_gpus(*args):
    import gc
    for arg in args:
        del arg
    torch.cuda.empty_cache()
    gc.collect()
