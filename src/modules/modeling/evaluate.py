import multiprocessing
import os

from omegaconf import OmegaConf

from API_KEYS import PERSPECTIVE_API_KEY
from src.modules.data.load import read_dataset_to_hf
from src.modules.modeling.inference import run_inference_new
from src.modules.utils import use_perspective_api, seed_all


def real_toxicity_prompt_generation_evaluator(hf_model, tokenizer, evaluator, out_dir):
    """
    Evaluates the model using the real toxicity prompts. returns the process for evaluating using perspective
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """

    # load the dataset and select the necessary ones
    dataset = read_dataset_to_hf(evaluator.data.name)["train"].shuffle(seed=evaluator.seed)
    dataset = dataset.select(range(evaluator.data.num_examples))

    # reformat the dataset such that it is in generation format
    def reformat_row(row):
        return {"prompt": row["prompt"]["text"]}
    dataset = dataset.map(reformat_row)

    # saves a sample of the prompt to a parallel file along with configs
    print("sample of example fed into model: \n" + dataset[0]["prompt"])
    with open(os.path.join(out_dir, "template.jsonl"), "w") as f:
        f.write(dataset[0]["prompt"] + "\n")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        OmegaConf.save(evaluator, f, resolve=True)

    # runs the generation and saves the output
    out_fn = os.path.join(out_dir, "generation_output_test3.jsonl")
    print("saving to ", out_fn)

    run_inference_new("generate", hf_model, tokenizer, dataset, out_fn, batch_size=evaluator.batch_size , generation_kwargs=evaluator.generation_kwargs)

    # creates a processes that calls google perspective API and saves the output
    progress_file = os.path.join(out_dir, "perspective_api_progress_includingprompt.json")
    use_perspective_api(out_fn, PERSPECTIVE_API_KEY, progress_file)


def hidden_state_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir):

    #load the dataset and select the necessary ones
    dataset = read_dataset_to_hf(evaluator.data.name)["train"].shuffle(seed=evaluator.seed)
    dataset = dataset.select(range(evaluator.data.num_examples))

    pass

def evaluate_model_with_single_evaluators(hf_model, tokenizer, evaluator, out_dir):
    """
    Evaluates the model using a single evaluator.
    :param hf_model: the loaded model
    :param evaluator: the evaluator
    :return: nothing
    """

    # reset the seed for each evaluator
    seed_all(evaluator.seed)

    if evaluator.label == "realtoxicityprompts_generation":
        real_toxicity_prompt_generation_evaluator(hf_model, tokenizer, evaluator, out_dir)
    elif evaluator.label == "civilcomments_hiddenstate":
        hidden_state_civilcomments_evaluator(hf_model, tokenizer, evaluator, out_dir)


def evaluate_model_with_multiple_evaluators(hf_model, tokenizer, evaluators, model_dir):
    """
    Evaluates the model using a list of evaluators.
    :param hf_model: the loaded model
    :param tokenizer: the tokenizer
    :param evaluators: the list of evaluators
    :param out_dir: the directory of the model that we will output our directories into
    :return: nothing
    """


    for evaluator in evaluators:
        out_dir = os.path.join(model_dir, evaluator.label)
        os.makedirs(out_dir, exist_ok=True)
        evaluate_model_with_single_evaluators(hf_model, tokenizer, evaluator, out_dir)

