import json

import torch
from tqdm import tqdm


class TokenizerConfig:
    def __init__(self):
        self.prev_padding_side = None
        self.prev_truncation_side = None

    def save_prev_config(self, tokenizer):
        self.prev_padding_side = tokenizer.padding_side
        self.prev_truncation_side = tokenizer.truncation_side

    def reset_config(self, tokenizer):
        tokenizer.padding_side = self.prev_padding_side
        tokenizer.truncation_side = self.prev_truncation_side

    def prepare_generation(self, tokenizer):
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"


def run_inference(model, tokenizer, prompt_hf_dataset, out_fn, batch_size=1, **kwargs):
    """Run inference on the given model and tokenizer using the given dataset
        Assumes that the dataset contains an entry called "prompt"
        """
    # set the tokenizer side to left during generation
    tokenizer_config = TokenizerConfig()
    tokenizer_config.save_prev_config(tokenizer)
    tokenizer_config.prepare_generation(tokenizer)

    has_label = "label" in prompt_hf_dataset.column_names
    has_target_mask = "target_mask" in prompt_hf_dataset.column_names

    with open(out_fn, "w") as out_file:
        progress_bar = tqdm(total=len(prompt_hf_dataset))
        ind = 0
        while (ind < len(prompt_hf_dataset)):
            prompts = prompt_hf_dataset[ind:ind + batch_size]["prompt"]
            if has_label:
                labels = prompt_hf_dataset[ind:ind + batch_size]["label"]
            else:
                labels = None

            if has_target_mask:
                target_mask = prompt_hf_dataset[ind:ind + batch_size]["target_mask"]
            else:
                target_mask = None

            # makes the decision of which output to save
            if (kwargs["type"] == "generate"):
                run_generate(model, tokenizer, prompts, labels, out_file, kwargs["generation_kwargs"])
            elif (kwargs["type"] == "logits"):
                run_logits_compare(model, tokenizer, prompts, labels, out_file, kwargs["target_token_ids"])
            elif (kwargs["type"] == "hidden_state"):
                run_hidden_state(model, tokenizer, prompts, labels, out_file, batch_size)
            elif (kwargs["type"] == "get_loss"):
                run_get_loss(model, tokenizer, prompts, target_mask, out_file, batch_size)
            else:
                raise ValueError("invalid type of generation " + kwargs["type"])

            progress_bar.update(batch_size)
            ind += batch_size

    # reset the padding side
    tokenizer_config.reset_config(tokenizer)


def run_generate(model, tokenizer, prompts, labels, out_file, generation_kwargs):
    """Run inference on the given model and tokenizer using the given dataset
    Assumes that the dataset contains an entry called "prompt"
    """

    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    generated_ids = model.generate(**model_inputs, **generation_kwargs)
    final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for i in range(len(prompts)):
        out_file.write(json.dumps({"completion": final[i][len(prompts[i]):],
                            "prompt": prompts[i],
                            "label": labels[i] if labels else None}
                           ) + "\n")


def run_logits_compare(model, tokenizer, prompts, labels, out_file, target_token_ids):
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    logits = obtain_logit(model, **model_inputs)

    # take the logit of the last token
    last_token = logits[:, -1, :]

    predictions = last_token[:, target_token_ids[0]] > last_token[:, target_token_ids[1]]

    for i in range(len(prompts)):
        out_file.write(json.dumps({"completion": predictions[i].item(),
                            "label": labels[i] if labels else None,
                            "prompt": prompts[i]
                            }
                           ) + "\n")


def run_hidden_state(model, tokenizer, prompts, labels, out_file, batch_size=1):
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    try:
        with torch.no_grad():
            model.eval()
            outputs = model(**model_inputs, output_hidden_states=True)

            batched_hidden_states = outputs.hidden_states[-1][:, -1, :]

        for i in range(len(prompts)):
            out_file.write(json.dumps({"hidden_state": batched_hidden_states[i].tolist(),
                                "label": labels[i] if labels else None,
                                "prompt": prompts[i]
                                }
                               ) + "\n")
    except Exception as e:
        print(e)

def run_get_loss(model, tokenizer, prompts, target_mask, out_file, batch_size):

    assert target_mask is not None, "target mask must be provided for get_loss"

    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True).to("cuda")

    # records which characters each token maps to
    offset_mapping = model_inputs.pop("offset_mapping")

    # forward pass and get tokens
    logits = obtain_logit(model, **model_inputs)

    # get the loss on each token
    labels = model_inputs["input_ids"].clone().cpu()
    cross_entropy_per_token = calculate_loss_across_tokens(logits, labels, shift=True)

    # we now select the important tokens
    important_tokens_batch = []

    #loop over all inputs (O(n^2) but batch size and max sequence length are constants or bounded)
    for i in range(len(prompts)):
        important_tokens = []
        #select the correct offset_mapping for the correct sequence
        current_offset_mapping = offset_mapping[i]

        for token_range_ind in range(len(current_offset_mapping)):
            token_range = current_offset_mapping[token_range_ind]
            if (sum(target_mask[i][token_range[0]:token_range[1]]) > 0):
                important_tokens.append(token_range_ind)

        #we shift everything by 1 because of next token prediction
        important_tokens_batch.append(torch.tensor(important_tokens) - 1)

    for i in range(len(prompts)):
        loss = cross_entropy_per_token[i][important_tokens_batch[i]].mean()
        perplexity = torch.exp(loss)

        out_file.write(json.dumps({"loss": loss.item(),
                            "perplexity": perplexity.item(),
                            "prompt": prompts[i]
                            }
                           ) + "\n")

def obtain_logit(model, input_ids, attention_mask):
    """Given a input_id sequence, return the logit of the next token prediction
    model: the model to use for inference already on cuda
    input_ids: the input_ids to use for inference (already on cuda)
    """

    #model takes in batched inputs
    if (len(input_ids.shape) < 2):
        input_ids = input_ids.unsqueeze(0)
        attention_masks = attention_mask.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().float()

    return logits

#supports batching (B, N, d) as logits and (B, N) as labels
#returns (B, N) list as output
def calculate_loss_across_tokens(logits, labels, shift = False):
    from torch.nn.functional import cross_entropy
    if (shift):
        logits = logits[..., :-1, :]
        labels = labels[..., 1:]
    new_logits = logits.reshape(-1, logits.shape[-1])
    new_labels = labels.reshape(-1)
    cross = cross_entropy(new_logits, new_labels, reduction="none").reshape(logits.shape[:-1])
    return cross



def calculate_perplexity(loss):
    """Given batched loss, calculate the perplexity of the batch"""
    return torch.exp(loss.mean()).item()