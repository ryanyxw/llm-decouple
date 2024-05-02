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




def run_generate(model, tokenizer, prompt_hf_dataset, output_file, max_gen_len, batch_size=1):
    """Run inference on the given model and tokenizer using the given dataset
    Assumes that the dataset contains an entry called "prompt"
    """
    # set the tokenizer side to left during generation
    tokenizer_config = TokenizerConfig()
    tokenizer_config.save_prev_config(tokenizer)
    tokenizer_config.prepare_generation(tokenizer)

    has_label = "label" in prompt_hf_dataset.column_names

    with open(output_file, "w") as f:
        progress_bar = tqdm(total=len(prompt_hf_dataset))
        ind = 0
        while (ind < len(prompt_hf_dataset)):
            prompts = prompt_hf_dataset[ind:ind + batch_size]["prompt"]
            if has_label:
                labels = prompt_hf_dataset[ind:ind + batch_size]["label"]

            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            generated_ids = model.generate(**model_inputs, max_new_tokens=max_gen_len)
            final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i in range(len(prompts)):
                f.write(json.dumps({"completion": final[i][len(prompts[i]) - 1:],
                                    "prompt": prompts[i],
                                    "label": labels[i] if has_label else None}
                                   ) + "\n")
            progress_bar.update(batch_size)
            ind += batch_size

    # reset the padding side
    tokenizer_config.reset_config(tokenizer)


def run_logits_compare(model, tokenizer, prompt_hf_dataset, output_file, target_token_ids, batch_size=1):

    # set the tokenizer side to left during generation
    tokenizer_config = TokenizerConfig()
    tokenizer_config.save_prev_config(tokenizer)
    tokenizer_config.prepare_generation(tokenizer)

    has_label = "label" in prompt_hf_dataset.column_names

    with open(output_file, "w") as f:
        progress_bar = tqdm(total=len(prompt_hf_dataset))
        ind = 0
        while (ind < len(prompt_hf_dataset)):
            prompts = prompt_hf_dataset[ind:ind + batch_size]["prompt"]
            if has_label:
                labels = prompt_hf_dataset[ind:ind + batch_size]["label"]

            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            print(model_inputs["input_ids"].shape)

            logits = obtain_logit(model, **model_inputs)

            # take the logit of the last token
            last_token = logits[:, -1, :]

            predictions = last_token[:, target_token_ids[0]] > last_token[:, target_token_ids[1]]

            for i in range(len(prompts)):
                f.write(json.dumps({"completion": predictions[i].item(),
                                    "label": labels[i] if has_label else None,
                                    "prompt": prompts[i]
                                    }
                                   ) + "\n")
            progress_bar.update(batch_size)
            ind += batch_size

    # reset the padding side
    tokenizer_config.reset_config(tokenizer)

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