import json

import torch
from tqdm import tqdm


def run_inference(model, tokenizer, prompt_hf_dataset, output_file, max_gen_len, batch_size=1):
    """Run inference on the given model and tokenizer using the given dataset
    Assumes that the dataset contains an entry called "prompt"
    """

    prev_padding_side = tokenizer.padding_side
    # set the tokenizer side to left during generation
    tokenizer.padding_side = "left"

    with open(output_file, "w") as f:
        progress_bar = tqdm(total=len(prompt_hf_dataset))
        ind = 0
        while (ind < len(prompt_hf_dataset)):
            prompts = prompt_hf_dataset[ind:ind + batch_size]["prompt"]
            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            generated_ids = model.generate(**model_inputs, max_length=max_gen_len)
            final = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for i in range(len(prompts)):
                f.write(json.dumps({"prompt": prompts[i], "completion": final[i]}) + "\n")
            progress_bar.update(batch_size)
            ind += batch_size

    # reset the padding side
    tokenizer.padding_side = prev_padding_side

def obtain_logit(model, input_ids):
    """Given a input_id sequence, return the logit of the next token prediction
    model: the model to use for inference already on cuda
    input_ids: the input_ids to use for inference (already on cuda)
    """

    #model takes in batched inputs
    if (len(input_ids.shape) < 2):
        input_ids = input_ids.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        outputs = model(input_ids)
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