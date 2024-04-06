import torch
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