from typing import Dict, Union, Any

import torch
from torch import nn
from transformers.trainer import Trainer
import json

#Refer to https://github.com/huggingface/transformers/blob/v4.38.1/src/transformers/trainer.py#L2876 for original training step
class SelectiveLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
                How the loss is computed by Trainer. By default, all models return the loss in the first element.

                Subclass and override for custom behavior.
                """
        # forward pass

        labels = inputs["input_ids"]
        outputs = model(input_ids = inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)

        logits = outputs.logits

        # We shift the labels
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        loss_mask = inputs["loss_mask"][..., 1:].contiguous()
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        # batch_size x seq_len
        nll_loss = log_probs.gather(dim=-1, index=labels).squeeze(-1)
        nll_loss = nll_loss.masked_fill_(~loss_mask.bool(), 0)

        # sum over the sequence length
        loss = -1 * nll_loss.sum() / loss_mask.sum()

        return loss
