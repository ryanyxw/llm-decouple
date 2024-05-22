from typing import Dict, Union, Any

import torch
from torch import nn
from tqdm import tqdm
from transformers.trainer import Trainer
import json

from src.modules.modeling.inference import obtain_logit


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
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        self.model.eval()

        correct_preds = 0

        with torch.no_grad():
            for i, data in tqdm(enumerate(eval_dataloader)):
                input_ids = data["input_ids"]

                attention_mask = data["attention_mask"]

                logits = obtain_logit(self.model, input_ids, attention_mask)

                # take the logit of the last token
                last_token = logits[:, -1, :]

                #for llama: "hate" is 26277, "not" is 451
                predictions = last_token[:, 26277] > last_token[:, 451]

                if predictions[0] == data["final_label"]:
                    correct_preds += 1

        metrics = {"test_accuracy": correct_preds / len(eval_dataloader)}
        # metrics = {'wer': wer}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
