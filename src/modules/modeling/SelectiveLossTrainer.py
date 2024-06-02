from typing import Dict, Union, Any

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
from transformers.trainer import Trainer
import json

from src.modules.modeling.inference import obtain_logit
from src.modules.templates import DYNAHATE_LABEL_IDS


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

        # for ROC-AUC curves
        round_3_labels = []
        round_4_labels = []
        round_3_probs = []
        round_4_probs = []

        total_round_3 = 0
        total_round_4 = 0

        logit_level_correct_3 = 0
        logit_level_correct_4 = 0
        logit_level_correct_total = 0
        generation_correct_3 = 0
        generation_correct_4 = 0
        generation_correct_total = 0

        with torch.no_grad():
            for i, data in tqdm(enumerate(eval_dataloader)):

                input_ids = data["input_ids"]

                attention_mask = data["attention_mask"]

                logits = obtain_logit(self.model, input_ids, attention_mask)

                # take the logit of the last token
                last_token = logits[:, -1, :]

                #for olmo
                true_token = DYNAHATE_LABEL_IDS[True]
                false_token = DYNAHATE_LABEL_IDS[False]

                predictions = last_token[:, true_token] > last_token[:, false_token]

                # record total number for each round and ROCAUC
                if (data["round_info"] == 3):
                    total_round_3 += 1
                    round_3_labels += data["final_label"].tolist()
                    probs = torch.nn.functional.softmax(last_token[0, [true_token, false_token]], dim=-1)
                    #we record the probability of the true token
                    round_3_probs += [probs.tolist()[0]]
                else:
                    total_round_4 += 1
                    round_4_labels += data["final_label"].tolist()
                    probs = torch.nn.functional.softmax(last_token[0, [true_token, false_token]], dim=-1)
                    #we record the probability of the true token
                    round_4_probs += [probs.tolist()[0]]

                #for logit level
                if predictions[0] == data["final_label"]:
                    if data["round_info"] == 3:
                        logit_level_correct_3 += 1
                    else:
                        logit_level_correct_4 += 1
                    logit_level_correct_total += 1

                # for generation
                if last_token[0].argmax() == DYNAHATE_LABEL_IDS[data["final_label"].tolist()[0]]:
                    if data["round_info"] == 3:
                        generation_correct_3 += 1
                    else:
                        generation_correct_4 += 1
                    generation_correct_total += 1

        phase_3_rocauc = roc_auc_score(round_3_labels, round_3_probs)
        phase_4_rocauc = roc_auc_score(round_4_labels, round_4_probs)

        metrics = {"test_logit_accuracy_round3": logit_level_correct_3 / total_round_3,
                    "test_logit_accuracy_round4": logit_level_correct_4 / total_round_4,
                    "test_logit_accuracy_total": logit_level_correct_total / len(eval_dataloader),
                    "test_generation_accuracy_round3": generation_correct_3 / total_round_3,
                    "test_generation_accuracy_round4": generation_correct_4 / total_round_4,
                    "test_generation_accuracy_total": generation_correct_total / len(eval_dataloader),
                    "test_rocauc_round3": phase_3_rocauc,
                    "test_rocauc_round4": phase_4_rocauc,
                }

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
