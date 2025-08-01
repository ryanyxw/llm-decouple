import os
from typing import Dict, Union, Any

import numpy as np
import torch
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.stats import sem
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from torch import nn
from tqdm import tqdm
from transformers.trainer import Trainer
import json

from src.modules.modeling.inference import obtain_logit
from src.modules.templates import DYNAHATE_LABEL_IDS, WILDGUARD_PROMPT_ONLY_LABELS, TOXIC_CLASSIFICATION_LABELS


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
        attention_mask = inputs["attention_mask"][..., 1:].contiguous()
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        do_vanilla = False # whether we even use the loss_mask
        mode = "masked"

        if mode != "vanilla" and do_vanilla:
            raise ValueError("when do_vanilla is true, mode must be set to vanilla")

        # create masks for cross_entropy and masked/unlikelihood
        if do_vanilla == True:
            label_mask_for_cross_entropy = attention_mask.bool()
        else:
            label_mask_for_cross_entropy = loss_mask.bool() & attention_mask.bool()
            label_mask_for_special_tokens = (~loss_mask.bool()) & attention_mask.bool()

        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        # batch_size x seq_len
        ll_loss = log_probs.gather(dim=-1, index=labels).squeeze(-1)

        # we take out the special and padding tokens
        ll_loss = ll_loss.masked_fill_(~label_mask_for_cross_entropy, 0)

        # return loss
        if mode == "masked" or mode == "vanilla":
            loss = -1 * ll_loss.sum() / label_mask_for_cross_entropy.sum()
            return loss

        elif mode == "unlikelihood":
            probs = log_probs.exp()

            probs_for_ul = probs[label_mask_for_special_tokens]
            labels_for_ul = labels[label_mask_for_special_tokens]

            # Compute 1 - p (we clamp to avoid log(0))
            p_unlike = 1.0 - probs_for_ul.gather(-1, labels_for_ul).squeeze(-1)
            p_unlike = torch.clamp(p_unlike, min=1e-9)

            ul_loss = torch.log(p_unlike)

            loss = -1 * (ll_loss.sum() + ul_loss.sum()) / (label_mask_for_cross_entropy.sum() + label_mask_for_special_tokens.sum())
            return loss



    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run evaluation and returns metrics
        :param eval_dataset: this MUST be a dictionary of datasets to evaluate on
        :param ignore_keys: ignore
        :param metric_key_prefix: ignore
        :return:
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset

        assert isinstance(eval_dataset, dict), "eval_dataset must be a dictionary of datasets to evaluate on"

        metrics = {}

        # loops through all the datasets and evaluate on them
        for eval_dataset_name, _eval_dataset in eval_dataset.items():
            self._memory_tracker.start()

            eval_dataloader = self.get_eval_dataloader(_eval_dataset)

            self.model.eval()

            # this is for legacy (on dynahate dataset)
            if eval_dataset_name == "dynahate":
                temp_metrics = self.dynahate_eval(eval_dataloader)
                metrics.update(temp_metrics)
            elif "NEW_civilcomments_finetune_auroc" in eval_dataset_name:
                temp_metrics = self.NEW_civilcomments_finetune_auroc(eval_dataloader)
                metrics.update(temp_metrics)
            elif "wildguard_prompt" in eval_dataset_name or "wildguard_lowdata_prompt" in eval_dataset_name:
                temp_metrics = self.wildguard_prompt_eval(eval_dataloader)
                metrics.update(temp_metrics)
            elif "paradetox" in eval_dataset_name:
                temp_metrics = self.paradetox_eval(eval_dataloader)
                metrics.update(temp_metrics)
            elif "toxigen_prompt_sft" in eval_dataset_name or "toxigen_lowdata_prompt_sft" in eval_dataset_name:
                temp_metrics = self.toxigen_prompt_sft_eval(eval_dataloader)
                metrics.update(temp_metrics)
            elif "dynahate_prompt_sft" in eval_dataset_name or "dynahate_lowdata_prompt_sft" in eval_dataset_name:
                temp_metrics = self.dynahate_prompt_sft_eval(eval_dataloader)
                metrics.update(temp_metrics)
            # elif "toxigen_finetune_prompts" in eval_dataset_name:
            #     temp_metrics = self.toxigen_eval(eval_dataloader)
            #     metrics.update(temp_metrics)
            else:
                raise ValueError(f"Unknown dataset {eval_dataset_name}")

            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics


    def dynahate_eval(self, eval_dataloader):
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

                # for olmo
                true_token = DYNAHATE_LABEL_IDS[True]
                false_token = DYNAHATE_LABEL_IDS[False]

                predictions = last_token[:, true_token] > last_token[:, false_token]

                # record total number for each round and ROCAUC
                if (data["round_info"] == 3):
                    total_round_3 += 1
                    round_3_labels += data["final_label"].tolist()
                    probs = torch.nn.functional.softmax(last_token[0, [true_token, false_token]], dim=-1)
                    # we record the probability of the true token
                    round_3_probs += [probs.tolist()[0]]
                else:
                    total_round_4 += 1
                    round_4_labels += data["final_label"].tolist()
                    probs = torch.nn.functional.softmax(last_token[0, [true_token, false_token]], dim=-1)
                    # we record the probability of the true token
                    round_4_probs += [probs.tolist()[0]]

                # for logit level
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

        return metrics

    def NEW_civilcomments_finetune_auroc(self, eval_dataloader):
        total_predictions = []
        total_probs = []
        total_labels = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(eval_dataloader)):
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                loss_mask = data["loss_mask"]

                logits = obtain_logit(self.model, input_ids, attention_mask).to("cpu")

                # take the logit of the answer token (corresponding to the token with value of 1 in loss_mask
                loss_mask = loss_mask.to(logits.device)  # Move loss_mask to the same device as logits if necessary

                # Get the logits corresponding to the last non-padding token
                last_token = logits[loss_mask.bool()].view(logits.shape[0], -1)

                # check its accuracy on the next token
                true_token = self.tokenizer.encode(" Yes")[0]
                false_token = self.tokenizer.encode(" No")[0]

                predictions = last_token[:, true_token] > last_token[:, false_token]
                target_token_logits = [last_token[:, true_token], last_token[:, false_token]]
                target_token_logits_transformed = torch.stack(target_token_logits).T
                labels = data["is_harmful"].to("cpu")

                probs = torch.nn.functional.softmax(target_token_logits_transformed, dim=1)[:, 0]

                total_predictions.append(predictions)
                total_labels.append(labels)
                total_probs.append(probs)

        total_predictions = torch.cat(total_predictions)
        total_labels = torch.cat(total_labels)
        total_probs = torch.cat(total_probs)

        # calculate f1, precision, recall, accuracy, roc_auc across three batches of data
        f1_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []
        roc_auc_scores = []

        examples_per_step = len(total_predictions) // 3
        for i in range(3):
            start = i * examples_per_step
            end = (i + 1) * examples_per_step

            f1_scores.append(f1_score(total_labels[start:end], total_predictions[start:end]))
            precision_scores.append(precision_score(total_labels[start:end], total_predictions[start:end]))
            recall_scores.append(recall_score(total_labels[start:end], total_predictions[start:end]))
            accuracy_scores.append(accuracy_score(total_labels[start:end], total_predictions[start:end]))
            roc_auc_scores.append(roc_auc_score(total_labels[start:end], total_probs[start:end]))

        # we now log our results
        num_steps = self.state.global_step
        output_dir = os.path.join(self.args.output_dir, "across_training")
        os.makedirs(output_dir, exist_ok=True)

        out_fn = os.path.join(output_dir, f"evaluation_step_{num_steps}.jsonl")
        out_file = open(out_fn, "w")

        # print the results to output directory
        print_metrics = {
            "F1 Score": (np.mean(f1_scores), sem(f1_scores), f1_scores),
            "Precision": (np.mean(precision_scores), sem(precision_scores), precision_scores),
            "Recall": (np.mean(recall_scores), sem(recall_scores), recall_scores),
            "Accuracy": (np.mean(accuracy_scores), sem(accuracy_scores), accuracy_scores),
            "ROC AUC": (np.mean(roc_auc_scores), sem(roc_auc_scores), roc_auc_scores)
        }
        out_file.write("Balanced Dataset Metrics (Mean ± StdErr):\n")
        for metric, (mean, stderr, raw) in print_metrics.items():
            out_file.write(f"{metric}: {mean:.4f} ± {stderr:.4f} | Raw: {raw}\n")
        out_file.close()

        # report mean of all metrics
        metrics = {
            "F1 Score": print_metrics["F1 Score"][0],
            "Precision Score": print_metrics["Precision"][0],
            "Recall Score": print_metrics["Recall"][0],
            "Accuracy Score": print_metrics["Accuracy"][0],
            "ROC AUC Score": print_metrics["ROC AUC"][0],
        }

        return metrics

    def dynahate_prompt_sft_eval(self, eval_dataloader):
        total_predictions = []
        total_labels = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(eval_dataloader)):
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                loss_mask = data["loss_mask"]

                logits = obtain_logit(self.model, input_ids, attention_mask).to("cpu")

                # take the logit of the answer token (corresponding to the token with value of 1 in loss_mask
                loss_mask = loss_mask.to(logits.device)  # Move loss_mask to the same device as logits if necessary

                # Get the logits corresponding to the last non-padding token
                last_token = logits[loss_mask.bool()].view(logits.shape[0], -1)

                # check its accuracy on the next token
                true_token = self.tokenizer.encode(" Yes")[0]
                false_token = self.tokenizer.encode(" No")[0]

                predictions = last_token[:, true_token] > last_token[:, false_token]
                labels = data["is_harmful"].to("cpu")

                total_predictions.append(predictions)
                total_labels.append(labels)

        total_predictions = torch.cat(total_predictions)
        total_labels = torch.cat(total_labels)

        total_f1 = f1_score(total_labels, total_predictions)

        # report accuracy
        total_accuracy = (total_predictions == total_labels).sum().item() / len(total_labels)

        metrics = {"Total F1": total_f1,
                   "Total Accuracy": total_accuracy,
                   }

        return metrics

    def toxigen_prompt_sft_eval(self, eval_dataloader):
        total_predictions = []
        total_labels = []

        adversarial_predictions = []
        adversarial_labels = []

        nonadversarial_predictionsn = []
        nonadversarial_labels = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(eval_dataloader)):
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                loss_mask = data["loss_mask"]

                logits = obtain_logit(self.model, input_ids, attention_mask).to("cpu")

                # take the logit of the answer token (corresponding to the token with value of 1 in loss_mask
                loss_mask = loss_mask.to(logits.device)  # Move loss_mask to the same device as logits if necessary

                # Get the logits corresponding to the last non-padding token
                last_token = logits[loss_mask.bool()].view(logits.shape[0], -1)

                # check its accuracy on the next token
                true_token = self.tokenizer.encode(" Yes")[0]
                false_token = self.tokenizer.encode(" No")[0]

                predictions = last_token[:, true_token] > last_token[:, false_token]
                labels = data["is_harmful"].to("cpu")

                # we also select the adversarial evaluations seperately
                curr_adversarial_labels = data["is_adversarial"].to("cpu")

                total_predictions.append(predictions)
                total_labels.append(labels)

                adversarial_predictions.append(predictions[curr_adversarial_labels])
                adversarial_labels.append(labels[curr_adversarial_labels])

                nonadversarial_predictionsn.append(predictions[~curr_adversarial_labels])
                nonadversarial_labels.append(labels[~curr_adversarial_labels])

        total_predictions = torch.cat(total_predictions)
        total_labels = torch.cat(total_labels)

        adversarial_predictions = torch.cat(adversarial_predictions)
        adversarial_labels = torch.cat(adversarial_labels)

        nonadversarial_predictionsn = torch.cat(nonadversarial_predictionsn)
        nonadversarial_labels = torch.cat(nonadversarial_labels)

        total_f1 = f1_score(total_labels, total_predictions)
        adversarial_f1 = f1_score(adversarial_labels, adversarial_predictions)
        nonadversarial_f1 = f1_score(nonadversarial_labels, nonadversarial_predictionsn)

        # report accuracy
        total_accuracy = (total_predictions == total_labels).sum().item() / len(total_labels)
        adversarial_accuracy = (adversarial_predictions == adversarial_labels).sum().item() / len(adversarial_labels)
        nonadversarial_accuracy = (nonadversarial_predictionsn == nonadversarial_labels).sum().item() / len(nonadversarial_labels)
        # print(f"Total F1: {total_f1}")
        # print(f"Adversarial F1: {adversarial_f1}")

        metrics = {"Total F1": total_f1,
                   "Adv F1": adversarial_f1,
                   "Non-Adv F1": nonadversarial_f1,
                   "Total Accuracy": total_accuracy,
                   "Adv Accuracy": adversarial_accuracy,
                   "Non-Adv Accuracy": nonadversarial_accuracy
                   }

        return metrics

    def wildguard_prompt_eval(self, eval_dataloader):
        total_predictions = []
        total_labels = []

        adversarial_predictions = []
        adversarial_labels = []

        nonadversarial_predictionsn = []
        nonadversarial_labels = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(eval_dataloader)):

                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                loss_mask = data["loss_mask"]

                logits = obtain_logit(self.model, input_ids, attention_mask).to("cpu")

                # take the logit of the answer token (corresponding to the token with value of 1 in loss_mask
                loss_mask = loss_mask.to(logits.device)  # Move loss_mask to the same device as logits if necessary

                # Get the logits corresponding to the last non-padding token
                last_token = logits[loss_mask.bool()].view(logits.shape[0], -1)

                # check its accuracy on the next token
                true_token = self.tokenizer.encode(WILDGUARD_PROMPT_ONLY_LABELS[True])[0]
                false_token = self.tokenizer.encode(WILDGUARD_PROMPT_ONLY_LABELS[False])[0]

                predictions = last_token[:, true_token] > last_token[:, false_token]
                labels = data["is_harmful"].to("cpu")

                # we also select the adversarial evaluations seperately
                curr_adversarial_labels = data["is_adversarial"].to("cpu")

                total_predictions.append(predictions)
                total_labels.append(labels)

                adversarial_predictions.append(predictions[curr_adversarial_labels])
                adversarial_labels.append(labels[curr_adversarial_labels])

                nonadversarial_predictionsn.append(predictions[~curr_adversarial_labels])
                nonadversarial_labels.append(labels[~curr_adversarial_labels])

        total_predictions = torch.cat(total_predictions)
        total_labels = torch.cat(total_labels)

        adversarial_predictions = torch.cat(adversarial_predictions)
        adversarial_labels = torch.cat(adversarial_labels)

        nonadversarial_predictionsn = torch.cat(nonadversarial_predictionsn)
        nonadversarial_labels = torch.cat(nonadversarial_labels)

        total_f1 = f1_score(total_labels, total_predictions)
        adversarial_f1 = f1_score(adversarial_labels, adversarial_predictions)
        nonadversarial_f1 = f1_score(nonadversarial_labels, nonadversarial_predictionsn)

        total_accuracy = (total_predictions == total_labels).sum().item() / len(total_labels)
        adversarial_accuracy = (adversarial_predictions == adversarial_labels).sum().item() / len(adversarial_labels)
        nonadversarial_accuracy = (nonadversarial_predictionsn == nonadversarial_labels).sum().item() / len(nonadversarial_labels)

        # print(f"Total F1: {total_f1}")
        # print(f"Adversarial F1: {adversarial_f1}")

        metrics = {"Total F1": total_f1,
                    "Adv F1": adversarial_f1,
                   "Non-Adv F1": nonadversarial_f1,
                   "Total Accuracy": total_accuracy,
                   "Adv Accuracy": adversarial_accuracy,
                   "Non-Adv Accuracy": nonadversarial_accuracy
                   }

        return metrics


    # for an older version of toxigen evaluation (more complicated)
    # def toxigen_eval(self, eval_dataloader):
    #     id_predictions = []
    #     id_labels = []
    #
    #     ood_predictions = []
    #     ood_labels = []
    #
    #     with torch.no_grad():
    #         for i, data in tqdm(enumerate(eval_dataloader)):
    #
    #             input_ids = data["input_ids"]
    #             attention_mask = data["attention_mask"]
    #
    #             logits = obtain_logit(self.model, input_ids, attention_mask).to("cpu")
    #
    #             # take the logit of the last token
    #             last_token = logits[:, -1, :]
    #
    #             # check its accuracy on the next token
    #             true_token = self.tokenizer.encode(TOXIC_CLASSIFICATION_LABELS[True])[0]
    #             false_token = self.tokenizer.encode(TOXIC_CLASSIFICATION_LABELS[False])[0]
    #
    #             predictions = last_token[:, true_token] > last_token[:, false_token]
    #             labels = data["is_hate"].to("cpu")
    #
    #             # we also select the adversarial evaluations seperately
    #             curr_id_label = data["is_id"].to("cpu")
    #
    #             for i in range(len(curr_id_label)):
    #                 if curr_id_label[i]:
    #                     id_predictions.append(predictions[i].item())
    #                     id_labels.append(labels[i].item())
    #                 else:
    #                     ood_predictions.append(predictions[i].item())
    #                     ood_labels.append(labels[i].item())
    #
    #     id_predictions = torch.tensor(id_predictions)
    #     id_labels = torch.tensor(id_labels)
    #
    #
    #     ood_predictions = torch.tensor(ood_predictions)
    #     ood_labels = torch.tensor(ood_labels)
    #
    #     id_f1 = f1_score(id_labels, id_predictions)
    #     ood_f1 = f1_score(ood_labels, ood_predictions)
    #
    #     # we also get accuracy
    #     id_accuracy = (id_predictions == id_labels).sum().item() / len(id_labels)
    #     ood_accuracy = (ood_predictions == ood_labels).sum().item() / len(ood_labels)
    #     # print(f"Total F1: {total_f1}")
    #     # print(f"Adversarial F1: {adversarial_f1}")
    #
    #     metrics = {"ID F1": id_f1,
    #                 "OOD F1": ood_f1,
    #                 "ID Accuracy": id_accuracy,
    #                 "OOD Accuracy": ood_accuracy}
    #
    #     return metrics

    def paradetox_eval(self, eval_dataloader):
        # note that we want to save the generations for later as well
        num_steps = self.state.global_step
        output_dir = os.path.join(self.args.output_dir, "generations")
        os.makedirs(output_dir, exist_ok=True)

        # store final bleu scores in a file
        out_fn_bleu_fn = os.path.join(self.args.output_dir, "bleu_scores.json")
        out_file_bleu = open(out_fn_bleu_fn, "a")

        out_fn = os.path.join(output_dir, f"evaluation_step_{num_steps}.jsonl")
        out_file = open(out_fn, "w")

        total_bleu = []

        # we loop over the eval dataset
        with torch.no_grad():
            for i, data in tqdm(enumerate(eval_dataloader)):
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                golden_label = data["golden_label"]

                # we strip the extra paddings
                max_length = (input_ids != self.tokenizer.pad_token_id).sum(dim=1).max().item()
                input_ids = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]

                max_length_label = (golden_label != self.tokenizer.pad_token_id).sum(dim=1).max().item()
                golden_label = golden_label[:, -max_length_label:]

                generation = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
                new_generation = generation[:, max_length:]

                # we first write the generation into jsonl file
                for gen, label in zip(new_generation, golden_label):
                    gen = gen[gen != self.tokenizer.pad_token_id]
                    label = label[label != self.tokenizer.pad_token_id]
                    gen_words = self.tokenizer.decode(gen, skip_special_tokens=True)
                    label_words = self.tokenizer.decode(label, skip_special_tokens=True)
                    out_file.write(json.dumps({"generation": gen_words, "label": label_words}))
                    out_file.write("\n")

                    tokenized_gen_words = word_tokenize(gen_words)
                    tokenized_label_words = word_tokenize(label_words)

                    bleu_score = sentence_bleu([tokenized_label_words], tokenized_gen_words,
                                               smoothing_function=SmoothingFunction().method1)

                    # append to bleu arr
                    # bleu_score = sentence_bleu([gen_words], label_words)
                    total_bleu.append(bleu_score)
                # we then get the BLEU score

        out_file.close()

        average_bleu = sum(total_bleu) / len(total_bleu)

        out_file_bleu.write(json.dumps({"step": num_steps, "bleu": average_bleu}))
        out_file_bleu.write("\n")
        out_file_bleu.close()

        print(f"Average BLEU: {average_bleu}")
        return {"BLEU": average_bleu}

