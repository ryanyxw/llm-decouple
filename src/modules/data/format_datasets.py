# this file is responsible for reformatting the datasets into a unified format
import re

from datasets import concatenate_datasets, Dataset

from src.modules.data.format_utils import reformat_dialogue_with_template, select_binary_balanced_dataset, \
    partition_dataset, preprocess_conversation, tokenize_input_output_pair

from src.modules.data.load import read_dataset_to_hf
from src.modules.templates import *


def reformat_realtoxicity_prompts_for_inferencing(dataset):
    """
    Dataset({
        features: ['filename', 'begin', 'end', 'challenging', 'prompt', 'continuation'],
        num_rows: 1000
    })
    """

    def reformat_row(row):
        return {"prompt": row["prompt"]["text"]}

    return dataset.map(reformat_row, batched=False)

def reformat_google_civil_comments_for_inferencing(dataset, demonstration_dataset, label_threshold, template_name):
    """
    Dataset({
        features: ['text', 'toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'],
        num_rows: 1000
    })
    """

    demonstration_prefix = ""
    for example in demonstration_dataset:
        demonstration_prefix += reformat_dialogue_with_template(example["text"], CIVIL_COMMENTS_LABELS[example["toxicity"] >= label_threshold], "default", is_final_dialogue=False)
    def reformat_row(row):
        final_instruction = demonstration_prefix + reformat_dialogue_with_template(row["text"], "", "default", is_final_dialogue=True)
        return {"prompt": final_instruction,
                "label": row["toxicity"] >= label_threshold}
    return dataset.map(reformat_row, batched=False)

def reformat_unused_comments_for_inferencing(dataset, demonstration_dataset, binary_eval_func, template_name):
    """
     Dataset({
        features: ['utturance', 'tags'],
        num_rows: 7526086
    })
    """
    demonstration_prefix = ""

    for example in demonstration_dataset:
        raise Exception("bug: why are we using civil comments templte? ")
        demonstration_prefix += reformat_dialogue_with_template(example["utturance"]["text"], CIVIL_COMMENTS_LABELS[binary_eval_func(example)], template_name, is_final_dialogue=False)

    def reformat_row(row):
        final_instruction = demonstration_prefix + reformat_dialogue_with_template(row["utturance"]["text"], "", "default",
                                                                                   is_final_dialogue=True)
        return {"prompt": final_instruction,
                "label": binary_eval_func(row)}

    return dataset.map(reformat_row, batched=False)

def prepare_dynahate_lowdata_prompt_sft_dataset(tokenizer, seed, max_seq_len, num_proc=1, use_loss_mask=False):
    # we first read toxigen dataset from huggingface
    dataset = read_dataset_to_hf("/mnt/nfs1/ryan/decouple/data/dynahate/raw.csv")["train"].shuffle(seed=seed)

    # split dynahate according to pretrainer's guide (only use test set of rounds 3 and 4
    train_dataset = dataset.filter(lambda x: x["split"] == "train")
    test_dataset = dataset.filter(lambda x: x["split"] == "test" and ("3" in x["round"] or "4" in x["round"]))

    # low data means selecting the first 10k sequences from the 32k training data
    train_dataset = train_dataset.select(range(10000))

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        if add_label:
            label = " Yes" if row["label"] == "hate" else " No"
            final_instruction = prompt.format(input=row["text"].strip(), output=label)
        else:
            final_instruction = prompt.format(input=row["text"].strip(), output="")

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_harmful": row["label"] == "hate",
                    "skip": True}

        # we now calculate attention_mask and loss_mask for the input excluding padded tokens
        attention_mask = [int(i != tokenizer.pad_token_id) for i in input_ids]
        last_token_ind = sum(attention_mask) - 1
        loss_mask = [0] * last_token_ind + [1] + [0] * (len(input_ids) - last_token_ind - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_harmful": row["label"] == "hate",
                "skip": False}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True}, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False}, remove_columns=test_dataset.column_names)


    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset

def prepare_dynahate_prompt_sft_dataset(tokenizer, seed, max_seq_len, num_proc=1, use_loss_mask=False):
    # we first read toxigen dataset from huggingface
    dataset = read_dataset_to_hf("/mnt/nfs1/ryan/decouple/data/dynahate/raw.csv")["train"].shuffle(seed=seed)

    # split dynahate according to pretrainer's guide (only use test set of rounds 3 and 4
    # train on around 32k, text on 2010
    train_dataset = dataset.filter(lambda x: x["split"] == "train")
    test_dataset = dataset.filter(lambda x: x["split"] == "test" and ("3" in x["round"] or "4" in x["round"]))

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        if add_label:
            label = " Yes" if row["label"] == "hate" else " No"
            final_instruction = prompt.format(input=row["text"].strip(), output=label)
        else:
            final_instruction = prompt.format(input=row["text"].strip(), output="")

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_harmful": row["label"] == "hate",
                    "skip": True}

        # we now calculate attention_mask and loss_mask for the input excluding padded tokens
        attention_mask = [int(i != tokenizer.pad_token_id) for i in input_ids]
        last_token_ind = sum(attention_mask) - 1
        loss_mask = [0] * last_token_ind + [1] + [0] * (len(input_ids) - last_token_ind - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_harmful": row["label"] == "hate",
                "skip": False}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True}, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False}, remove_columns=test_dataset.column_names)


    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset


def prepare_NEW_civil_comments_finetune_auroc_dataset(tokenizer, seed, max_seq_len, num_proc=1, use_loss_mask=False):
    # we first read toxigen dataset from huggingface
    dataset = read_dataset_to_hf("google/civil_comments").shuffle(seed=seed)

    toxicity_threshold = 0.5

    # split dynahate according to pretrainer's guide (only use test set of rounds 3 and 4
    # train on around 32k, text on 2010
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        if add_label:
            label = " Yes" if row["toxicity"] >= toxicity_threshold else " No"
            final_instruction = prompt.format(input=row["text"].strip(), output=label)
        else:
            final_instruction = prompt.format(input=row["text"].strip(), output="")

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_harmful": row["label"] == "hate",
                    "skip": True}

        # we now calculate attention_mask and loss_mask for the input excluding padded tokens
        attention_mask = [int(i != tokenizer.pad_token_id) for i in input_ids]
        last_token_ind = sum(attention_mask) - 1
        loss_mask = [0] * last_token_ind + [1] + [0] * (len(input_ids) - last_token_ind - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_harmful": row["toxicity"] >= toxicity_threshold,
                "skip": False}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True}, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False}, remove_columns=test_dataset.column_names)


    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset

def prepare_toxigen_lowdata_prompt_sft_dataset(tokenizer, seed, max_seq_len, num_proc=1, use_loss_mask=False):
    # we first read toxigen dataset from huggingface
    dataset = read_dataset_to_hf("toxigen/toxigen-data", name="train")["train"].shuffle(seed=seed)

    # select 10k training examples and 2k test examples -> total dataset is 250k examples
    train_dataset = dataset.select(range(10000))
    test_dataset = dataset.select(range(len(dataset) - 2000, len(dataset)))

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        if add_label:
            label = " Yes" if row["prompt_label"] == 1 else " No"
            final_instruction = prompt.format(input=row["generation"].strip(), output=label)
        else:
            final_instruction = prompt.format(input=row["generation"].strip(), output="")
        is_prompt_adversarial = row["generation_method"] == 'ALICE'

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_adversarial": False,
                    "is_harmful": row["prompt_label"] == 1,
                    "skip": True}

        # we now calculate attention_mask and loss_mask for the input excluding padded tokens
        attention_mask = [int(i != tokenizer.pad_token_id) for i in input_ids]
        last_token_ind = sum(attention_mask) - 1
        loss_mask = [0] * last_token_ind + [1] + [0] * (len(input_ids) - last_token_ind - 1)

        assert sum(loss_mask) != 0

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_adversarial": is_prompt_adversarial,
                "is_harmful": row["prompt_label"] == 1,
                "skip": False}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True}, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False}, remove_columns=test_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # filter out adversarial examples from training dataset
    train_dataset = train_dataset.filter(lambda x: x["is_adversarial"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset

def prepare_toxigen_prompt_sft_dataset(tokenizer, seed, max_seq_len, num_proc=1, use_loss_mask=False):
    # we first read toxigen dataset from huggingface
    dataset = read_dataset_to_hf("toxigen/toxigen-data", name="train")["train"].shuffle(seed=seed)

    # select 20k training examples and 2k test examples -> total dataset is 250k examples
    train_dataset = dataset.select(range(25000))
    test_dataset = dataset.select(range(len(dataset) - 2000, len(dataset)))

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        if add_label:
            label = " Yes" if row["prompt_label"] == 1 else " No"
            final_instruction = prompt.format(input=row["generation"].strip(), output=label)
        else:
            final_instruction = prompt.format(input=row["generation"].strip(), output="")
        is_prompt_adversarial = row["generation_method"] == 'ALICE'

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_adversarial": False,
                    "is_harmful": row["prompt_label"] == 1,
                    "skip": True}

        # we now calculate attention_mask and loss_mask for the input excluding padded tokens
        attention_mask = [int(i != tokenizer.pad_token_id) for i in input_ids]
        last_token_ind = sum(attention_mask) - 1
        loss_mask = [0] * last_token_ind + [1] + [0] * (len(input_ids) - last_token_ind - 1)

        assert sum(loss_mask) != 0

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_adversarial": is_prompt_adversarial,
                "is_harmful": row["prompt_label"] == 1,
                "skip": False}

    prompt = TOXIC_CLASSIFICATION_WITH_PROMPT

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True}, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False}, remove_columns=test_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # filter out adversarial examples from training dataset
    train_dataset = train_dataset.filter(lambda x: x["is_adversarial"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset

def prepare_wildguard_lowdata_prompt_dataset(tokenizer, seed, max_seq_len, num_proc=1, use_loss_mask=False):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("allenai/wildguardmix", name="wildguardtrain")["train"].shuffle(seed=seed)
    test_dataset = read_dataset_to_hf("allenai/wildguardmix", name="wildguardtest")["test"].shuffle(seed=seed)

    # select 10k training examples and 2k test examples
    train_dataset = train_dataset.select(range(10000))

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(human_request=row["prompt"])
        is_prompt_harmful = row["prompt_harm_label"] == 'harmful'
        if (add_label):
            final_instruction += WILDGUARD_PROMPT_ONLY_LABELS[is_prompt_harmful]

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_adversarial": False,
                    "is_harmful": is_prompt_harmful,
                    "skip": True}

        # we now calculate attention_mask and loss_mask for the input excluding padded tokens
        attention_mask = [int(i != tokenizer.pad_token_id) for i in input_ids]
        last_token_ind = sum(attention_mask) - 1
        loss_mask = [0] * last_token_ind + [1] + [0] * (len(input_ids) - last_token_ind - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_adversarial": row["adversarial"],
                "is_harmful": is_prompt_harmful,
                "skip": False}

    prompt = WILDGUARD_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True}, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False}, remove_columns=test_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # filter out adversarial examples from training dataset
    train_dataset = train_dataset.filter(lambda x: x["is_adversarial"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset


def prepare_wildguard_prompt_dataset(tokenizer, seed, max_seq_len, num_proc=1, use_loss_mask=False):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("allenai/wildguardmix", name="wildguardtrain")["train"].shuffle(seed=seed)
    test_dataset = read_dataset_to_hf("allenai/wildguardmix", name="wildguardtest")["test"].shuffle(seed=seed)

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(human_request=row["prompt"])
        is_prompt_harmful = row["prompt_harm_label"] == 'harmful'
        if (add_label):
            final_instruction += WILDGUARD_PROMPT_ONLY_LABELS[is_prompt_harmful]

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_adversarial": False,
                    "is_harmful": is_prompt_harmful,
                    "skip": True}

        # we now calculate attention_mask and loss_mask for the input excluding padded tokens
        attention_mask = [int(i != tokenizer.pad_token_id) for i in input_ids]
        last_token_ind = sum(attention_mask) - 1
        loss_mask = [0] * last_token_ind + [1] + [0] * (len(input_ids) - last_token_ind - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_adversarial": row["adversarial"],
                "is_harmful": is_prompt_harmful,
                "skip": False}

    prompt = WILDGUARD_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True}, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False}, remove_columns=test_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # filter out adversarial examples from training dataset
    train_dataset = train_dataset.filter(lambda x: x["is_adversarial"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset



def prepare_wildguard_undesired_prompt_dataset(tokenizer, seed, max_seq_len, num_proc=1, use_loss_mask=False):
    """ this experiment first filters the wildguard dataset for desired cateogires, and then evaluates model capability at predicting if response is harmful or not"""

    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("allenai/wildguardmix", name="wildguardtrain")["train"].shuffle(seed=seed)
    test_dataset = read_dataset_to_hf("allenai/wildguardmix", name="wildguardtest")["test"].shuffle(seed=seed)

    # we first filter the dataset for the subcategories we want to use
    target_categories = ["social_stereotypes_and_unfair_discrimination", "toxic_language_hate_speech", "defamation_encouraging_unethical_or_unsafe_actions"]

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(human_request=row["prompt"])
        is_prompt_harmful = row["prompt_harm_label"] == 'harmful'
        if (add_label):
            final_instruction += WILDGUARD_PROMPT_ONLY_LABELS[is_prompt_harmful]

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "is_adversarial": False,
                    "is_harmful": is_prompt_harmful,
                    "skip": True}

        attention_mask = [1] * len(input_ids)
        loss_mask = [0] * (len(input_ids) - 1) + [1]

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "is_adversarial": row["adversarial"],
                "is_harmful": is_prompt_harmful,
                "skip": False}

    prompt = WILDGUARD_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True}, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=num_proc, fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False}, remove_columns=test_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    return train_dataset, test_dataset

def prepare_paradetox_dataset(tokenizer, seed, max_seq_len, num_proc):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("s-nlp/paradetox")["train"].shuffle(seed=seed)

    # create the dataset with 90-10 split between train and test
    test_dataset = train_dataset.select(range(int(0.1 * len(train_dataset))))
    train_dataset = train_dataset.select(range(len(test_dataset), len(train_dataset)))

    # set the tokenizer padding to be on the left side
    old_tokenizer_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(toxic_text=row["en_toxic_comment"])
        if (add_label):
            final_instruction += row["en_neutral_comment"] + tokenizer.eos_token

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "golden_label": row["en_neutral_comment"],
                    "skip": True}

        # add masks to the input
        if (add_label):
            question = tokenizer.encode(prompt.format(toxic_text=row["en_toxic_comment"]))
            answer = tokenizer.encode(row["en_neutral_comment"]) + [tokenizer.eos_token_id]
            temp_question = question + answer

            if temp_question != input_ids[:len(temp_question)]:
                import pdb
                pdb.set_trace()
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * (len(input_ids) - len(answer)) + [1] * len(answer)
        else:
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * len(input_ids)

        # we prepare the golden label, which must also be tokenized and padded
        golden_label = tokenizer.encode(row["en_neutral_comment"], padding="max_length", max_length=max_seq_len - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "golden_label": golden_label,
                "skip": False}

    prompt = PARADETOX_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True},
                                      remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=1,
                                    fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False},
                                    remove_columns=test_dataset.column_names)


    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    # set the tokenizer padding back to the original side
    tokenizer.padding_side = old_tokenizer_side

    return train_dataset, test_dataset


def prepare_paradetox_lowdata_dataset(tokenizer, seed, max_seq_len, num_proc):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("s-nlp/paradetox")["train"].shuffle(seed=seed)

    num_train = 3200
    # create the dataset with 90-10 split between train and test
    test_dataset = train_dataset.select(range(int(0.1 * len(train_dataset))))
    train_dataset = train_dataset.select(range(len(test_dataset), len(test_dataset) + num_train))

    # set the tokenizer padding to be on the left side
    old_tokenizer_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(toxic_text=row["en_toxic_comment"])
        if (add_label):
            final_instruction += row["en_neutral_comment"] + tokenizer.eos_token

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "golden_label": row["en_neutral_comment"],
                    "skip": True}

        # add masks to the input
        if (add_label):
            question = tokenizer.encode(prompt.format(toxic_text=row["en_toxic_comment"]))
            answer = tokenizer.encode(row["en_neutral_comment"]) + [tokenizer.eos_token_id]
            temp_question = question + answer

            if temp_question != input_ids[-len(temp_question):]:
                import pdb
                pdb.set_trace()
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * (len(input_ids) - len(answer)) + [1] * len(answer)
        else:
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * len(input_ids)

        #     attention_mask = [0] * (len(input_ids) - len(temp_question)) + [1] * len(temp_question)
        #     loss_mask = [0] * (len(input_ids) - len(answer)) + [1] * len(answer)
        # else:
        #     attention_mask = [0] * input_ids.count(tokenizer.pad_token_id) + [1] * (
        #                 len(input_ids) - input_ids.count(tokenizer.pad_token_id))
        #     loss_mask = [0] * len(input_ids)

        # we prepare the golden label, which must also be tokenized and padded
        golden_label = tokenizer.encode(row["en_neutral_comment"], padding="max_length", max_length=max_seq_len - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "golden_label": golden_label,
                "skip": False}

    prompt = PARADETOX_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True},
                                      remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=1,
                                    fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False},
                                    remove_columns=test_dataset.column_names)


    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    # set the tokenizer padding back to the original side
    tokenizer.padding_side = old_tokenizer_side

    return train_dataset, test_dataset


def prepare_paradetox_lowdata_ood_dataset(tokenizer, seed, max_seq_len, num_proc):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("s-nlp/paradetox")["train"].shuffle(seed=seed)

    def filter_for_test(row):
        list = ["dick", "bitches", "fucks", "nigga", "shits", "retarded", "dumbass", "shitty", "pussies", "lunatic", "crap", "moron", "bastard", "jackass"]
        for word in list:
            if word in row["en_toxic_comment"]:
                return True
        return False

    # create the dataset and filter out specific occurances of words
    test_dataset = train_dataset.filter(filter_for_test)
    train_dataset = train_dataset.filter(lambda row: not filter_for_test(row))

    # set the tokenizer padding to be on the left side
    old_tokenizer_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    num_train = 3200
    train_dataset = train_dataset.select(range(num_train))

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(toxic_text=row["en_toxic_comment"])
        if (add_label):
            final_instruction += row["en_neutral_comment"] + tokenizer.eos_token

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "golden_label": row["en_neutral_comment"],
                    "skip": True}

        # add masks to the input
        if (add_label):
            question = tokenizer.encode(prompt.format(toxic_text=row["en_toxic_comment"]))
            answer = tokenizer.encode(row["en_neutral_comment"]) + [tokenizer.eos_token_id]
            temp_question = question + answer

            if temp_question != input_ids[-len(temp_question):]:
                import pdb
                pdb.set_trace()
            attention_mask = [0] * (len(input_ids) - len(temp_question)) + [1] * len(temp_question)
            loss_mask = [0] * (len(input_ids) - len(answer)) + [1] * len(answer)
        else:
            attention_mask = [0] * input_ids.count(tokenizer.pad_token_id) + [1] * (
                        len(input_ids) - input_ids.count(tokenizer.pad_token_id))
            loss_mask = [0] * len(input_ids)

        # we prepare the golden label, which must also be tokenized and padded
        golden_label = tokenizer.encode(row["en_neutral_comment"], padding="max_length", max_length=max_seq_len - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "golden_label": golden_label,
                "skip": False}

    prompt = PARADETOX_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True},
                                      remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=1,
                                    fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False},
                                    remove_columns=test_dataset.column_names)


    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    # set the tokenizer padding back to the original side
    tokenizer.padding_side = old_tokenizer_side

    return train_dataset, test_dataset

def prepare_paradetox_ood_dataset(tokenizer, seed, max_seq_len, num_proc):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("s-nlp/paradetox")["train"].shuffle(seed=seed)

    def filter_for_test(row):
        list = ["dick", "bitches", "fucks", "nigga", "shits", "retarded", "dumbass", "shitty", "pussies", "lunatic", "crap", "moron", "bastard", "jackass"]
        for word in list:
            if word in row["en_toxic_comment"]:
                return True
        return False

    # create the dataset and filter out specific occurances of words
    test_dataset = train_dataset.filter(filter_for_test)
    train_dataset = train_dataset.filter(lambda row: not filter_for_test(row))

    # set the tokenizer padding to be on the left side
    old_tokenizer_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, add_label=False):
        final_instruction = prompt.format(toxic_text=row["en_toxic_comment"])
        if (add_label):
            final_instruction += row["en_neutral_comment"] + tokenizer.eos_token

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "golden_label": row["en_neutral_comment"],
                    "skip": True}

        # add masks to the input
        if (add_label):
            question = tokenizer.encode(prompt.format(toxic_text=row["en_toxic_comment"]))
            answer = tokenizer.encode(row["en_neutral_comment"]) + [tokenizer.eos_token_id]
            temp_question = question + answer

            if temp_question != input_ids[-len(temp_question):]:
                import pdb
                pdb.set_trace()
            attention_mask = [0] * (len(input_ids) - len(temp_question)) + [1] * len(temp_question)
            loss_mask = [0] * (len(input_ids) - len(answer)) + [1] * len(answer)
        else:
            attention_mask = [0] * input_ids.count(tokenizer.pad_token_id) + [1] * (
                        len(input_ids) - input_ids.count(tokenizer.pad_token_id))
            loss_mask = [0] * len(input_ids)

        # we prepare the golden label, which must also be tokenized and padded
        golden_label = tokenizer.encode(row["en_neutral_comment"], padding="max_length", max_length=max_seq_len - 1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "golden_label": golden_label,
                "skip": False}

    prompt = PARADETOX_PROMPT_ONLY_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": True},
                                      remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(reformat_row, batched=False, num_proc=1,
                                    fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "add_label": False},
                                    remove_columns=test_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
    test_dataset = test_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])
    test_dataset = test_dataset.remove_columns(["skip"])

    # set the tokenizer padding back to the original side
    tokenizer.padding_side = old_tokenizer_side

    return train_dataset, test_dataset


# def prepare_toxigen_prompt_dataset(tokenizer, seed, max_seq_len, num_proc):
#     # we first read toxigen dataset from huggingface
#     dataset = read_dataset_to_hf("toxigen/toxigen-data", name="prompts")
#
#     # we define in-distribubtion and out-of-distribution datasets
#     # for some reason "trans" does not have neutral category, so we omit it from our dataset
#     train_categories = ["black", "native_american", "immigrant", "lgbtq", "women", "chinese", "latino", "bisexual", "asian", "mental_disability", "jewish", "muslim"]
#
#     test_categories = ["mexican", "physical_disability", "middle_east"]
#
#     # create the training dataset
#     train_dataset_list = []
#     for train_category in train_categories:
#         category_neutral = dataset[f"neutral_{train_category}_1k"]
#         category_hate = dataset[f"hate_{train_category}_1k"]
#
#         found_neutral_sentences = []
#         found_hate_sentences = []
#
#         # we loop through examples and select the non-repeating ones
#         for i in range(len(category_neutral)):
#             text = category_neutral[i]["text"]
#             sentences = text.strip("\\n-\n").split("\\n")
#             for sentence in sentences:
#                 sentence = sentence.strip("- ")
#                 if sentence not in found_neutral_sentences:
#                     found_neutral_sentences.append(sentence)
#
#         for i in range(len(category_hate)):
#             text = category_hate[i]["text"]
#             sentences = text.strip("\\n-\n").split("\\n")
#             for sentence in sentences:
#                 sentence = sentence.strip("- ")
#                 if sentence not in found_hate_sentences:
#                     found_hate_sentences.append(sentence)
#
#         # we now create a new dataset with the non-repeating sentences
#         category_neutral = Dataset.from_dict({"text": found_neutral_sentences})
#         category_hate = Dataset.from_dict({"text": found_hate_sentences})
#
#         category_neutral = category_neutral.add_column("is_hate", [False] * len(category_neutral))
#         category_hate = category_hate.add_column("is_hate", [True] * len(category_hate))
#         category_neutral = category_neutral.add_column("category", [train_category] * len(category_neutral))
#         category_hate = category_hate.add_column("category", [train_category] * len(category_hate))
#
#         train_dataset_list.append(category_neutral)
#         train_dataset_list.append(category_hate)
#
#     test_dataset_list = []
#     for test_category in test_categories:
#         category_neutral = dataset[f"neutral_{test_category}_1k"]
#         category_hate = dataset[f"hate_{test_category}_1k"]
#
#         found_neutral_sentences = []
#         found_hate_sentences = []
#
#         # we loop through examples and select the non-repeating ones
#         for i in range(len(category_neutral)):
#             text = category_neutral[i]["text"]
#             sentences = text.strip("\\n-\n").split("\\n")
#             for sentence in sentences:
#                 sentence = sentence.strip("- ")
#                 if sentence not in found_neutral_sentences:
#                     found_neutral_sentences.append(sentence)
#
#         for i in range(len(category_hate)):
#             text = category_hate[i]["text"]
#             sentences = text.strip("\\n-\n").split("\\n")
#             for sentence in sentences:
#                 sentence = sentence.strip("- ")
#                 if sentence not in found_hate_sentences:
#                     found_hate_sentences.append(sentence)
#
#         # we now create a new dataset with the non-repeating sentences
#         category_neutral = Dataset.from_dict({"text": found_neutral_sentences})
#         category_hate = Dataset.from_dict({"text": found_hate_sentences})
#
#         category_neutral = category_neutral.add_column("is_hate", [False] * len(category_neutral))
#         category_hate = category_hate.add_column("is_hate", [True] * len(category_hate))
#         category_neutral = category_neutral.add_column("category", [test_category] * len(category_neutral))
#         category_hate = category_hate.add_column("category", [test_category] * len(category_hate))
#
#         test_dataset_list.append(category_neutral)
#         test_dataset_list.append(category_hate)
#
#     # we create our training and testing datasets
#     train_dataset = concatenate_datasets(train_dataset_list).shuffle(seed=seed)
#
#     test_dataset_id = train_dataset.select(range(int(0.2 * len(train_dataset))))
#     train_dataset = train_dataset.select(range(len(test_dataset_id), len(train_dataset)))
#     test_dataset_ood = concatenate_datasets(test_dataset_list).shuffle(seed=seed)
#
#     # reformat the dataset
#     def reformat_row(row, prompt, labels, tokenizer, add_label=False):
#         # we first strip the weird "- " from the beginnings and "\\n-\n" from the end of toxigen prompts
#         text = row["text"].strip("- ").strip("\\n-\n")
#
#         final_instruction = prompt.format(input=text, output="")
#         if (add_label):
#             final_instruction += labels[row["is_hate"]]
#
#         input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)
#
#         if len(input_ids) > tokenizer.model_max_length - 1:
#             return {"input_ids": [],
#                     "attention_mask": [],
#                     "loss_mask": [],
#                     "is_hate": False,
#                     "category": "none",
#                     "skip": True}
#
#         # add masks to the input
#         if (add_label):
#             attention_mask = [1] * len(input_ids)
#             loss_mask = [0] * (len(input_ids) - 1) + [1]
#         else:
#             attention_mask = [1] * len(input_ids)
#             loss_mask = [0] * len(input_ids)
#
#         return {"input_ids": input_ids,
#                 "attention_mask": attention_mask,
#                 "loss_mask": loss_mask,
#                 "is_hate": row["is_hate"],
#                 "category": row["category"],
#                 "skip": False}
#
#     prompt = TOXIC_CLASSIFICATION_WITH_PROMPT
#     labels = TOXIC_CLASSIFICATION_LABELS
#
#     train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
#                                       fn_kwargs={"prompt": prompt, "labels": labels, "tokenizer": tokenizer, "add_label": True},
#                                       remove_columns=train_dataset.column_names)
#     test_dataset_id = test_dataset_id.map(reformat_row, batched=False, num_proc=1,
#                                     fn_kwargs={"prompt": prompt, "labels": labels, "tokenizer": tokenizer, "add_label": False},
#                                     remove_columns=test_dataset_id.column_names)
#     test_dataset_ood = test_dataset_ood.map(reformat_row, batched=False, num_proc=1,
#                                     fn_kwargs={"prompt": prompt, "labels": labels, "tokenizer": tokenizer, "add_label": False},
#                                     remove_columns=test_dataset_ood.column_names)
#
#     # filter out examples that are too long
#     train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)
#     test_dataset_id = test_dataset_id.filter(lambda x: x["skip"] == False, num_proc=num_proc)
#     test_dataset_ood = test_dataset_ood.filter(lambda x: x["skip"] == False, num_proc=num_proc)
#
#     # drop unnecessary columns
#     train_dataset = train_dataset.remove_columns(["skip"])
#     test_dataset_id = test_dataset_id.remove_columns(["skip"])
#     test_dataset_ood = test_dataset_ood.remove_columns(["skip"])
#
#     # we now merge test dataset together with id or ood labeling
#     test_dataset_id = test_dataset_id.add_column("is_id", [True] * len(test_dataset_id))
#     test_dataset_ood = test_dataset_ood.add_column("is_id", [False] * len(test_dataset_ood))
#
#     test_dataset = concatenate_datasets([test_dataset_id, test_dataset_ood])
#
#     return train_dataset, test_dataset


def prepare_tofu_dataset(tokenizer, seed, max_seq_len, num_proc, apply_loss_mask):
    # we first read wildguard dataset from huggingface
    train_dataset = read_dataset_to_hf("locuslab/TOFU", name="retain_perturbed")["train"].shuffle(seed=seed)

    # create tofu_names as a copy of the original tofu names
    tofu_names = TOFU_NAMES.copy()
    tokenized_tofu_names = []

    # add the first and last names of the characters to the tofu names
    for name in TOFU_NAMES:
        tofu_names.append(name.split(" ")[0])
        tofu_names.append(name.split(" ")[-1])

    for name in tofu_names:
        tokenized_tofu_names.append(tokenizer.encode(name))
        tokenized_tofu_names.append(tokenizer.encode(" " + name))

    def filter_for_noname(row):
        # we filter out examples where the dataset doesn't contain the name of the individual
        for name in tofu_names:
            if name in row["question"] or name in row["answer"]:
                return True
        return False

    train_dataset = train_dataset.filter(filter_for_noname)

    # reformat the dataset
    def reformat_row(row, prompt, tokenizer, tokenized_tofu_names, apply_loss_mask):
        # final_instruction = prompt.format(question=row["question"], answer=row["answer"])
        final_instruction = row["answer"] # we only train on the answer

        input_ids = tokenizer.encode(final_instruction, padding="max_length", max_length=max_seq_len - 1)

        # input_ids = tokenizer.encode(final_instruction)

        stringified_input_ids = " ".join([str(x) for x in input_ids])

        # we select out all the sensitive name tokens
        for name_tokens in tokenized_tofu_names:
            stringified_name_tokens = " ".join([str(x) for x in name_tokens])
            while stringified_name_tokens in stringified_input_ids:
                # remove the name tokens from the input_ids
                occurence = stringified_input_ids.index(stringified_name_tokens)
                stringified_input_ids = stringified_input_ids[:occurence] + stringified_input_ids[occurence + len(stringified_name_tokens) + 1:]

        filtered_input_ids = [int(x) for x in stringified_input_ids.split(" ")]

        loss_mask = []
        # we now create the loss mask by checking if the token was filtered outo r not
        orig_input_ids_ptr = 0
        filtered_input_ids_ptr = 0
        while (orig_input_ids_ptr < len(input_ids)):
            if (input_ids[orig_input_ids_ptr] == filtered_input_ids[filtered_input_ids_ptr]):
                loss_mask += [1]
                orig_input_ids_ptr += 1
                filtered_input_ids_ptr += 1
            else:
                loss_mask += [0]
                orig_input_ids_ptr += 1

        # set attention mask to exclude pad token
        attention_mask = [1 if i != tokenizer.pad_token_id else 0 for i in input_ids]

        # [1] * len(input_ids))
        if not apply_loss_mask:
            loss_mask = [1] * len(input_ids)

        if len(input_ids) > tokenizer.model_max_length - 1:
            return {"input_ids": [],
                    "attention_mask": [],
                    "loss_mask": [],
                    "skip": True}

        # we prepare the golden label, which must also be tokenized and padded
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "skip": False}

    prompt = TOFU_TEMPLATE

    train_dataset = train_dataset.map(reformat_row, batched=False, num_proc=1,
                                      fn_kwargs={"prompt": prompt, "tokenizer": tokenizer, "tokenized_tofu_names": tokenized_tofu_names, "apply_loss_mask": apply_loss_mask},
                                      remove_columns=train_dataset.column_names)

    # filter out examples that are too long
    train_dataset = train_dataset.filter(lambda x: x["skip"] == False, num_proc=num_proc)

    # drop unnecessary columns
    train_dataset = train_dataset.remove_columns(["skip"])

    return train_dataset


def prepare_dataset_for_training(tokenizer, seed, num_proc, **kwargs):
    """Load and reformat a dataset for training
    params:
    dataset_name: str, the name of the dataset
    seed: int, seed for shuffling
    tokenizer: tokenizer, the tokenizer to use
    kwargs: dict, additional arguments"""

    exp_name = kwargs["exp_name"]
    max_seq_len = kwargs["max_seq_len"]

    if "wildguard_prompt" in exp_name:
        # we read the wildguard dataset from huggingface
        train_dataset, test_dataset = prepare_wildguard_prompt_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {exp_name: test_dataset} # to turn it into an eval dictionary for hf trainer
        return train_dataset, eval_dataset
    if "wildguard_lowdata_prompt" in exp_name:
        # we read the wildguard dataset from huggingface
        train_dataset, test_dataset = prepare_wildguard_lowdata_prompt_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {exp_name: test_dataset}
        return train_dataset, eval_dataset
    if exp_name == "paradetox":
        # we read paradetox dataset from huggingface
        train_dataset, test_dataset = prepare_paradetox_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {exp_name: test_dataset} # to turn it into an eval dictionary for hf trainer
        return train_dataset, eval_dataset
    if exp_name == "paradetox_ood":
        # we read paradetox dataset from huggingface
        train_dataset, test_dataset = prepare_paradetox_ood_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {exp_name: test_dataset} # to turn it into an eval dictionary for hf trainer
        return train_dataset, eval_dataset
    if exp_name == "paradetox_lowdata" or exp_name == "NEW_paradetox_lowdata":
        # we read paradetox dataset from huggingface
        train_dataset, test_dataset = prepare_paradetox_lowdata_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {exp_name: test_dataset}
        return train_dataset, eval_dataset
    if exp_name == "paradetox_lowdata_ood" or exp_name == "NEW_paradetox_lowdata_ood":
        train_dataset, test_dataset = prepare_paradetox_lowdata_ood_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {exp_name: test_dataset}
        return train_dataset, eval_dataset
    if exp_name == "toxigen_prompt_sft":
        train_dataset, test_dataset = prepare_toxigen_prompt_sft_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {exp_name: test_dataset}
        return train_dataset, eval_dataset
    if exp_name == "toxigen_lowdata_prompt_sft":
        train_dataset, test_dataset = prepare_toxigen_lowdata_prompt_sft_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset  = {exp_name: test_dataset}
        return train_dataset, eval_dataset
    if exp_name == "dynahate_prompt_sft":
        train_dataset, test_dataset = prepare_dynahate_prompt_sft_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {exp_name: test_dataset}
        return train_dataset, eval_dataset
    if exp_name == "dynahate_lowdata_prompt_sft":
        train_dataset, test_dataset = prepare_dynahate_lowdata_prompt_sft_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {exp_name: test_dataset}
        return train_dataset, eval_dataset
    if exp_name == "NEW_civilcomments_finetune_auroc":
        train_dataset, test_dataset = prepare_NEW_civil_comments_finetune_auroc_dataset(tokenizer, seed, max_seq_len, num_proc)
        eval_dataset = {exp_name: test_dataset}
        return train_dataset, eval_dataset
    # if exp_name == "toxigen_finetune_prompts":
    #     train_dataset, test_dataset = prepare_toxigen_prompt_dataset(tokenizer, seed, max_seq_len, num_proc)
    #     eval_dataset = {exp_name: test_dataset}
    #     return train_dataset, eval_dataset
    if "tofu" in exp_name:
        train_dataset = prepare_tofu_dataset(tokenizer, seed, max_seq_len, num_proc, kwargs["apply_loss_mask"])
        return train_dataset, {}
    else:
        raise ValueError(f"Unknown dataset: {exp_name}")

# NOTE: this function is depricated
def load_and_reformat_dataset(dataset_name, dataset_file, splits, seed, num_proc=1, tokenizer=None, max_seq_len=None, use_loss_mask=False, **kwargs):
    """Load and reformat a dataset. If training or evaluation dataset, we also do tokenization. Else we just load and reformat
    params:
    dataset_name: str, the name of the dataset
    dataset_file: str, the path to the dataset file
    splits: dict, a dictionary of splits
    seed: int, seed for shuffling
    tokenizer: tokenizer, the tokenizer to use
    kwargs: dict, additional arguments"""

    if (dataset_name == "real-toxicity-prompts"):
        # This is a generation dataset, so we select num_generate_examples examples without much reformatting
        if "generation" not in splits:
            raise Exception("real toxicity prompts currently only supports generation")

        generation_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        generation_dataset = generation_dataset.select(range(splits["generation"]))
        return {"generation": reformat_realtoxicity_prompts_for_inferencing(generation_dataset)}
    elif (dataset_name == "civil_comments"):
        # check if demonstrations and generation in split
        if "demonstration" not in splits:
            raise Exception("civil comments should have demonstrations")
        if "generation" not in splits:
            raise Exception("civil comments should have generation")

        generation_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        #first make a simple partition of the dataset to select demonstrations
        demonstration_dataset = generation_dataset.select(range(10000))
        query_dataset = generation_dataset.select(range(10000, len(generation_dataset)))

        #we use the "train" partition to select demonstrations
        demonstration_dataset = select_binary_balanced_dataset(demonstration_dataset,
                                                                 lambda x: x["toxicity"] >= kwargs["label_threshold"],
                                                                 seed, splits["demonstration"])


        generation_dataset = select_binary_balanced_dataset(query_dataset, lambda x: x["toxicity"] >= kwargs["label_threshold"], seed, splits["generation"] // 2)

        return {"generation": reformat_google_civil_comments_for_inferencing(generation_dataset, demonstration_dataset, kwargs["label_threshold"], kwargs["template_name"])}
    elif (dataset_name == "unused_data"):
        # check if demonstrations and generation in split
        if "demonstration" not in splits:
            raise Exception("civil comments should have demonstrations")
        if "generation" not in splits:
            raise Exception("civil comments should have generation")

        # A jsonl file where each entry has a "parent" and "child" key
        generation_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        #use a smaller sample of total datafile since it is too large
        demonstration_dataset = generation_dataset.select(range(10000))
        query_dataset = generation_dataset.select(range(10000, 50000))

        def binary_eval_func(row):
            return row["tags"]["attributes"]["toxic_conversations__jigsaw_hatespeech_document_v2____label__toxic"][0][-1] >= kwargs["label_threshold"]

        # we use the "train" partition to select demonstrations
        demonstration_dataset = select_binary_balanced_dataset(demonstration_dataset,
                                                               binary_eval_func,
                                                               seed, splits["demonstration"])
        generation_dataset = select_binary_balanced_dataset(query_dataset, binary_eval_func, seed, splits["generation"] // 2)

        return {"generation": reformat_unused_comments_for_inferencing(generation_dataset, demonstration_dataset, binary_eval_func, kwargs["template_name"])}
    elif (dataset_name == "reddit"):
        # check for train splits
        if "train" not in splits:
            raise Exception("dynahate should have train split")
        ### setup the data, tokenizer, and preprocessing
        raw_dataset = read_dataset_to_hf(dataset_file)["train"]
        preprocessed_dataset = preprocess_conversation(raw_dataset, tokenizer, max_seq_len, seed=seed, num_proc=num_proc, use_loss_mask=use_loss_mask)
        preprocessed_dataset = preprocessed_dataset.select(range(splits["train"]))
        return {"train": preprocessed_dataset}
    elif (dataset_name == "dynahate"):
        # check for train and eval splits
        if "train" not in splits:
            raise Exception("dynahate should have train split")
        if "eval" not in splits:
            raise Exception("dynahate should have eval split")

        raw_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        def reformat_row(row):
            prompt = HATE_CLASSIFICATION_WITHOUT_LABEL.format(input=row["text"])
            label = DYNAHATE_LABELS[row["label"] == "hate"]
            return {"prompt": prompt,
                    "label": label}

        preprocessed_dataset = raw_dataset.map(reformat_row, batched=False)


        train_dataset = preprocessed_dataset.filter(lambda x: x["split"] == "train", batched=False, num_proc=num_proc)

        # we only want to evaluate on rounds 3 and 4
        eval_dataset = preprocessed_dataset.filter(lambda x: x["split"] == "test" and x["round.base"] - 2 > 0, batched=False, num_proc=num_proc)

        # using -1 means using the entire dataset
        if splits["train"] > 0:
            train_dataset = train_dataset.select(range(splits["train"]))
        if splits["eval"] > 0:
            eval_dataset = eval_dataset.select(range(splits["eval"]))


        # performs padding and tokenization
        def perform_tokenization(example):
            prompt_tokenized, label_tokenized = tokenize_input_output_pair(tokenizer, example["prompt"], example["label"])
            current_len = len(prompt_tokenized) + len(label_tokenized)

            if current_len > max_seq_len:
                example["skip"] = True
                example["input_ids"] = []
                example["attention_mask"] = []
                example["loss_mask"] = []
                return example

            new_input_id = prompt_tokenized + label_tokenized + [tokenizer.eos_token_id] * (max_seq_len - current_len)
            new_attention_mask = [1] * current_len + [0] * (max_seq_len - current_len)
            new_loss_mask = [0] * len(prompt_tokenized) + [1] * len(label_tokenized) + [0] * (max_seq_len - current_len)
            try:
                assert (len(new_input_id) == len(new_attention_mask) == len(new_loss_mask) == max_seq_len)
            except:
                import pdb
                pdb.set_trace()
            example["input_ids"] = new_input_id
            example["attention_mask"] = new_attention_mask
            example["loss_mask"] = new_loss_mask
            example["skip"] = False

            return example

        train_dataset = train_dataset.map(perform_tokenization, remove_columns=train_dataset.column_names,
                                          num_proc=num_proc)
        train_dataset = train_dataset.filter(lambda x: x["skip"] == False)

        def tokenize_evaluation(example):
            prompt_tokenized, _ = tokenize_input_output_pair(tokenizer, example["prompt"], "something")

            if (len(prompt_tokenized) > max_seq_len):
                example["skip"] = True
                example["input_ids"] = []
                example["attention_mask"] = []
                example["final_label"] = example["label"]
                example["round_info"] = example["round.base"]
                return example

            example["input_ids"] = prompt_tokenized
            example["attention_mask"] = [1] * len(prompt_tokenized)
            example["final_label"] = example["label"] == DYNAHATE_LABELS[True]
            example["round_info"] = example["round.base"]
            example["skip"] = False
            return example


        eval_dataset = eval_dataset.map(tokenize_evaluation, remove_columns=eval_dataset.column_names,
                                        num_proc=num_proc)
        eval_dataset = eval_dataset.filter(lambda x: x["skip"] == False)

        return {"train": train_dataset, "eval": eval_dataset}
    elif (dataset_name == "custom_hf_dataset"):
        # check if generation in split
        if "generation" not in splits:
            raise Exception("civil comments should have generation")

        generation_dataset = read_dataset_to_hf(dataset_file)["train"].shuffle(seed=seed)

        if splits["generation"] > 0:
            generation_dataset = generation_dataset.select(range(splits["generation"]))

        return {"generation": generation_dataset}
    else:
        raise ValueError(f"Unknown dataset: {dataset_file}")

