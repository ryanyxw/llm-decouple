# this file is responsible for reformatting the datasets into a unified format
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

