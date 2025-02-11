from tqdm import tqdm

from src.modules.templates import *
from datasets import concatenate_datasets, Dataset


# note that we assume pretraining dataset has no loss mask
# def load_generator(path):
#     with open(path, "r") as f:
#         for line in f:
#             yield {"input_ids": json.loads(line)}
#
# gen_kwargs = {"path": configs.data.input_data_fn}

# dataset = read_dataset_to_hf(load_generator, gen_kwargs=gen_kwargs).shuffle(seed=configs.seed).select(range(configs.data.num_data_examples))

def select_binary_balanced_dataset(hf_dataset, binary_eval_func, seed, num_examples_per_class, num_proc=1):
    """ returns a set of examples that are balanced according to the eval_func"""
    false_dataset = hf_dataset.filter(lambda x: not binary_eval_func(x), num_proc=num_proc).shuffle(seed=seed).select(range(num_examples_per_class))
    true_dataset = hf_dataset.filter(binary_eval_func, num_proc=num_proc).shuffle(seed=seed).select(range(num_examples_per_class))
    return concatenate_datasets([false_dataset, true_dataset]).shuffle(seed=seed)

def select_n_ary_balanced_dataset(hf_dataset, n_ary_eval_func, seed, num_examples_per_class):
    """ returns a set of examples that are balanced according to each class. n_ary_eval_func is a list of boolean functions"""
    datasets = []
    for i in range(len(n_ary_eval_func)):
        dataset = hf_dataset.filter(lambda x: n_ary_eval_func[i](x)).shuffle(seed=seed).select(range(num_examples_per_class))
        datasets.append(dataset)

    return concatenate_datasets(datasets).shuffle(seed=seed)

def partition_dataset(hf_dataset, partitions):
    """
    partitions the dataset into the given splits
    :param hf_dataset: the dataset
    :param partitions: a dictionary of the form {"split_name": num_examples}
    """
    if (sum(partitions.values()) > len(hf_dataset)):
        raise Exception(f"sum of partitions {sum(partitions.values())} is greater than dataset size {len(hf_dataset)}")
    dataset_splits = {}
    current_total = 0
    for key, value in partitions.items():
        dataset_splits[key] = hf_dataset.select(range(current_total, current_total + value))
        current_total += value
    return dataset_splits


# performs concatenation of each line in dataset similar to pretraining
def format_to_pretraining(hf_dataset, tokenizer, max_seq_len):
    """
    Assumes the dataset is already tokenized. Assumes there is an "input_ids" field. Will give [1] to eos_token_id for attention_mask and loss_mask
    :param hf_dataset: the dataset. Assumes "input_ids" to be in the dataset.
    :param tokenizer: the tokenizer for the eos token
    :param max_seq_len: each sequence is concatenated until this length
    :return:
    """
    assert(tokenizer.eos_token_id is not None)

    def format_generator(dataset, tqdm_object):
        current_dict = {k: [] for k in dataset.column_names}
        carry_over = {k: [] for k in dataset.column_names} # this is for storing the carry overs
        for example in dataset:
            tqdm_object.update(1)
            for k in dataset.column_names:
                current_dict[k] += example[k]
                if (k == "input_ids"):
                    current_dict[k] += [tokenizer.eos_token_id]
                elif (k == "loss_mask" or k == "attention_mask"):
                    current_dict[k] += [1]
                else:
                    raise Exception(f"Unknown column name {k}")

            while (len(current_dict["input_ids"]) >= max_seq_len):
                for k, v in current_dict.items():
                    if len(v) > max_seq_len:
                        carry_over[k] = v[max_seq_len:]
                        current_dict[k] = v[:max_seq_len]

                yield current_dict
                current_dict = {k: carry_over[k] for k in dataset.column_names}
                carry_over = {k: [] for k in dataset.column_names}

    tqdm_object = tqdm(total=len(hf_dataset))
    processed_dataset = [x for x in format_generator(hf_dataset, tqdm_object)]

    formatted_dataset = Dataset.from_list(processed_dataset)

    return formatted_dataset




# prepares the conversation. Adds a newline between components by default
def prepare_conversation(tokenizer, max_seq_len, text1, text1_role, text2, text2_role, pad_tokens=True):

    text1_prefix_tokens = tokenizer.encode(f"{text1_role.strip()}\n", add_special_tokens=False)
    text1_tokens = tokenizer.encode(f"{text1.strip()}\n", add_special_tokens=False)
    text2_prefix_tokens = tokenizer.encode(f"{text2_role.strip()}\n", add_special_tokens=False)
    text2_tokens = tokenizer.encode(f"{text2.strip()}\n", add_special_tokens=False)

    # shed off equally
    if len(text1_tokens) + len(text2_tokens) + len(text1_prefix_tokens) + len(
            text2_prefix_tokens) > max_seq_len:
        excess_len = len(text1_tokens) + len(text2_tokens) + len(text1_prefix_tokens) + len(
            text2_prefix_tokens) - max_seq_len + 1
        diff = len(text1_tokens) - len(text2_tokens)
        if (diff >= 0 and diff >= excess_len):
            # we shave off excess from text1_tokens only
            text1_tokens = text1_tokens[excess_len:]
        elif (diff < 0 and -1 * diff >= excess_len):
            # we shave off excess from text2_tokens only
            text2_tokens = text2_tokens[:-excess_len]
        elif (diff >= 0 and diff < excess_len):
            # first shave off from text1_tokens, the shave off equally
            text1_tokens = text1_tokens[diff:]
            excess_len = excess_len - diff
            text1_tokens = text1_tokens[excess_len // 2:]
            text2_tokens = text2_tokens[:-(excess_len // 2)]
        else:
            # first shave off from text2_tokens, then shave off equally
            diff = -1 * diff
            text2_tokens = text2_tokens[:-diff]
            excess_len = excess_len - diff
            text1_tokens = text1_tokens[excess_len // 2:]
            text2_tokens = text2_tokens[:-(excess_len // 2)]

    input_ids = text1_prefix_tokens + text1_tokens + text2_prefix_tokens + text2_tokens
    loss_mask = len(text1_prefix_tokens) * [0] + len(text1_tokens) * [0] + len(text2_prefix_tokens) * [
        1] + len(text2_tokens) * [1]
    attention_mask = [1] * len(input_ids)

    assert len(input_ids) == len(loss_mask)

    #This performs padding

    if pad_tokens and len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        loss_mask += [0] * pad_len
        attention_mask += [0] * pad_len

    return {"input_ids": input_ids, "loss_mask": loss_mask, "attention_mask": attention_mask}

def preprocess_conversation(hf_dataset, tokenizer, max_seq_len, seed, num_proc=1, use_loss_mask=True, pad_tokens=True):
    """assumes that each hf_dataset line contains:
    1. parent.text
    2. child.text
    """
    # we now begin the preprocessing
    def process_input(input):
        """assumes that each line in the dataset is a conversation containing "parent.text" and "child.text" """
        parent = input["parent"]
        first_prefix = f"User: "
        child = input["child"]
        second_prefix = f"Response: "

        return prepare_conversation(tokenizer, max_seq_len, parent["text"], first_prefix, child["text"], second_prefix, pad_tokens=pad_tokens)

    hf_dataset_tokenized = hf_dataset.map(process_input,
                                      batched=False,
                                      remove_columns=hf_dataset.column_names,
                                      num_proc=num_proc,
                                      load_from_cache_file=False)

    # we shuffle the dataset and return
    hf_dataset_tokenized = hf_dataset_tokenized.shuffle(seed=seed)

    #if we don't want to use loss mask, then we train on all tokens
    if (not use_loss_mask):
        def add_loss_to_all_except_pad(example):
            example["loss_mask"] = [1 if x != tokenizer.pad_token_id else 0 for x in example["input_ids"]]
            return example
        hf_dataset_tokenized = hf_dataset_tokenized.map(add_loss_to_all_except_pad, batched=False, num_proc=num_proc)

    return hf_dataset_tokenized



def reformat_dialogue_with_template(input, output, template_name, is_final_dialogue=False):
    """ This function formats a string given a template.
    If is_final_dialogue is true (actual query), we don't add the EOS token and remove a space if there is any at the end
    """
    if (template_name == "Llama-2-7b-chat-hf"):
        input = LLAMA_SYSTEM + input
        if is_final_dialogue:
            return LLAMA_BOS + LLAMA_CHAT_TEMPLATE.format(input=input.strip(), output=output.strip()).strip()
        return LLAMA_BOS + LLAMA_CHAT_TEMPLATE.format(input=input, output=output) + LLAMA_EOS
    if (template_name == "dynahate"):
        if is_final_dialogue:
            return HATE_CLASSIFICATION_WITH_LABEL.format(input=input, output=output).strip()
        return HATE_CLASSIFICATION_WITH_LABEL.format(input=input, output=output)
    if (template_name == "default"):
        if is_final_dialogue:
            return DEFAULT_TOXIC_TEMPLATE_WITH_LABEL.format(input=input, output=output).strip()
        return DEFAULT_TOXIC_TEMPLATE_WITH_LABEL.format(input=input, output=output)
    else:
        print("NOTE: USING DEFAULT TEMPLATE")
        if is_final_dialogue:
            return INPUT_TEMPLATE_WITH_OUTPUT.format(input=input, output=output).strip()
        return DEFAULT_TOXIC_TEMPLATE_WITH_LABEL.format(input=input, output=output)

def tokenize_input_output_pair(tokenizer, input, output):
    """
    Tokenizes the input and output pair with the correct spacing.
    :param tokenizer: a hf tokenizer
    :param input: str
    :param output: str
    :return: tokenized input and output
    """
    if "olmo" in tokenizer.name_or_path.lower():
        if input[-1] != " ":
            raise Exception("Input expected to end with a space, got " + input[-1] + " instead")
        if output[0] == " ":
            raise Exception("Output expected to NOT start with a space, got space instead")
        input = input[:-1]
        output = " " + output
        input_tokens = tokenizer.encode(input, add_special_tokens=False)
        output_tokens = tokenizer.encode(output, add_special_tokens=False)
        return input_tokens, output_tokens
    else:
        raise Exception(f"Tokenizer {tokenizer.name_or_path} not supported")



