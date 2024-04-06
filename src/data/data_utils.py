from functools import lru_cache

from datasets import load_dataset
from transformers import AutoTokenizer
import json


def setup_dataset_hf(dataset_path, **kwargs):
    if (dataset_path.split(".")[-1] in ["jsonl", "json"]):
        return load_dataset("json", data_files=dataset_path)
    dataset = load_dataset(dataset_path, **kwargs)
    return dataset

def load_tokenizer(path_to_tokenizer):
    return AutoTokenizer.from_pretrained(path_to_tokenizer)

def file_generator(file, process_func):
    """generator for reading a file line by line"""
    while True:
        line = file.readline()
        if not line:
            break
        yield process_func(line)


@lru_cache(maxsize=None)
def build_nxt(pattern: tuple) -> tuple:
    # The function is being cached. Use tuple to avoid the cache being tampered out of scope.
    nxt = [0]
    current = 1
    match_idx = 0

    while current < len(pattern):
        if pattern[match_idx] == pattern[current]:
            current += 1
            match_idx += 1
            nxt.append(match_idx)
        elif match_idx != 0:
            match_idx = nxt[match_idx - 1]
        else:
            nxt.append(0)
            current += 1

    return tuple(nxt)


def kmp(seq, pattern, first_appearance=False):
    """
    Search for the location of a subsequence in a list. Not sure if there is a python built-in
    implementation of kmp somewhere...
    """
    nxt = build_nxt(tuple(pattern))
    current = 0
    match_idx = 0

    matched = []

    while current < len(seq):
        if seq[current] == pattern[match_idx]:
            current += 1
            match_idx += 1
        elif match_idx != 0:
            match_idx = nxt[match_idx - 1]
        else:
            current += 1

        if match_idx == len(pattern):
            matched.append(current - len(pattern))
            if first_appearance:
                return matched
            match_idx = nxt[match_idx - 1]

    return matched
