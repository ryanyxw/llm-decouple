import os
from src.data.data_utils import setup_dataset_hf


def load_dolma(path_to_dataset, **kwargs):
    return setup_dataset_hf(path_to_dataset, **kwargs)