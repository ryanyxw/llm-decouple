from datasets import load_dataset

def setup_dataset_hf(dataset_path, **kwargs):
    if (dataset_path.split(".")[-1] in ["jsonl", "json"]):
        return load_dataset("json", data_files=dataset_path)
    dataset = load_dataset(dataset_path, **kwargs)
    return dataset
