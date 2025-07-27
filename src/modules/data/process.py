# this file performs preprocessing functions on data
import json
import os
import shutil
import tempfile
import concurrent.futures

import numpy as np
from datasets import concatenate_datasets
from tqdm import tqdm


#blueprint:
# 1. multiprocess_map_reduce: there is one function that essentially creates multiple processes and executes some process function on each one of them
#       and then concatenates the results -> types of things to concatenate will be specified by inputs
#       -> applicable for hf datasets to jsonl
#       -> can also be applied to already sharded files by wrapping them in an object with unified indexing
# 2. there is one function that essentially creates multiple shards for an iterable dataset
# 3. corresponding to 2, there is a function that collects the shards and concatenates them (maybe taking in a function for ordering

def single_process_save_to_jsonl(chunk_ind, chunk_start, chunk_end, temp_dir, hf_dataset):
    """
    This function saves a dataset to a jsonl file
    """
    print("entered chunk")
    output_file_path = os.path.join(temp_dir, f"chunk_{chunk_ind}.jsonl")

    hf_dataset.to_json(output_file_path)
    print("completed chunk" + str(chunk_ind))

def single_process_save_to_np(chunk_ind, chunk_start, chunk_end, temp_dir, hf_tokenized_data, input_ids_file_path,
                              label_mask_file_path, max_seq_len, total_tokens):
    """
    This function saves a dataset to a pre-definted memmap file (used for OLMO pretraining data preparation)
    hf_tokenized_data is the already filtered dataset that starts at chunk_start and ends at chunk_end
    """
    print("entered chunk")

    input_ids_file = np.memmap(
        input_ids_file_path, dtype=np.uint16, mode="r+",
        shape=(total_tokens,)
    )
    label_mask_file = np.memmap(
        label_mask_file_path, dtype=np.uint8, mode="r+",
        shape=(total_tokens,)
    )

    offset = chunk_start * max_seq_len
    tqdm_counter = tqdm(total=chunk_end - chunk_start) if chunk_ind == 0 else None
    for ex in hf_tokenized_data:
        ex_len = len(ex["input_ids"])
        input_ids_file[offset: offset + ex_len] = ex["input_ids"]
        label_mask_file[offset: offset + ex_len] = ex["loss_mask"]
        offset += ex_len
        if chunk_ind == 0:
            tqdm_counter.update(1)

    input_ids_file.flush()
    label_mask_file.flush()
    print("completed chunk" + str(chunk_ind))

def multiprocess_hf_map(function, hf_dataset, num_proc, fn_kwargs):
    """
    this applies function onto a hf_dataset across num_proc processes.
    :param function: a function that will call .map to the hf_dataset, returns a hf dataset
    :param hf_dataset: a hf dataset
    :param num_proc: number of processes to use
    :return:
    """
    num_lines_per_process = len(hf_dataset) // num_proc + 1 if len(hf_dataset) >= num_proc else 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = []

        for i in range(num_proc):
            shard = hf_dataset.shard(num_proc, i)
            print(f"chunk {i}: job sent")
            futures.append(executor.submit(function, shard, fn_kwargs))
        # for i in range(num_proc):
        #     chunk_start = i * num_lines_per_process
        #     chunk_end = (i + 1) * num_lines_per_process if (i + 1) * num_lines_per_process < len(hf_dataset) else len(hf_dataset)
        #     print(f"chunk {i}: {chunk_start} to {chunk_end} job sent")
        #     if (chunk_start >= len(hf_dataset)):
        #         break
        #     select_dataset = hf_dataset.select(range(chunk_start, chunk_end))
        #     futures.append(executor.submit(function, select_dataset, fn_kwargs))

        mapped_dataset = []
        for future in futures:
            mapped_dataset.append(future.result())
            print("completed chunk")

    # concatenate the files
    return concatenate_datasets(mapped_dataset)


# Global function that contains retry logic
def process_func_with_retries(process_func, process_id, chunk_start, chunk_end, temp_dir, temp_data, max_retries=3, **fn_args):
    for attempt in range(max_retries):
        try:
            print(f"Processing chunk {process_id}: {chunk_start} to {chunk_end}")
            # Your actual processing logic goes here
            process_func(process_id, chunk_start, chunk_end, temp_dir, temp_data, **fn_args)
        except Exception as e:
            print(f"Error in process {process_id}, attempt {attempt + 1}: {e}")
            if attempt + 1 == max_retries:
                raise  # Raise exception if all retries fail


def multiprocess_map_reduce(process_func, data, output_dict, num_proc=1, fn_args={}, buffer_size=1024*1024, max_retries=3):
    """
    This function creates multiple processes that executes process_func, and then concatenates the results together
    :param process_func: a function that takes in a process_id as well as start_ind and end_ind, along with the filtered dataset
    :param data: an iterable object
    :param output_dict: a dict where the key is the output file name and value is the process output file name
    :param num_proc: the number of processes
    :param buffer_size: buffer size used for copying
    :return: none
    """
    # create the temp_dir in the current directory
    temp_dir = tempfile.mkdtemp()
    print(f"temp_dir: {temp_dir}")

    # we save the temp_dir to the output dir
    if len(output_dict) != 0:
        with open(os.path.join(os.path.dirname(list(output_dict.keys())[0]), "temp_dir.txt"), "w") as temp_dir_file:
            temp_dir_file.write(temp_dir)

    chunk_size = len(data) // num_proc + 1 if len(data) >= num_proc else 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        # Split into chunks and assign to processes
        from datasets import Dataset
        is_hf = isinstance(data, Dataset)
        futures = []
        for i in range(num_proc):
            chunk_start = i * chunk_size
            chunk_end = (i + 1) * chunk_size if (i + 1) * chunk_size < len(data) else len(data)
            print(f"chunk {i}: {chunk_start} to {chunk_end}")
            if is_hf:
                temp_data = data.select(range(chunk_start, chunk_end))
            else:
                temp_data = data[chunk_start:chunk_end]
            futures.append(executor.submit(process_func_with_retries, process_func, i, chunk_start, chunk_end, temp_dir, temp_data, max_retries, **fn_args))
        concurrent.futures.wait(futures)

    print("completed multiprocessing, beginning concatenating of files")

    # concatenate the files
    for output_fn, process_output_fn in output_dict.items():
        with open(output_fn, 'wb') as output_file:
            for i in tqdm(range(num_proc)):
                temp_file_path = os.path.join(temp_dir, process_output_fn.format(i=i))
                with open(temp_file_path, 'rb') as temp_file:
                    shutil.copyfileobj(temp_file, output_file, length=buffer_size)

    # remove the temp_dir
    shutil.rmtree(temp_dir)


def single_process_dataset_by_line(chunk_ind, chunk_start, chunk_end, temp_dir, data, process_func):
    print(f"started {chunk_ind}")
    tqdm_counter = tqdm(total=chunk_end - chunk_start) if chunk_ind == 0 else None
    #open temp chunk and temp error file simultaneously
    with open(f"{temp_dir}/chunk_{chunk_ind}.jsonl", "w") as file, open(f"{temp_dir}/error_{chunk_ind}.jsonl", "w") as error_file:
        for i in range(chunk_start, chunk_end):
            try:
                file.write(json.dumps({"out": process_func(data[i])}) + "\n")
                if chunk_ind == 0:
                    tqdm_counter.update(1)
            except json.JSONDecodeError as e:
                error_log = {"index": i, "error": "JSONDecodeError", "message": str(e)}
                print(error_log)
                error_file.write(json.dumps(error_log) + "\n")
            except IOError as e:
                error_log = {"index": i, "error": "IOError", "message": str(e)}
                print(error_log)
                error_file.write(json.dumps(error_log) + "\n")
            except (KeyError, IndexError) as e:
                error_log = {"index": i, "error": type(e).__name__, "message": str(e)}
                print(error_log)
                error_file.write(json.dumps(error_log) + "\n")
            except Exception as e:
                # Catch-all for other exceptions
                error_log = {"index": i, "error": "UnhandledException", "message": str(e)}
                print(error_log)
                error_file.write(json.dumps(error_log) + "\n")
    print(f"completed {chunk_ind}")

def single_process(chunk_ind, chunk_start, chunk_end, temp_dir, data, process_func):
    print(f"started {chunk_ind}")
    tqdm_counter = tqdm(total=chunk_end - chunk_start) if chunk_ind == 0 else None
    #open temp chunk and temp error file simultaneously
    with open(f"{temp_dir}/chunk_{chunk_ind}.jsonl", "w") as file, open(f"{temp_dir}/error_{chunk_ind}.jsonl", "w") as error_file:
        for i in range(chunk_start, chunk_end):
            try:
                file.write(json.dumps(process_func(data[i])) + "\n")
                if chunk_ind == 0:
                    tqdm_counter.update(1)
            except Exception as e:
                print(f"Error in process {chunk_ind}: {e}")
                error_file.write(json.dumps({"index": i, "error": str(e)}) + "\n")
    print(f"completed {chunk_ind}")

def process_with_multiprocessing(process_func, data, output_fn, error_fn = None, num_proc=1, buffer_size=1024*1024):
    """
    This function performs a processon an iterable object
    :param process_func: function that processes a single index
    :param data: an iterable object -> no subnesting
    :param output_fn: the output file name
    :param num_proc: number of processes to use
    :param buffer_size: buffer size for file writing
    :return: none
    """
    #create the temp_dir in the current directory
    temp_dir = tempfile.mkdtemp()
    print(f"temp_dir: {temp_dir}")

    chunk_size = len(data) // num_proc + 1 if len(data) >= num_proc else 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        # Split into chunks and assign to processes
        futures = []
        for i in range(num_proc):
            chunk_start = i * chunk_size
            chunk_end = (i + 1) * chunk_size if (i + 1) * chunk_size < len(data) else len(data)
            print(f"chunk {i}: {chunk_start} to {chunk_end}")
            futures.append(executor.submit(single_process_dataset_by_line, i, chunk_start, chunk_end, temp_dir, data, process_func))
        concurrent.futures.wait(futures)

    print("completed multiprocessing, beginning concatenating of files")

    # concatenate the files
    with open(output_fn, 'wb') as output_file:
        for i in tqdm(range(num_proc)):
            temp_file_path = os.path.join(temp_dir, f'chunk_{i}.jsonl')
            with open(temp_file_path, 'rb') as temp_file:
                shutil.copyfileobj(temp_file, output_file, length=buffer_size)

    # concatenate the error files
    if error_fn is not None:
        with open(error_fn, 'wb') as error_file:
            for i in tqdm(range(num_proc)):
                temp_error_file_path = os.path.join(temp_dir, f'error_{i}.jsonl')
                with open(temp_error_file_path, 'rb') as temp_error_file:
                    shutil.copyfileobj(temp_error_file, error_file, length=buffer_size)

    # remove the temp_dir
    shutil.rmtree(temp_dir)
