# this file performs preprocessing functions on data
import json
import os
import shutil
import tempfile
import concurrent.futures
from tqdm import tqdm

def single_process(chunk_ind, chunk_start, chunk_end, temp_dir, data, process_func):
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

def process_with_multiprocessing(process_func, data, output_fn, error_fn = None, num_proc=1, buffer_size=1024*1024):
    """
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
            futures.append(executor.submit(single_process, i, chunk_start, chunk_end, temp_dir, data, process_func))
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
