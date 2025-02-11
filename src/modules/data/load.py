# sourced from https://github.com/Watchful1/PushshiftDumps/blob/master/scripts/single_file.py
import concurrent.futures
import os

import pandas as pd
import zstandard as zstd
from tqdm import tqdm

from datasets import load_dataset, Dataset
import types

from src.modules.data.process import single_process_save_to_jsonl, multiprocess_map_reduce

def read_lines_zst(file_name):
	"""Reads lines from a Zstandard compressed file."""
	with open(file_name, 'rb') as file_handle:
		reader = zstd.ZstdDecompressor(max_window_size=(2**29) * 2).stream_reader(file_handle)
		buffer = ""
		while True:
			chunk = reader.read(2**27)
			if not chunk:
				break

			buffer += chunk.decode(errors='ignore')  # More efficient concatenation
			lines = buffer.split("\n")

			# Yield all complete lines except the last one (which may be incomplete)
			for line in lines[:-1]:
				yield line

			# Store the remaining part for the next iteration
			buffer = lines[-1]

		reader.close()
	# def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
	# 	chunk = reader.read(chunk_size)
	# 	bytes_read += chunk_size
	# 	if previous_chunk is not None:
	# 		chunk = previous_chunk + chunk
	# 	try:
	# 		return chunk.decode()
	# 	except UnicodeDecodeError:
	# 		if bytes_read > max_window_size:
	# 			raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
	# 		return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)
	#
	# with open(file_name, 'rb') as file_handle:
	# 	buffer = ''
	# 	reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
	# 	while True:
	# 		chunk = read_and_decode(reader, 2**27, (2**29) * 2)
	#
	# 		if not chunk:
	# 			break
	# 		lines = (buffer + chunk).split("\n")
	#
	# 		for line in lines[:-1]:
	# 			yield line, file_handle.tell()
	#
	# 		buffer = lines[-1]
	#
	# 	reader.close()

def read_lines_from_file(file_name, process_func=None):
	"""a generator that opens a file and reads each line with preprocessing"""
	if process_func is None:
		process_func = lambda x: x

	for line in open(file_name, "r"):
		yield process_func(line)

def read_dataset_to_hf(dataset_path, **kwargs):
	#if the dataset_path is a generator function, load using generator
	if callable(dataset_path):
		return Dataset.from_generator(dataset_path, **kwargs)
	#otherwise, load using the appropriate method
	if (dataset_path.endswith("json.gz") or dataset_path.endswith("jsonl.gz")):
		df = pd.read_json(dataset_path, lines=True)
		return Dataset.from_pandas(df)
	if (dataset_path.split(".")[-1] in ["jsonl", "json"]) or (dataset_path.split(".")[-2] in ["jsonl", "json"] if len(dataset_path.split(".")) > 1 else False):
		return load_dataset("json", data_files=dataset_path, **kwargs)
	if (dataset_path.split(".")[-1] == "csv"):
		return load_dataset("csv", data_files=dataset_path, **kwargs)
	if (dataset_path.split(".")[-1] == "tsv"):
		return load_dataset("csv", data_files=dataset_path, delimiter="\t", **kwargs)
	dataset = load_dataset(dataset_path, **kwargs)
	return dataset

def save_hf_to_jsonl(dataset, file_path, num_proc=1):
	""" saves a dataset to a jsonl file using parallelism"""
	output_dict = {file_path: "chunk_{i}.jsonl"}

	multiprocess_map_reduce(single_process_save_to_jsonl,
							dataset,
							output_dict,
							num_proc=num_proc)

def single_process_save_to_disk(dataset, out_fn):
	"""saves a dataset to disk"""
	dataset.save_to_disk(out_fn)

def save_dataset_to_shards(dataset, file_path, num_shards, special_name, num_proc=1):
	"""saves a hf dataset to shards using parallelism"""

	# make the directory if it doesn't exist
	os.makedirs(file_path, exist_ok=True)

	# open num_proc processes and save the dataset to shards
	futures = []
	with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
		for i in range(num_shards):
			output_fn = os.path.join(file_path, f"shard_{special_name}_{i}.arrow")
			shard = dataset.shard(num_shards, i)
			futures.append(executor.submit(single_process_save_to_disk, shard, output_fn))

		for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
			pass


