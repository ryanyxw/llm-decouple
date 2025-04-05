# import argparse
import json

from tqdm import tqdm

# import os

# from tqdm import tqdm
# from transformers import DefaultDataCollator, TrainingArguments
#
# from src.modules.data.data_utils import load_tokenizer
# from src.modules.data.format_datasets import load_and_reformat_dataset
# from src.modules.data.load import read_dataset_to_hf
# from src.modules.modeling.SelectiveLossTrainer import SelectiveLossTrainer
# from peft import get_peft_model, LoraConfig
# import torch
# from omegaconf import OmegaConf
#
# from src.modules.modeling.evaluate import evaluate_model_with_multiple_evaluators
# from src.modules.modeling.inference import run_inference, obtain_logit
# from src.modules.modeling.modeling_utils import setup_model, free_gpus
# from src.modules.modeling.models.modeling_olmo_custom import CustomOlmoForCausalLM
from src.modules.templates import TOFU_NAMES
# from src.modules.utils import confirm_with_user, load_config, prepare_folder, validate_inputs, prepare_wandb, \
#     save_config, execute_shell_command

from openai import OpenAI
client = OpenAI()

def main():
    print("yay!")

    print("executing command...")


    tofu_documents = [
        "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_masked_seed0/checkpoint-39/tofu_custom/generation_output_test.jsonl",
        "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_masked_seed1/checkpoint-39/tofu_custom/generation_output_test.jsonl",
        "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_masked_seed2/checkpoint-39/tofu_custom/generation_output_test.jsonl",
        "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_unlikelihood_seed0/checkpoint-39/tofu_custom/generation_output_test.jsonl",
        "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_unlikelihood_seed1/checkpoint-39/tofu_custom/generation_output_test.jsonl",
        "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_unlikelihood_seed2/checkpoint-39/tofu_custom/generation_output_test.jsonl",
        "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_vanilla_seed0/checkpoint-39/tofu_custom/generation_output_test.jsonl",
        "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_vanilla_seed1/checkpoint-39/tofu_custom/generation_output_test.jsonl",
        "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_vanilla_seed2/checkpoint-39/tofu_custom/generation_output_test.jsonl",
        "/home/ryan/decouple/models/tofu/olmo1B_orig/tofu_custom/generation_output_test.jsonl"
    ]

    for document in tofu_documents:
        out_file_fn = document.replace(".jsonl", "_eval.jsonl")
        out_file = open(out_file_fn, "w")
        print(f"running evaluation on {document}...")

        # loop through the document
        with open(document, "r") as f:
            data = f.readlines()
            for line in tqdm(data, total=len(data)):
                line = json.loads(line)

                # call openai to check correctness of the output
                ground_truth = line["label"].strip()
                completion = line["completion"].strip()
                question = line["prompt"].strip("Question: ").strip("\nAnswer:").strip()

                # we remove hallucinated other "questions" if there are any
                if "\nQuestion" in completion:
                    completion = completion.split("\nQuestion")[0]
                if "\nQ:" in completion:
                    completion = completion.split("\nQ:")[0]

                # check if the completion is correct
                OPENAI_CORRECTNESS_PROMPT = f"Given a question and the correct answer, you will assess whether the candidate response contains the information in the correct answer. If the candidate response begins to hallucinate a next question-answer pair, ignore them. If a response is incomplete, grade it based on the text provided. Respond with either correct or incorrect and explain why. \n\nQuestion: {question}\n\nCorrect answer: {ground_truth}\n\nCandidate response: {completion}\n\nAnswer:"
                OPENAI_PARTIAL_CORRECT_PROMPT = f"Given a question and the correct answer, you will assess whether the candidate response contains any relevant information that is present in the correct answer but not in the question itself. If the candidate response begins to hallucinate a next question-answer pair, ignore them. If a response is incomplete, grade it based on the text provided. Respond with either yes or no and explain why. \n\nQuestion: {question}\n\nCorrect answer: {ground_truth}\n\nCandidate response: {completion}\n\nAnswer:"

                correctness_prompt = OPENAI_CORRECTNESS_PROMPT.format(question=question, ground_truth=ground_truth, completion=completion)
                partial_correct_prompt = OPENAI_PARTIAL_CORRECT_PROMPT.format(question=question, ground_truth=ground_truth, completion=completion)

                correctness_response = client.responses.create(
                    model="gpt-4o",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": correctness_prompt
                                }
                            ]
                        }
                    ],
                    text={
                        "format": {
                            "type": "text"
                        }
                    },
                    reasoning={},
                    tools=[],
                    temperature=1,
                    max_output_tokens=2048,
                    top_p=1,
                    store=True
                )

                partial_correct_response = client.responses.create(
                    model="gpt-4o",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": partial_correct_prompt
                                }
                            ]
                        }
                    ],
                    text={
                        "format": {
                            "type": "text"
                        }
                    },
                    reasoning={},
                    tools=[],
                    temperature=1,
                    max_output_tokens=2048,
                    top_p=1,
                    store=True
                )

                # write the output to the file
                out_file.write(json.dumps({
                    "question": question,
                    "completion": completion,
                    "label": ground_truth,
                    "correctness_response": correctness_response.output[0].content[0].text,
                    "partial_correct_response": partial_correct_response.output[0].content[0].text,
                }) + "\n")

        # close the file
        out_file.close()

    print("yay!")

def get_scores():
    scored_documents = [
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_masked_seed0/checkpoint-39/tofu_custom/generation_output_test_eval.jsonl",
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_masked_seed1/checkpoint-39/tofu_custom/generation_output_test_eval.jsonl",
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_masked_seed2/checkpoint-39/tofu_custom/generation_output_test_eval.jsonl",
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_unlikelihood_seed0/checkpoint-39/tofu_custom/generation_output_test_eval.jsonl",
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_unlikelihood_seed1/checkpoint-39/tofu_custom/generation_output_test_eval.jsonl",
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_unlikelihood_seed2/checkpoint-39/tofu_custom/generation_output_test_eval.jsonl",
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_vanilla_seed0/checkpoint-39/tofu_custom/generation_output_test_eval.jsonl",
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_vanilla_seed1/checkpoint-39/tofu_custom/generation_output_test_eval.jsonl",
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_vanilla_seed2/checkpoint-39/tofu_custom/generation_output_test_eval.jsonl",
        "/home/ryan/decouple/models/tofu/olmo1B_orig/tofu_custom/generation_output_test_eval.jsonl"
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_masked/checkpoint-36/tofu_custom/generation_output_test_eval.jsonl",
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_unlikelihood/checkpoint-36/tofu_custom/generation_output_test_eval.jsonl",
        # "/home/ryan/decouple/models/tofu/olmo1B_orig/NEW_FIXED_HF_tofu_3epoch_vanilla/checkpoint-36/tofu_custom/generation_output_test_eval.jsonl"
    ]

    tofu_names = TOFU_NAMES
    tofu_names_processed = []
    for name in tofu_names:
        tofu_names_processed += name.lower().split(" ")

    correctness_answers = {False: ["Incorrect", "incorrect"], True: ["Correct", "correct"]}
    partial_correct_answers = {False: ["No", "no"], True: ["Yes", "yes"]}

    for document in scored_documents:
        with open(document, "r") as f:
            data = f.readlines()

            num_correct = 0
            num_partial_correct = 0
            num_entity_mention = 0
            num_documents = 0


            for line in data:
                line = json.loads(line)
                correctness_response = line["correctness_response"]
                partial_correct_response = line["partial_correct_response"]
                output=line["completion"]

                # check if the entity is mentioned
                entity_mentioned = False
                # we remove hallucinated other "questions" if there are any
                if "\nQuestion" in output:
                    output = output.split("\nQuestion")[0]
                if "\nQ:" in output:
                    output = output.split("\nQ:")[0]
                for name in tofu_names_processed:
                    if name in output.lower():
                        entity_mentioned = True
                        break

                found_answer_for_correctness = False
                for answer in correctness_answers[True]:
                    if correctness_response.startswith(answer):
                        num_correct += 1
                        found_answer_for_correctness = True
                        break
                if not found_answer_for_correctness:
                    for answer in correctness_answers[False]:
                        if correctness_response.startswith(answer):
                            found_answer_for_correctness = True
                            break
                if not found_answer_for_correctness:
                    print(f"correctness response not found: {correctness_response}")


                found_answer_for_partial_correctness = False
                for answer in partial_correct_answers[True]:
                    if partial_correct_response.startswith(answer):
                        num_partial_correct += 1
                        found_answer_for_partial_correctness = True
                        break
                if not found_answer_for_partial_correctness:
                    for answer in partial_correct_answers[False]:
                        if partial_correct_response.startswith(answer):
                            found_answer_for_partial_correctness = True
                            break

                if not found_answer_for_partial_correctness:
                    print(f"partial correctness response not found: {partial_correct_response}")

                if entity_mentioned:
                    num_entity_mention += 1

                num_documents += 1

        print(f"document: {document}")
        print(f"num_correct: {num_correct}")
        print(f"num_partial_correct: {num_partial_correct}")
        print(f"num_entity_mention: {num_entity_mention}")
        print(f"num_documents: {num_documents}")
        print(f"correctness accuracy: {num_correct/num_documents}")
        print(f"partial correctness accuracy: {num_partial_correct/num_documents}")
        print(f"entity mention accuracy: {num_entity_mention/num_documents}")


if __name__ == "__main__":
    # main()
    get_scores()