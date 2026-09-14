import jsonlines
import json
import logging
import argparse
import numpy as np
import random  
import os
import glob
from tqdm import tqdm
import re

# python convert_captions_to_llava.py --dataset_dir library_tasks_data/caption --clean_dir repo_llava_format/ --patterns_file caption_patterns.txt --errors_file err_patterns.txt --instructs_file unique_caption_instructs.txt

# Create the parser
parser = argparse.ArgumentParser(description="Convert caption jsons into llava format.")
parser.add_argument("--dataset_dir", type=str, default="library_data/", help="Directory containing jsonl files.")
parser.add_argument("--clean_dir", type=str, default="library_llava/", help="Path to save the converted json files.")
parser.add_argument("--patterns_file", type=str, default="patterns.txt", help="File containing patterns to be removed from text.")
parser.add_argument("--instructs_file", type=str, default="instructions.txt", help="File containing unique instructions to be used as human prompt.")
parser.add_argument("--errors_file", type=str, default="err_patterns.txt", help="File containing error patterns to be skipped.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for caption instructions.")
args = parser.parse_args()
random.seed(args.seed)

with open(args.patterns_file, 'rb') as f:
    patterns = [line.decode('unicode_escape').rstrip() for line in f]

if not args.errors_file:
    errors_responses = []
else:
    with open(args.errors_file, 'rb') as f:
        errors_responses = [line.decode('unicode_escape').rstrip() for line in f]

# Load the unique values text file and split by double newline
with open(args.instructs_file) as f:
    instructions = f.read().split('\n\n')

os.makedirs(args.clean_dir, exist_ok=True)

# Go through the jsonl files in the dataset folder
for file in os.listdir(args.dataset_dir):
    jsonl_file = os.path.join(args.dataset_dir, file)
    llava_data = []

    # Convert each file into the llava json format
    with jsonlines.open(jsonl_file) as reader:
        for obj in reader:
            book_name = obj["book_name"]
            page_number = obj["page_number"]
            generated_text = obj["generated_text"]

            if any(error_response in generated_text for error_response in errors_responses):
                continue

            # Remove all text after a particular pattern is detected
            for pattern in patterns:
                generated_text = re.split(re.escape(pattern), generated_text, flags=re.IGNORECASE)[0]

            # Remove occurrences of "\n===\n\n" and "\n===" and convert occurrences of "\n===\n" to "\n"
            generated_text = generated_text.replace("\n===","").replace("\n===\n\n", " ").replace("\n===\n", "\n").strip()

            # If there is still "===" in the generated text, remove text after the "===" and including "===".
            if "===" in generated_text:
                generated_text = generated_text.split("===")[0].strip()

            # Use regex to detect if "\nOr" is in generated_text and remove all text after and including "\nOr"
            generated_text = re.split("\n+Or", generated_text, flags=re.IGNORECASE)[0]
            generated_text = generated_text.strip().replace("\n===", "").rstrip("===").rstrip("---").rstrip("\n")

            # If generated_text is empty, skip
            if not generated_text or generated_text == "":
                continue

            # Make a conversation list where the first text is a randomly sampled instruction with "from": "human", and the second text is the generated_text with "from":"gpt"
            conversation = [
                {"from": "human", "value": random.choice(instructions)},
                {"from": "gpt", "value": generated_text}
            ]

            llava_obj = {
                "id": f"{book_name}_{page_number}",
                "image": f"{book_name}_{page_number}.jpg",
                "conversations": conversation
            }
            llava_data.append(llava_obj)

    # Write the llava data to a new json file
    with open(os.path.join(args.clean_dir, f"{os.path.splitext(file)[0]}.json"), 'w') as f:
        json.dump(llava_data, f, indent=4)
