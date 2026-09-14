# This script converts the json files into the format required by LLaVA.

import jsonlines
import json
import logging
import argparse
import numpy as np
import random  
import os
from transformers import AutoTokenizer
import glob
from tqdm import tqdm
import re

# Create the parser
parser = argparse.ArgumentParser(description="Convert jsons into llava format.")
parser.add_argument("--dataset_dir", type=str, default="library_data/vqa", help="Directory containing jsonl files.")
parser.add_argument("--clean_dir", type=str, default="library_llava/", help="Path to save the converted json files.")
parser.add_argument("--patterns_file", type=str, default="patterns.txt", help="File containing patterns to be removed from text.")
parser.add_argument("--errors_file", type=str, default="err_patterns.txt", help="File containing error patterns to be skipped.")
parser.add_argument("--tokenizer_name", type=str, default="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B", help="Tokenizer to use to check the lengths of the conversations.")
parser.add_argument("--model_length", type=int, default=3072, help="Model length to fit the conversations to.")
parser.add_argument("--system_message_file", type=str, default="system_message.txt", help="File containing the system message.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for image-token placement.")
args = parser.parse_args()
random.seed(args.seed)

with open(args.patterns_file, 'rb') as f:
    patterns = [line.decode('unicode_escape').rstrip() for line in f]

if not args.errors_file:
    errors_responses = []
else:
    with open(args.errors_file, 'rb') as f:
        errors_responses = [line.decode('unicode_escape').rstrip() for line in f]

os.makedirs(args.clean_dir, exist_ok=True)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

def convert_to_chat_template(conversation):
    # This function converts the conversation from the format {'from': role, 'value': content} to the chat template format
    roles = {'human': 'user', 'gpt': 'assistant'}
    return [{'role': roles[turn['from']], 'content': turn['value']} for turn in conversation]

def fit_to_model_length(model_length, conversation):
    while True:
        updated_conversation = convert_to_chat_template(conversation)
        tokenized_conversation = tokenizer.apply_chat_template(updated_conversation)
        if len(tokenized_conversation) <= model_length:
            break
        drop_count = 3 if len(conversation) % 2 == 1 else 2
        conversation = conversation[:-drop_count]
    return conversation

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

            # Remove occurrences of "\n===\n\n" and convert occurrences of "\n===\n" to "\n"
            generated_text = generated_text.replace("\n===\n\n", " ").replace("\n===\n", "\n")

            # Split the generated text into a list of conversations
            conversations = []
            turns = re.split("(?i)Individual: |Assistant: |Child: |AI: ", generated_text)[1:]  # Ignore the first split as it will be empty
            
            for i in range(0, len(turns), 2):
                if i+1 >= len(turns):
                    continue

                child_turn = turns[i].strip().replace("\n===", "")
                ai_turn = turns[i+1].strip().replace("\n===", "").rstrip("===").rstrip("---").rstrip("\n")
                
                if i == 0 and "<image>" not in child_turn:
                    child_turn = random.choice([f"<image>\n{child_turn}", f"{child_turn}\n<image>"])
                
                # Skip turn if 'the description' is mentioned
                if 'the description' in child_turn.lower() or 'the description' in ai_turn.lower():
                    continue
                
                conversations.append({"from": "human", "value": child_turn})
                conversations.append({"from": "gpt", "value": ai_turn})

            if not conversations or conversations == []:
                continue

            conversations = fit_to_model_length(args.model_length, conversations)

            llava_obj = {
                "id": f"{book_name}_{page_number}",
                "image": f"{book_name}_{page_number}.jpg",
                "conversations": conversations
            }
            llava_data.append(llava_obj)

    # Write the llava data to a new json file
    with open(os.path.join(args.clean_dir, f"{os.path.splitext(file)[0]}.json"), 'w') as f:
        json.dump(llava_data, f, indent=4)
