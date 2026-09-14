import os
import json
import jsonlines
import re
from tqdm import tqdm
import logging
import argparse

parser = argparse.ArgumentParser(description="Postprocess detailed captions.")
parser.add_argument("--base_dir", default="sample-cleaned/", help="Directory containing the cleaned files.")
parser.add_argument("--save_dir", default="sample-postprocessed/", help="Directory to save the postprocessed files.")
parser.add_argument("--log_file", type=str, default="captions_postprocess.log", help="File name to save the logs.")
args = parser.parse_args()

BASE_DIR = args.base_dir
SAVE_DIR = args.save_dir
os.makedirs(SAVE_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(filename=args.log_file, level=logging.INFO)

# Patterns to remove
patterns_to_remove = [
    r'^Certainly! ',
]

def clean_text(text, patterns):
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = text.strip()
    return text

for filename in tqdm(os.listdir(BASE_DIR), desc="Processing Files"):
    if filename.endswith(".jsonl"):
        # If file exists in save directory, skip
        if os.path.exists(os.path.join(SAVE_DIR, filename)):
            continue
        with jsonlines.open(os.path.join(BASE_DIR, filename)) as reader, \
             jsonlines.open(os.path.join(SAVE_DIR, filename), mode='a') as writer:
            for obj in reader:
                text = obj.get("caption", "")                
                text = clean_text(text, patterns_to_remove)
                writer.write({"book_name": obj["book_name"], "page_number": obj["page_number"], "caption": text})
