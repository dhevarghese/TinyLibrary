import os
import json
import jsonlines
import re
from tqdm import tqdm
import logging
import argparse

parser = argparse.ArgumentParser(description="Postprocess cleaned text.")
parser.add_argument("--base_dir", default="sample-voted/", help="Directory containing the age categorized files.")
parser.add_argument("--save_file", default="sample-voted-postprocessed", help="File to save the postprocessed data.")
args = parser.parse_args()

BASE_DIR = args.base_dir
SAVE_FILE = args.save_file if args.save_file.endswith(".jsonl") else args.save_file + ".jsonl"

# Set up logging
logging.basicConfig(filename='new4_age_voting_postprocess.log', level=logging.INFO)

def extract_age_group(text):
    # Regular expression pattern to match "Age group: <GROUP>", "Age group:\n      <GROUP>", or "Age group: \n     <GROUP>"
    pattern = r'^Age group: ?\n? *?(3-5|6-8|9-12\+?|(?<!9-)12\+)'
    # Find the first match of the pattern in the text
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        age_group = match.group(1)
        return '9-12' if age_group == '9-12+' else age_group
    else:
        return None

def age_group_key(age_group):
    # Define the order of age groups
    age_group_order = ['3-5', '6-8', '9-12', '12+']
    # Return a tuple where the first element is the count of the age group and the second element is the index of the age group in age_group_order
    return age_group_count[age_group], -age_group_order.index(age_group)

for filename in tqdm(os.listdir(BASE_DIR), desc="Processing Files"):
    if filename.endswith(".jsonl"):
        with jsonlines.open(os.path.join(BASE_DIR, filename)) as reader, jsonlines.open(SAVE_FILE, mode='a') as writer:
            logging.info("Book: %s", filename)
            age_group_count = {'3-5': 0, '6-8': 0, '9-12': 0, '12+': 0}
            page_count = 0
            for obj in reader:
                text = obj.get("age_category", "")
                age_group = extract_age_group(text)
                page_count += 1
                if age_group is None:
                    logging.info("No age group extracted for: page number: %s", obj["start_page"])
                    continue
                age_group_count[age_group] += 1                 
            
            # Get the majority vote
            logging.info("Age Group Count: %s", age_group_count)
            # if the sum of the age group counts is 0, log it
            if sum(age_group_count.values()) == 0:
                logging.info("No age group votes")
                continue
            if sum(age_group_count.values()) == 1:
                logging.info("Book age group decided by single vote. Number of pages in book %s", page_count)
            # If there are age group counts with the same maximum value, log it
            if list(age_group_count.values()).count(max(age_group_count.values())) > 1:
                logging.info("Multiple majority votes")

            majority_age_group = max(age_group_count, key=age_group_key)
            logging.info("Voted Majority: %s\n", majority_age_group)
            writer.write({"book_name": filename, "age-group": majority_age_group})
