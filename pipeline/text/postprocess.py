import os
import json
import jsonlines
import re
from tqdm import tqdm
import logging
import argparse

parser = argparse.ArgumentParser(description="Postprocess cleaned OCR text.")
parser.add_argument("--base_dir", default="sample-cleaned/", help="Directory containing the cleaned files.")
parser.add_argument("--save_dir", default="sample-postprocessed/", help="Directory to save the postprocessed files.")
parser.add_argument("--original_dir", default="sample-original/", help="Directory containing the original files.")
args = parser.parse_args()

BASE_DIR = args.base_dir
SAVE_DIR = args.save_dir
ORIGINAL_DIR = args.original_dir

os.makedirs(SAVE_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(filename='cleaning_postprocess.log', level=logging.INFO)

# Patterns to remove
patterns_to_remove = [
    r'Digitized by Google',
    r'\(Digitized by Google\)',
    r'The Baldwin Library',
    r'(Printed on title)',
    r'Copyright Moloughlin\n',
    r'COPYRIGHT © ISS9 by D. Lothrop Company.',
    r'\n{5,}',
]

def clean_text(text, patterns, filename):
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove: \n numbers \n
    text = re.sub(r'\n\d+\n', '\n', text)
    # Remove filename text from the start of the text
    filename_pattern = re.sub(r'(\w)s\b', r'\1\'?s', filename, flags=re.IGNORECASE) # If the filename contains words that end with 's', check for an optional apostrophe before the 's'
    text = re.sub(r'^' + filename_pattern + r'[\n:. ]', '', text, flags=re.IGNORECASE)
    # Remove number from the start of the text
    text = re.sub(r'^\d+[.: \n]', '', text)
    # Remove digits at the end of the text, possibly followed by a period
    text = re.sub(r'\d+\.?$', '', text)
    # Remove "Page Number: <digits>" at the end of the text
    text = re.sub(r'\(?Page Number: \d+\)?$', '', text, flags=re.IGNORECASE)
    # Remove trailing whitespace characters
    text = text.strip()
    return text

for filename in tqdm(os.listdir(BASE_DIR), desc="Processing Files"):
    if filename.endswith(".jsonl"):
        # If file exists in save directory, skip
        if os.path.exists(os.path.join(SAVE_DIR, filename)):
            continue
        with jsonlines.open(os.path.join(BASE_DIR, filename)) as reader, \
             jsonlines.open(os.path.join(ORIGINAL_DIR, filename)) as original_reader, \
             jsonlines.open(os.path.join(SAVE_DIR, filename), mode='a') as writer:
            original_data = {obj["page_number"]: obj["text"] for obj in original_reader} 

            text_by_page = {}
            for obj in reader:
                page_number = obj["page_number"]
                text = obj.get("generated_text", "")
                
                # If text is unreadable or empty, skip the page.
                if "The OCR text provided is unreadable" in text or "The provided OCR text is empty." in text or "This text appears to be blank or incomplete. No cleaning was performed." in text or "Text omitted due to missing pages" in text:
                    continue

                # Remove entry if text is impossible to clean or understand 
                if "It is impossible to clean or understand text that only consists of the letters"in text or "There seems to be a mistake. This text contains only one word and does not make any sense in the context" in text or "It is not clear what the provided OCR text represents as it only contains seemingly random characters" in text:
                    continue
                
                # If the text starts with "Cleaned:" at the beginning, extract the text between that and "==="
                if text.startswith("Cleaned:\n     "):
                    start = len("Cleaned:\n     ")
                elif text.startswith("Cleaned:\n"):
                    start = len("Cleaned:\n")
                else:
                    start = 0
                end = text.find("===")
                text = text[start:end]
                
                # If the text does not need any cleaning
                if "This text does not contain any OCR errors that require cleaning." in text and page_number not in text_by_page:
                    try:
                        text = original_data[page_number]
                    except KeyError:
                        logging.info("Original text not found for book %s page number: %s", filename, page_number)
                        print("ERROR: Original text not found for book %s page number: %s" % (filename, page_number))
                        continue

                finish_reason = obj.get("finish_reason", "")
                if finish_reason != "stop":
                    logging.info("Check %s, page number: %s", filename[:-5], obj["page_number"])
                    logging.info("Finish Reason: %s \n", finish_reason)

                # Check: for ... or --- in the generated text
                if "..." in text or "---" in text:
                    logging.info("Check %s, page number: %s", filename[:-5], obj["page_number"])
                    logging.info("Generated Text: %s \n", text)

                text = clean_text(text.strip(), patterns_to_remove, filename[:-6])

                # Re-concatenate texts with tokens greater than 512
                if page_number not in text_by_page:
                    text_by_page[page_number] = text
                else:
                    text_by_page[page_number] += " " + text

            for page_number, text in text_by_page.items():
                writer.write({"page_number": page_number, "text": text})
