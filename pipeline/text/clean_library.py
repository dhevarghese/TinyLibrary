import mii
from pathlib import Path
import json
import logging
import argparse
import numpy as np
import random  
import torch
import os
from transformers import AutoTokenizer
import nltk
from nltk.corpus import words
import time
import glob
from tqdm import tqdm

# Create the parser
parser = argparse.ArgumentParser(description="Clean OCR detected text in library pages.")

# Add arguments
parser.add_argument("--library_dir", type=str, default="library-pages/", help="Directory containing book jsonl files.")
parser.add_argument("--clean_dir", type=str, default="cleaned-library/", help="Path to save the cleaned pages.")
parser.add_argument("--debug", action='store_true', help="If set, the debug mode will be enabled.")
parser.add_argument("--max_batch_size", type=int, default=5000, help="Maximum batch size for processing pages.")
parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model to use for cleaning pages.")
parser.add_argument("--system_message_file", type=str, default="system_message.txt", help="File containing the system message.")
parser.add_argument("--samples_file", type=str, default="samples.txt", help="File containing the samples.")
parser.add_argument("--log_file", type=str, default="cleaning.log", help="File name to save the logs.")

# Parse the arguments
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))

# Define a filter that only allows messages when local_rank is zero
class RankFilter(logging.Filter):
    def filter(self, record):
        return local_rank == 0

# set up logging
logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode="a")
logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

os.makedirs(args.clean_dir, exist_ok=True)
library_files = glob.glob(os.path.join(args.library_dir, "*.jsonl"))

if args.debug:
    library_files = library_files[:10]
    logger.info(f"Debug mode is on. Only processing 10 books.")

def construct_prompt(system_prompt_file, samples_file):
    # Open and read the system prompt and samples files
    with open(system_prompt_file, 'r') as file:
        system_prompt = file.read()
    with open(samples_file, 'r') as file:
        samples = file.read().split('===\n')

    prompt = []
    for i in range(0, len(samples), 2):
        # Add the user content
        if i == 0:
            prompt.append({
                "role": "user",
                "content": system_prompt + "\n\n" + samples[i] + '==='
            })
        else:
            prompt.append({
                "role": "user",
                "content": samples[i] + '==='
            })

        # Add the assistant content
        if i+1 < len(samples):
            prompt.append({
                "role": "assistant",
                "content": samples[i+1] + '==='
            })
    return prompt

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')

word_list = set(words.words())
def has_english_word(text):
    text_words = set(nltk.word_tokenize(text.lower()))
    return bool(text_words & word_list) # Returns True if there is atleast one english word in the text

def chunk_text(text, tokenizer, chunk_size=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= chunk_size:
        return [text]
    else:
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
        return chunks

PROMPT = construct_prompt(args.system_message_file, args.samples_file)
data = []
for file_index, book_file in enumerate(tqdm(library_files, desc="Cleaning books")):
    output_dir = Path(args.clean_dir)
    book_file_name = Path(book_file).stem
    output_file = output_dir / f"{book_file_name}.jsonl"
    if os.path.exists(output_file):
        logger.info(f"{output_file} already exists. Skipping...")
        continue

    logger.info(f"Reading data from {book_file}")
    file_data = read_jsonl(book_file)
    file_data = [datum for datum in file_data if has_english_word(datum["text"])] # Preprocess to check if there is atleast one identifiable english word
    logger.info(f"Total number of pages: {len(file_data)}")
    for datum in file_data:
        page_number = datum["page_number"]
        text = datum["text"]
        chunks = chunk_text(text, tokenizer) # Split the text into chunks
        for i, chunk in enumerate(chunks):
            new_datum = {"book_name": book_file_name, "page_number": f"{page_number}", "text": chunk}
            data.append(new_datum)
    
    # If len(data) has reached max batch size, then pass to pipe, else continue
    if len(data) < args.max_batch_size and file_index < len(library_files) - 1:
        continue    

    prompts = [[{**prompt, "content": prompt["content"].replace("<prompt_template>", datum["text"])} for prompt in PROMPT] for datum in data]
    # Deepspeed pipe calls encode of tokenizer; Need to apply chat template ourselves
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in prompts]
    file_keys = [[datum["book_name"], datum["page_number"]] for datum in data]
    logger.info(f"Prompts and file keys are ready to be used!")

    pipe = mii.pipeline(args.model_name)
    start_time = time.time()
    responses = pipe(prompts, max_new_tokens=1024, temperature=1.0) 
    time_taken = time.time() - start_time
    logger.info(f"Time taken for inference: {time_taken}")

    if pipe.is_rank_0:
        results = {}
        for file_key, response in zip(file_keys,responses):
            try: 
                book_name, page_number = file_key[0], file_key[1]
                generated_text = response.generated_text.split("<stop>")[0]
                prompt_length = response.prompt_length
                generated_length = response.generated_length
                finish_reason = response.finish_reason
                item = dict(page_number=page_number, generated_text=generated_text, prompt_length=prompt_length, generated_length=generated_length, finish_reason=finish_reason)
                if book_name not in results:
                    results[book_name] = []
                results[book_name].append(item)
            except Exception as e:
                logger.error(f"Error in processing page {page_number}")
                logger.error(f"Error: {e}")
                continue
        for book_name, book_results in results.items():
            output_file = os.path.join(args.clean_dir, f"{book_name}.jsonl")
            write_jsonl(output_file, book_results)
    pipe.destroy()
    data = []
