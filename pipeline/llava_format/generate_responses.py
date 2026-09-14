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
from huggingface_hub import snapshot_download
import time
import glob
from tqdm import tqdm

# Create the parser
parser = argparse.ArgumentParser(description="Generate responses for the provided system message and prompt.")

# Add arguments
parser.add_argument("--library_file", type=str, default="library-pages.jsonl", help="Jsonl containing library text.")
parser.add_argument("--output_path", type=str, default="responses/", help = "Path to save the generated responses")
parser.add_argument("--output_file", type=str, default="caption_3_5", help = "File name to save the generated responses")
parser.add_argument("--debug", action='store_true', help="If set, the debug mode will be enabled.")
parser.add_argument("--max_batch_size", type=int, default=5000, help="Maximum batch size for processing.")
parser.add_argument(
    "--model_name",
    type=str,
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Hugging Face model used to generate the synthetic annotations.",
)
parser.add_argument(
    "--model_revision",
    type=str,
    default="e1945c40cd546c78e41f1151f4db032b271faeaa",
    help="Exact Hugging Face model revision to resolve before generation.",
)
parser.add_argument("--system_message_file", type=str, default="system_message.txt", help="File containing the system message.")
parser.add_argument("--samples_file", type=str, default="samples.txt", help="File containing the samples.")
parser.add_argument("--log_file", type=str, default="responses.log", help="File name to save the logs.")
parser.add_argument("--local_rank", type=int, default=0, help="Local rank of the process.")

# Read arguments from command line
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)
local_rank = int(os.getenv('LOCAL_RANK', '0'))

# Define a filter that only allows messages when local_rank is zero
class RankFilter(logging.Filter):
    def filter(self, record):
        return local_rank == 0

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

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


# Resolve the remote identifier once so Transformers and MII load the same
# immutable model snapshot. This revision is the repository state available
# during the original TinyLibrary annotation period.
model_path = snapshot_download(
    repo_id=args.model_name,
    revision=args.model_revision,
    ignore_patterns="original/*",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
PROMPT = construct_prompt(args.system_message_file, args.samples_file)

# set up logging
logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode="a")
logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())

data = []
file_data = read_jsonl(args.library_file)
logger.info(f"Total number of pages: {len(file_data)}")
if args.debug:
    file_data = file_data[:10]
    args.max_batch_size = 10
    logger.info(f"Debug mode is on. Only processing first 10 pages.")


for page_index, datum in enumerate(tqdm(file_data, desc="Generating responses")):
    data.append(datum)
    if len(data) < args.max_batch_size and page_index < len(file_data) - 1:
        continue    

    prompts = [[{**prompt, "content": prompt["content"].replace("<GENERATED CAPTION>", datum["caption"])} for prompt in PROMPT] for datum in data]
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in prompts]
    print(prompts[-1])

    file_keys = [[datum["book_name"], datum["page_number"]] for datum in data]
    logger.info(f"Prompts and file keys are ready to be used!")

    pipe = mii.pipeline(model_path)
    start_time = time.time()
    responses = pipe(prompts, max_new_tokens=2048, temperature=1.0) 
    time_taken = time.time() - start_time
    logger.info(f"Time taken for inference: {time_taken}")

    if pipe.is_rank_0:
        results = []
        for file_key, response in zip(file_keys,responses):
            try: 
                book_name, page_number = file_key[0], file_key[1]
                generated_text = response.generated_text.split("<stop>")[0]
                prompt_length = response.prompt_length
                generated_length = response.generated_length
                finish_reason = response.finish_reason
                item = dict(book_name=book_name, page_number=page_number, generated_text=generated_text, prompt_length=prompt_length, generated_length=generated_length, finish_reason=finish_reason)
                results.append(item)
            except Exception as e:
                logger.error(f"Error in processing page {page_number}")
                logger.error(f"Error: {e}")
                continue
        output_file = os.path.join(args.output_path, f"{args.output_file}.jsonl")
        write_jsonl(output_file, results)
    pipe.destroy()
    data = []
