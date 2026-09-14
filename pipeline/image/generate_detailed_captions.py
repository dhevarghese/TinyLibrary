# Description: This script generates captions for images in a shard using the CogVLM-2 model.

import os
import torch
import datetime
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import webdataset as wds
import jsonlines

try:
    from .resume import get_processed_keys, pending_items, split_key
except ImportError:  # Direct execution: python pipeline/image/generate_detailed_captions.py
    from resume import get_processed_keys, pending_items, split_key

parser = argparse.ArgumentParser(description="CogVLM-2 Image Descriptions.")
parser.add_argument("input_dir", help="path to shards directory")
parser.add_argument("output_dir", help="path to output directory")
parser.add_argument("first_shard", type=int, help="number of the first shard to process")
parser.add_argument("shard_count", type=int, help="number of shards to process")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--write_size", default=100, type=int, help="Number of batches to process before writing to file.") 
parser.add_argument("--host", default=None, type=str, help="Host to obtain shards from.")
parser.add_argument("--quant", action="store_true")
parser.add_argument("--model_name", default="THUDM/cogvlm2-llama3-chat-19B")
parser.add_argument("--query", default="USER: Walk me through this image in detail. ASSISTANT:")

def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        device_copy = item.to(tgt)
        return device_copy
    elif isinstance(item, list):
        return [recur_move_to(v, tgt, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item

def collate_fn(features, tokenizer) -> dict:
    images = [feature.pop("images") for feature in features]
    for feature in features:  # Remove "labels" from features
        feature.pop("labels", None)
    tokenizer.padding_side = "left"
    padded_features = tokenizer.pad(features)
    inputs = {**padded_features, "images": images}
    return inputs

class Worker():
    def __init__(self, args):
        self.args = args
        self.torch_type = torch.float16 if self.args.quant else torch.bfloat16
        self.device = f"cuda:{self.args.gpu}" if self.args.gpu is not None else "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        if self.args.quant:
            bnb_quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_type, 
                bnb_4bit_use_double_quant=True, 
            )
            print(f"Loading 4 bit quantized model...")
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    torch_dtype=self.torch_type,
                    # low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    quantization_config=bnb_quantization_config,
                    device_map=self.device if self.args.gpu is not None else "auto"
                )
                .eval()
            )
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    torch_dtype=self.torch_type,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                .to(self.device)
                .eval()
            )

    def do_shards(self):
        start_time_overall = datetime.datetime.now()
        for shard in range(self.args.first_shard,
                           self.args.first_shard + self.args.shard_count):
            self._do_shard(shard)
        print(f"{self.args.shard_count} Shards done in {datetime.datetime.now() - start_time_overall}")

    def _do_shard(self, shard):
        output_name = f"{self.args.output_dir}/library-{shard:06d}.jsonl"

        processed_keys = set()
        if os.path.exists(output_name):
            processed_keys = get_processed_keys(output_name)
            shard_count = self.get_shard_count(shard)
            if len(processed_keys) == shard_count:
                print(f"{output_name} already processed, skipping")
                return
            if len(processed_keys) > shard_count:
                raise ValueError(
                    f"{output_name} contains more unique records than shard {shard}"
                )
            print(f"{output_name} has {shard_count - len(processed_keys)} images left to process")
        
        url = self._shard_url(shard)
        if not self.args.host:
            if not os.path.exists(url):
                print(f"{url} doesn’t exist, skipping")
                return
        pil_dataset = wds.WebDataset(url).decode("pil", handler=wds.ignore_and_continue).to_tuple("__url__", "__key__", "jpg").batched(self.args.batch_size)
        results = []
        processed_count = 0
        start_time = datetime.datetime.now()
        for urls, keys, images in pil_dataset:
            # Resume by page identity rather than by output length. This remains
            # correct when a previous run used a different batch size.
            pending = pending_items(keys, images, processed_keys)
            if not pending:
                start_time = datetime.datetime.now()
                continue
            keys, images = zip(*pending)

            input_list = []
            for image in images:
                input_list.append(self.model.build_conversation_input_ids(
                    self.tokenizer, query=self.args.query, history=[], images=[image],
                    template_version='chat'))

            input_batch = collate_fn(input_list, self.tokenizer)
            input_batch = recur_move_to(input_batch, self.model.device, lambda x: isinstance(x, torch.Tensor))
            input_batch = recur_move_to(input_batch, self.torch_type, lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x))

            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            gen_kwargs = {"max_new_tokens": 2048, "pad_token_id": pad_token_id, "do_sample": True, "top_p": 0.4, "temperature": 0.8, "top_k": 1}
            with torch.no_grad():
                outputs = self.model.generate(**input_batch, **gen_kwargs)
                outputs = outputs[:, input_batch["input_ids"].shape[1]:]
                strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for key, string in zip(keys, strings):
                res = string.split("</s>")[0]
                book_name, page_number = split_key(key)
                item = dict(book_name=book_name, page_number=page_number, caption=res)
                results.append(item)
                processed_keys.add((book_name, page_number))

            processed_count += len(strings)
            t = datetime.datetime.now() - start_time
            print(f"Processed {processed_count} images in {t}, avg = {t/processed_count}")

            # Store after processing every write_size batches; to prevent loss of computed data
            if processed_count % self.args.write_size == 0:
                with jsonlines.open(output_name, mode='a') as writer:
                    writer.write_all(results)
                    results = []
            
        if results:
            with jsonlines.open(output_name, mode='a') as writer:
                writer.write_all(results)
        
    def get_shard_count(self, shard):
        url = self._shard_url(shard)
        dataset = wds.WebDataset(url)
        line_count = sum(1 for _ in dataset) # Count the number of lines in the shard
        return line_count

    def _shard_url(self, shard):
        path = f"{self.args.input_dir}/library-{shard:06d}.tar"
        return f"pipe:ssh {self.args.host} cat {path}" if self.args.host else path
    
def main():
    torch.cuda.empty_cache()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    worker = Worker(args)
    worker.do_shards()

if __name__ == "__main__":
    main()
