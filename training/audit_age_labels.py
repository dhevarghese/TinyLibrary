"""Label-stability audit: reclassify book age bands with an out-of-family LLM.

Mirrors the original Mistral-7B protocol (pipeline/age/categorize_library.py):
same system message, same few-shot samples file, same 5000-token page
chunking, majority vote over chunks per book. Differences (documented in the
paper): out-of-family classifier, greedy decoding (original sampled at T=1),
max_new_tokens=16 (label only).

    python audit_age_labels.py --library_dir .../postprocessed-children-library \
        --prompt_dir .../prompts/age-group --labels new4-age-category-library.jsonl \
        --out audit_qwen.jsonl
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--library_dir", required=True)
parser.add_argument("--prompt_dir", required=True)
parser.add_argument("--labels", required=True, help="new4-age-category-library.jsonl")
parser.add_argument("--out", required=True)
parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
parser.add_argument("--max_context_length", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

BANDS = ["3-5", "6-8", "9-12", "12+"]
LABEL_RE = re.compile(r"12\+|9-12|6-8|3-5|9\+")

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16).to(device).eval()


def construct_prompt(prompt_dir):
    # identical to pipeline/age/categorize_library.py
    system_prompt = open(os.path.join(prompt_dir, "system_message.txt")).read()
    samples = open(os.path.join(prompt_dir, "samples.txt")).read().split("===\n")
    prompt = []
    for i in range(0, len(samples), 2):
        content = samples[i] + "==="
        if i == 0:
            content = system_prompt + "\n\n" + content
        prompt.append({"role": "user", "content": content})
        if i + 1 < len(samples):
            prompt.append({"role": "assistant", "content": samples[i + 1] + "==="})
    return prompt


PROMPT = construct_prompt(args.prompt_dir)

# --- chunk books exactly like the original script ---
chunks = []  # (book_name, text)
for book_file in sorted(glob.glob(os.path.join(args.library_dir, "*.jsonl"))):
    name = os.path.splitext(os.path.basename(book_file))[0]
    pages = [json.loads(l) for l in open(book_file)]
    tokens_so_far, joined = 0, ""
    for datum in pages:
        tokens_so_far += len(tokenizer.tokenize(datum["text"]))
        joined += datum["text"]
        if tokens_so_far > args.max_context_length:
            chunks.append((name, joined))
            tokens_so_far, joined = 0, ""
    if tokens_so_far > 0:
        chunks.append((name, joined))
print(f"{len(chunks)} chunks from {len(set(c[0] for c in chunks))} books")


def build_text(chunk_text):
    msgs = [{**p, "content": p["content"].replace("<prompt_template>", chunk_text)}
            for p in PROMPT]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


votes = defaultdict(list)
order = sorted(range(len(chunks)), key=lambda i: len(chunks[i][1]))
with torch.no_grad():
    for s in tqdm(range(0, len(order), args.batch_size)):
        batch = [chunks[i] for i in order[s:s + args.batch_size]]
        texts = [build_text(t) for _, t in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=7000).to(device)
        out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        for (name, _), g in zip(batch, gen):
            m = LABEL_RE.search(g)
            votes[name].append(m.group(0) if m else "unparsed")

# --- majority vote + compare with the Mistral labels ---
ref = {}
for l in open(args.labels):
    r = json.loads(l)
    ref[os.path.splitext(r["book_name"])[0]] = r["age-group"]

rows, agree, adjacent, total = [], 0, 0, 0
conf = Counter()
for name, vs in votes.items():
    pred = Counter(vs).most_common(1)[0][0]
    gold = ref.get(name)
    rows.append({"book": name, "pred": pred, "gold": gold, "votes": vs})
    if gold in BANDS and pred in BANDS:
        total += 1
        conf[(gold, pred)] += 1
        if pred == gold:
            agree += 1
        if abs(BANDS.index(pred) - BANDS.index(gold)) <= 1:
            adjacent += 1

with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"model={args.model}; books audited={len(rows)}; comparable={total}")
print(f"exact agreement: {agree/total:.3f}; adjacent (within one band): {adjacent/total:.3f}")
odd = Counter(r["pred"] for r in rows if r["pred"] not in BANDS)
print(f"non-canonical predictions: {dict(odd)}")
print(f"{'gold/pred':>10} " + " ".join(f"{b:>6}" for b in BANDS))
for g in BANDS:
    print(f"{g:>10} " + " ".join(f"{conf[(g, p)]:>6}" for p in BANDS))
