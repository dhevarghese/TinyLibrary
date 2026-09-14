"""Score each manifest sample with a small pretrained LM for perplexity-ordered curricula.

Difficulty = mean NLL of the assistant text conditioned on the user prompt (teacher-model data scoring). 
Also records the exact token count per sample for step-budget estimates.

Run on a GPU node:
    python compute_perplexity.py --manifest manifest.jsonl --out manifest_scored.jsonl
"""
import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--scorer", default="HuggingFaceTB/SmolLM2-135M")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=1024)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(args.scorer)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.scorer, torch_dtype=torch.bfloat16).to(device).eval()

records = [json.loads(l) for l in open(args.manifest)]
# Length-sorted batching: less padding waste, and keeps the float32 loss
# buffer (batch x seq x vocab) bounded by similar-length groupings.
sorted_order = sorted(range(len(records)), key=lambda i: records[i]["n_words"])


def sample_texts(rec):
    """(prompt, full) pair so we can mask the prompt out of the NLL."""
    prompt = rec["texts"][0]["user"] + "\n"
    full = "\n".join(t["user"] + "\n" + t["assistant"] for t in rec["texts"])
    return prompt, full


@torch.no_grad()
def score_batch(batch):
    prompts, fulls = zip(*(sample_texts(r) for r in batch))
    enc = tokenizer(list(fulls), return_tensors="pt", padding=True,
                    truncation=True, max_length=args.max_length).to(device)
    prompt_lens = [len(tokenizer(p, truncation=True, max_length=args.max_length)["input_ids"])
                   for p in prompts]
    logits = model(**enc).logits
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for i, pl in enumerate(prompt_lens):  # don't score the prompt tokens
        labels[i, :pl] = -100
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, ignore_index=-100, reduction="none")
    n_scored = (shift_labels != -100).sum(dim=1).clamp(min=1)
    mean_nll = (nll.sum(dim=1) / n_scored).tolist()
    n_tokens = enc["attention_mask"].sum(dim=1).tolist()
    return mean_nll, n_tokens


for i in tqdm(range(0, len(sorted_order), args.batch_size)):
    batch = [records[j] for j in sorted_order[i:i + args.batch_size]]
    mean_nll, n_tokens = score_batch(batch)
    for rec, nll, nt in zip(batch, mean_nll, n_tokens):
        rec["nll"] = round(nll, 4)
        rec["n_tokens"] = int(nt)

with open(args.out, "w") as out:  # original manifest order
    for rec in records:
        out.write(json.dumps(rec) + "\n")

print(f"scored {len(records)} samples with {args.scorer} -> {args.out}")
