"""Build a flat training manifest from the LLaVA-format JSONs in updated_jsons/.

Each output line: one training sample with age band, task, cleaned text turns,
word counts and Flesch-Kincaid grade (for readability-ordered curricula).

Example:
    python training/prepare_manifest.py \
        --json_dir /path/to/generated-json \
        --image_root /path/to/icdl-page-images \
        --out /path/to/manifest.jsonl
"""
import argparse
import glob
import json
import os
import re

from textstat import flesch_kincaid_grade

AGE_BANDS = ["3_5", "6_8", "9_12", "12"]
TASKS = ["caption", "vqa", "reasoning"]

parser = argparse.ArgumentParser()
parser.add_argument("--json_dir", required=True)
parser.add_argument("--image_root", required=True)
parser.add_argument("--out", default="manifest.jsonl")
args = parser.parse_args()


def parse_conversations(convs):
    """LLaVA conversations -> list of {user, assistant} turns, <image> tag stripped.

    nanoVLM inserts its own image token string, so the placeholder must go.
    """
    turns = []
    for human, gpt in zip(convs[::2], convs[1::2]):
        assert human["from"] == "human" and gpt["from"] == "gpt", convs
        user = re.sub(r"\s*<image>\s*", " ", human["value"]).strip()
        turns.append({"user": user, "assistant": gpt["value"].strip()})
    return turns


n_out, n_skipped, words_total = 0, 0, 0
band_words = {b: 0 for b in AGE_BANDS}
with open(args.out, "w") as out:
    for task in TASKS:
        for band in AGE_BANDS:
            path = os.path.join(args.json_dir, f"{task}_{band}.json")
            data = json.load(open(path))
            for datum in data:
                image = datum["image"]
                if not os.path.isfile(os.path.join(args.image_root, image)):
                    n_skipped += 1
                    continue
                turns = parse_conversations(datum["conversations"])
                if not turns or any(not t["user"] or not t["assistant"] for t in turns):
                    n_skipped += 1
                    continue
                text = " ".join(t["user"] + " " + t["assistant"] for t in turns)
                n_words = len(text.split())
                rec = {
                    "id": f"{task}_{band}_{datum['id']}",
                    "image": image,
                    "texts": turns,
                    "task": task,
                    "age_band": band,
                    "n_words": n_words,
                    "fk_grade": round(flesch_kincaid_grade(text), 3),
                }
                out.write(json.dumps(rec) + "\n")
                n_out += 1
                words_total += n_words
                band_words[band] += n_words

print(f"wrote {n_out} samples to {args.out} ({n_skipped} skipped)")
print(f"total words (user+assistant): {words_total:,}")
for band, w in band_words.items():
    print(f"  {band}: {w:,} words")
