"""Merge alternate-teacher NLL scores into the scored manifest.

compute_perplexity.py overwrites `nll`/`n_tokens` in its output, so alternate
teachers are run into separate files and merged here: the base manifest keeps
the SmolLM2 `nll` and `n_tokens` (used for budgets), and each alternate file
contributes an `nll_<name>` column. Also prints the teacher-agreement stats
used in the paper (Spearman rank correlation, easiest/hardest-quartile
overlap, per-band means).

    python merge_teacher_scores.py --base manifest_scored.jsonl \
        --alt qwen=manifest_qwen_raw.jsonl --out manifest_scored_multi.jsonl
"""
import argparse
import json

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--alt", action="append", required=True, help="name=file.jsonl")
parser.add_argument("--out", required=True)
args = parser.parse_args()

base = [json.loads(l) for l in open(args.base)]
BANDS = ["3_5", "6_8", "9_12", "12"]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


base_nll = np.array([r["nll"] for r in base])
for spec in args.alt:
    name, path = spec.split("=", 1)
    alt = [json.loads(l) for l in open(path)]
    assert len(alt) == len(base), f"{path}: {len(alt)} vs {len(base)} lines"
    for rb_, ra_ in zip(base, alt):
        assert rb_["image"] == ra_["image"], "line misalignment"
        rb_[f"nll_{name}"] = ra_["nll"]
    alt_nll = np.array([r[f"nll_{name}"] for r in base])

    rho = spearman(base_nll, alt_nll)
    q = len(base) // 4
    e0, e1 = np.argsort(base_nll)[:q], np.argsort(alt_nll)[:q]
    h0, h1 = np.argsort(base_nll)[-q:], np.argsort(alt_nll)[-q:]
    easy_ov = len(set(e0) & set(e1)) / q
    hard_ov = len(set(h0) & set(h1)) / q
    print(f"[{name}] spearman rho = {rho:.4f}; "
          f"easiest-quartile overlap = {easy_ov:.3f}; hardest = {hard_ov:.3f}")
    for band in BANDS:
        idx = [i for i, r in enumerate(base) if r["age_band"] == band]
        print(f"[{name}]   band {band:>4}: mean nll {alt_nll[idx].mean():.3f} "
              f"(smollm2 {base_nll[idx].mean():.3f}, n={len(idx)})")

with open(args.out, "w") as f:
    for r in base:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(base)} records -> {args.out}")
