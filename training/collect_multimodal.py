"""Collate multimodal (VQA / Winoground) results into a per-condition table.

Walks strict/results/<model>/main/zero_shot/causal/<task>/<dataset>/
best_temperature_report.txt (written by run_multimodal.py), extracts the
### AVERAGE ACCURACY, and aggregates mean +/- std over seeds. Epoch budget,
LM initialization, and modality remain separate; duplicate reruns are rejected.

    python collect_multimodal.py --results_dir $BASE/babylm-eval/strict/results
"""
import argparse
import glob
import os
import re
import statistics
from collections import defaultdict

from result_runs import (add_run_filter_args, compile_run_name_filter,
                         condition_label, parse_run_name, record_unique_score,
                         run_matches)

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True)
add_run_filter_args(parser)
args = parser.parse_args()
name_filter = compile_run_name_filter(parser, args.run_name_regex)

TASKS = ["vqa/vqa_filtered", "winoground/winoground_filtered"]
SHORT = {"vqa/vqa_filtered": "VQA", "winoground/winoground_filtered": "Winoground"}

scores = defaultdict(lambda: defaultdict(dict))
origins = {}
for report in sorted(glob.glob(f"{args.results_dir}/*/main/zero_shot/causal/*/*/best_temperature_report.txt")):
    parts = report.split(os.sep)
    model, task = parts[-7], f"{parts[-3]}/{parts[-2]}"
    spec = parse_run_name(model)
    if spec is None or task not in TASKS or not run_matches(spec, args, name_filter):
        continue
    with open(report, encoding="utf-8") as handle:
        txt = handle.read()
    acc = re.search(r"### AVERAGE ACCURACY\n([\d.]+)", txt)
    if acc:
        record_unique_score(
            scores, origins, spec.condition, task, spec, float(acc.group(1)), report
        )

present = [t for t in TASKS if any(t in c for c in scores.values())]
print("| condition | " + " | ".join(SHORT[t] for t in present) + " |")
print("|" + "---|" * (len(present) + 1))
for cond in sorted(scores):
    cells = []
    for t in present:
        seeds = scores[cond].get(t, {})
        if not seeds:
            cells.append("—")
        else:
            mean = statistics.mean(seeds.values())
            std = statistics.stdev(seeds.values()) if len(seeds) > 1 else 0.0
            cells.append(f"{mean:.2f} ± {std:.2f} (n={len(seeds)})")
    print(f"| {condition_label(cond)} | " + " | ".join(cells) + " |")
