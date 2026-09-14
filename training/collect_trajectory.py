"""Collate checkpoint-trajectory fast-eval results and plot learning curves.

Walks strict/results/<run>/step_N/zero_shot/causal/<task>/<sub>_fast/
best_temperature_report.txt into a long CSV, then plots accuracy-vs-step per
task with one line per condition (mean over seeds, +-1 sd band).

    python collect_trajectory.py --results_dir .../strict/results \
        --run_filter ep5 --out_dir .../analysis/ep5
"""
import argparse
import glob
import os
import re
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from result_runs import (add_run_filter_args, compile_run_name_filter,
                         condition_label, parse_run_name, run_matches)

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True)
parser.add_argument("--run_filter", help="legacy substring filter applied to complete run names")
parser.add_argument("--out_dir", required=True)
add_run_filter_args(parser)
args = parser.parse_args()
name_filter = compile_run_name_filter(parser, args.run_name_regex)
os.makedirs(args.out_dir, exist_ok=True)

# rows: (epochs, lm_init, modality, order, schedule, seed, step, task, acc)
rows = []
origins = {}
for report in sorted(glob.glob(f"{args.results_dir}/*/step_*/zero_shot/causal/*/*/best_temperature_report.txt")):
    parts = report.split(os.sep)
    model, step, task = parts[-7], parts[-6], parts[-2]
    if args.run_filter is not None and args.run_filter not in model:
        continue
    spec = parse_run_name(model)
    if spec is None or not run_matches(spec, args, name_filter):
        continue
    with open(report, encoding="utf-8") as handle:
        acc = re.search(r"### AVERAGE ACCURACY\n([\d.]+)", handle.read())
    if acc:
        step_number = int(step.removeprefix("step_"))
        identity = spec.condition, spec.seed, step_number, task
        if identity in origins:
            raise RuntimeError(
                "Multiple trajectory results match the same condition/seed/step/task:\n"
                f"  {origins[identity]}\n"
                f"  {report}\n"
                "Select one experiment with the run filters."
            )
        origins[identity] = report
        rows.append((*spec.condition, spec.seed, step_number, task, float(acc.group(1))))

csv_path = os.path.join(args.out_dir, "trajectories.csv")
with open(csv_path, "w") as f:
    f.write("epochs,lm_init,modality,order,schedule,seed,step,task,accuracy\n")
    for r in sorted(rows):
        f.write(",".join(map(str, r)) + "\n")
print(f"{len(rows)} rows -> {csv_path}")

tasks = sorted({r[7] for r in rows})
conditions = sorted({r[:5] for r in rows})
colors = plt.cm.tab10.colors

for task in tasks:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ci, condition in enumerate(conditions):
        epochs, lm_init, modality, order, schedule = condition
        # step -> [acc over seeds]
        by_step = defaultdict(list)
        for ep, init, mod, o, s, seed, step, t, acc in rows:
            if (ep, init, mod, o, s) == condition and t == task:
                by_step[step].append(acc)
        if not by_step:
            continue
        steps = sorted(by_step)
        means = [statistics.mean(by_step[s]) for s in steps]
        sds = [statistics.stdev(by_step[s]) if len(by_step[s]) > 1 else 0 for s in steps]
        style = "-" if schedule == "staged" else "--"
        ax.plot(steps, means, style, color=colors[ci % 10],
                label=condition_label(condition), lw=1.6)
        ax.fill_between(steps, [m - s for m, s in zip(means, sds)],
                        [m + s for m, s in zip(means, sds)], color=colors[ci % 10], alpha=0.12)
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy (%)")
    selection = args.run_filter or "selected runs"
    ax.set_title(f"{task} over training ({selection})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"trajectory_{task}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"plot -> {out}")
