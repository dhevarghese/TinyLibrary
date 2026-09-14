"""Generate camera-quality PDF figures for the BabyLM 2026 paper.

Reads analysis/ep{5,10}/trajectories.csv (from training/collect_trajectory.py)
plus the per-band corpus statistics reported in the paper and writes PDFs to
the requested output directory.

    python analysis/make_paper_figures.py --out paper-figures
"""
import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--analysis_dir", default=os.path.dirname(os.path.abspath(__file__)))
parser.add_argument("--out", required=True)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

ORDER_COLORS = {
    "developmental": "#1f77b4", "antidevelopmental": "#9467bd",
    "readability": "#2ca02c", "perplexity": "#d62728", "random": "#555555",
    "randomblock": "#8c564b", "qwenppl": "#e377c2", "gemmappl": "#ff7f0e",
}
ORDER_LABELS = {
    "developmental": "developmental", "antidevelopmental": "anti-developmental",
    "readability": "readability", "perplexity": "perplexity", "random": "random",
    "randomblock": "random-block", "qwenppl": "perplexity (Qwen3)",
    "gemmappl": "perplexity (Gemma)",
}
# shown as one condition regardless of schedule (staged-only or schedule-less)
SCHEDULE_FREE = {"random", "randomblock", "qwenppl", "gemmappl"}

# ---- Figure 1: per-band difficulty (FK monotone vs teacher NLL U-shape) ----
bands = ["3–5", "6–8", "9–12", "12+"]
fk = [3.51, 5.34, 7.76, 10.76]
nll = [2.856, 2.542, 2.523, 2.748]

fig, ax1 = plt.subplots(figsize=(3.2, 2.1))
ax2 = ax1.twinx()
l1, = ax1.plot(bands, fk, "o-", color="#2ca02c", lw=1.6, ms=4, label="FK grade level")
l2, = ax2.plot(bands, nll, "s--", color="#d62728", lw=1.6, ms=4, label="teacher NLL")
ax1.set_xlabel("age band")
ax1.set_ylabel("Flesch–Kincaid grade", color="#2ca02c")
ax2.set_ylabel("SmolLM2-135M NLL (nats/token)", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#2ca02c")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax1.legend(handles=[l1, l2], loc="upper left", frameon=False)
ax1.grid(alpha=0.25, axis="x")
fig.savefig(os.path.join(args.out, "band_difficulty.pdf"))
plt.close(fig)
print("band_difficulty.pdf")


def load(ep):
    rows = []
    with open(os.path.join(args.analysis_dir, f"ep{ep}", "trajectories.csv")) as f:
        for r in csv.DictReader(f):
            rows.append((r["order"], r["schedule"], int(r["seed"]),
                         int(r["step"]), r["task"], float(r["accuracy"])))
    return rows


def trajectory_panels(ep, task, fname, title):
    rows = [r for r in load(ep) if r[4] == task]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45), sharey=True)
    for ax, schedule in zip(axes, ["staged", "competence"]):
        for order in ["random", "randomblock", "qwenppl", "developmental",
                      "antidevelopmental", "readability", "perplexity"]:
            # random ignores the schedule (shown in both panels); the staged-only
            # controls (randomblock, qwenppl) appear in the staged panel only
            if order in ("randomblock", "qwenppl", "gemmappl") and schedule != "staged":
                continue
            sched = "staged" if order in SCHEDULE_FREE else schedule
            by_step = defaultdict(list)
            for o, s, seed, step, t, acc in rows:
                if (o, s) == (order, sched):
                    by_step[step].append(acc)
            if not by_step:
                continue
            steps = sorted(by_step)
            means = [statistics.mean(by_step[s]) for s in steps]
            sds = [statistics.stdev(by_step[s]) if len(by_step[s]) > 1 else 0 for s in steps]
            c = ORDER_COLORS[order]
            ls = ":" if order in SCHEDULE_FREE else "-"
            ax.plot(steps, means, ls, color=c, lw=1.4, label=ORDER_LABELS[order])
            ax.fill_between(steps, [m - s for m, s in zip(means, sds)],
                            [m + s for m, s in zip(means, sds)], color=c, alpha=0.12, lw=0)
        ax.set_title(f"{schedule.capitalize()} Schedule")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(args.out, fname))
    plt.close(fig)
    print(fname)


trajectory_panels(10, "blimp_fast", "trajectory_blimp_ep10.pdf", "BLiMP (fast) accuracy (%)")
trajectory_panels(5, "blimp_fast", "trajectory_blimp_ep5.pdf", "BLiMP (fast) accuracy (%)")
trajectory_panels(10, "entity_tracking_fast", "trajectory_entity_ep10.pdf",
                  "entity tracking (fast) accuracy (%)")

# ---- Per-seed final full-BLiMP strip plot (analysis/blimp_per_seed.csv) ----
seed_rows = []
with open(os.path.join(args.analysis_dir, "blimp_per_seed.csv")) as f:
    for r in csv.DictReader(f):
        seed_rows.append((r["epochs"], r["order"], r["schedule"], int(r["seed"]),
                          float(r["blimp"])))

conds = sorted({(o, s) for _, o, s, _, _ in seed_rows},
               key=lambda c: statistics.mean(v for e, o, s, _, v in seed_rows
                                             if (o, s) == c and e == "ep10"))
labels = [f"{ORDER_LABELS[o]}" + ("" if o in SCHEDULE_FREE else f"/{s[:4]}")
          for o, s in conds]

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
for ax, ep in zip(axes, ["ep5", "ep10"]):
    rand_mean = statistics.mean(v for e, o, s, _, v in seed_rows
                                if e == ep and o == "random")
    ax.axvline(rand_mean, color="#555555", ls=":", lw=1, zorder=0)
    for yi, (o, s) in enumerate(conds):
        vals = [v for e, oo, ss, _, v in seed_rows if (oo, ss) == (o, s) and e == ep]
        if not vals:  # randomblock control exists only in the 10-epoch grid
            continue
        ax.plot(vals, [yi] * len(vals), "o", color=ORDER_COLORS[o], ms=4,
                alpha=0.85, mew=0)
        ax.plot([statistics.mean(vals)], [yi], "|", color=ORDER_COLORS[o],
                ms=11, mew=1.8)
    ax.set_title(f"{ep[2:]} epochs")
    ax.set_xlabel("BLiMP accuracy (%)")
    ax.grid(alpha=0.25, axis="x")
axes[0].set_yticks(range(len(conds)), labels)
fig.tight_layout()
fig.savefig(os.path.join(args.out, "blimp_per_seed.pdf"))
plt.close(fig)
print("blimp_per_seed.pdf")
