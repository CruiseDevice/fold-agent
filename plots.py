"""Offline plots from committed results + captured traces. No model calls.

Usage: python plots.py  ->  plots/*.png
1. context_curve.png  — exam1 prompt tokens per step, both arms (instrumented
   re-run; official-run peaks drawn as dashed reference lines: baseline
   varies hugely run-to-run, folded stays flat — variance is itself a result)
2. peak_accuracy.png  — official per-exam peak tokens (bars) + EM (labels),
   and the accuracy summary across both eval settings."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

BLUE, ORANGE = "#4477AA", "#EE7733"
DARK_ORANGE = "#B4551B"


def load(name):
    return json.loads((ROOT / name).read_text())


# ---- 1. context-vs-steps curve (exam1, instrumented re-run) --------------
fig, ax = plt.subplots(figsize=(7, 4.2))
for arm, color in [("baseline", BLUE), ("folded", ORANGE)]:
    t = load(f"traces/exam1_{arm}.json")
    xs = [s["n"] for s in t["steps"]]
    ys = [s["prompt_tokens"] for s in t["steps"]]
    ax.plot(xs, ys, marker="o", color=color, label=f"{arm} (this run)")
    if arm == "folded":
        fx = [s["n"] - 1 for s in t["steps"] if s["fold_applied"]]  # fold applies before step n
        fy = [y for s, y in zip(t["steps"], ys) if s["fold_applied"]]
        ax.scatter(fx, fy, marker="v", s=70, color=color, zorder=5)
        ax.annotate("folds applied", xy=(fx[0], fy[0]), xytext=(fx[0] + 0.3, fy[0] - 500),
                    color=color, fontsize=9)

# official-run peaks as reference lines (same question, committed results)
for arm, color in [("baseline", BLUE), ("folded", ORANGE)]:
    peak = load(f"results_multi_{arm}.json")[0]["peak_tokens"]
    ax.axhline(peak, color=color, ls="--", lw=1, alpha=0.55)
    ax.text(0.2, peak + 150, f"official run peak: {peak:,}", color=color, fontsize=8.5)

ax.set_xlabel("step")
ax.set_ylabel("prompt tokens (approx, cl100k)")
ax.set_title("Exam 1: context growth — append-only vs folded")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "context_curve.png", dpi=160)

# ---- 2. official per-exam peaks + accuracy summary ------------------------
base = load("results_multi_baseline.json")
fold = load("results_multi_folded.json")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

ids = [r["id"] for r in base]
x = range(len(ids))
w = 0.38
ax1.bar([i - w / 2 for i in x], [r["peak_tokens"] for r in base], w, color=BLUE, label="baseline")
ax1.bar([i + w / 2 for i in x], [r["peak_tokens"] for r in fold], w, color=ORANGE, label="folded")
for i, (rb, rf) in enumerate(zip(base, fold)):
    ax1.text(i - w / 2, rb["peak_tokens"] + 120, f"{rb['em']:.2f}", ha="center", fontsize=8, color=BLUE)
    ax1.text(i + w / 2, rf["peak_tokens"] + 120, f"{rf['em']:.2f}", ha="center", fontsize=8, color=DARK_ORANGE)
ax1.set_xticks(list(x), ids)
ax1.set_ylabel("peak prompt tokens")
ax1.set_title("Multi-objective exams: peak context (EM above bars)")
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)
ax1.set_ylim(0, 10400)

def agg(rows, key):
    return sum(r[key] for r in rows) / len(rows)

labels = ["single EM", "single F1", "multi EM", "multi F1"]
base_vals = [agg(load("results_baseline.json"), "em"), agg(load("results_baseline.json"), "f1"),
             agg(base, "em"), agg(base, "f1")]
fold_vals = [agg(load("results_folded.json"), "em"), agg(load("results_folded.json"), "f1"),
             agg(fold, "em"), agg(fold, "f1")]
x2 = range(len(labels))
ax2.bar([i - w / 2 for i in x2], base_vals, w, color=BLUE, label="baseline")
ax2.bar([i + w / 2 for i in x2], fold_vals, w, color=ORANGE, label="folded")
ax2.set_xticks(list(x2), labels, fontsize=9)
ax2.set_ylim(0, 1)
ax2.set_ylabel("score")
for i, (bv, fv) in enumerate(zip(base_vals, fold_vals)):
    ax2.text(i - w / 2, bv + 0.02, f"{bv:.2f}", ha="center", fontsize=8, color=BLUE)
    ax2.text(i + w / 2, fv + 0.02, f"{fv:.2f}", ha="center", fontsize=8, color=DARK_ORANGE)
ax2.set_title("Accuracy: parity single, +8-10pp on exams")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(PLOTS / "peak_accuracy.png", dpi=160)
print("wrote plots/context_curve.png and plots/peak_accuracy.png")
