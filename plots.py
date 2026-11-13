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

# ---- 3. Phase H: the 20-question stress run (money plot) ------------------
fig, ax = plt.subplots(figsize=(7, 4.4))
for arm, color in [("baseline", BLUE), ("folded", ORANGE)]:
    t = load(f"traces/exam20_{arm}.json")
    xs = [s["n"] for s in t["steps"]]
    ys = [s["prompt_tokens"] for s in t["steps"]]
    ax.plot(xs, ys, marker="o", color=color, label=f"{arm} (instrumented re-run)")
    if arm == "folded":
        fx = [s["n"] - 1 for s in t["steps"] if s["fold_applied"]]
        fy = [y for s, y in zip(t["steps"], ys) if s["fold_applied"]]
        ax.scatter(fx, fy, marker="v", s=60, color=color, zorder=5,
                   label="fold applied")
peaks = {a: load(f"results_multi20_{a}.json")[0]["peak_tokens"] for a in ("baseline", "folded")}
for a, c in [("baseline", BLUE), ("folded", ORANGE)]:
    ax.axhline(peaks[a], color=c, ls="--", lw=1, alpha=0.55)
ax.text(21.2, peaks["baseline"] - 4200, f"official run peak: {peaks['baseline']:,}\n(gap to folded: 6.8x)",
       color=BLUE, fontsize=9, ha="right")
ax.text(21.2, peaks["folded"] + 800, f"official run peak: {peaks['folded']:,}",
       color=DARK_ORANGE, fontsize=9, ha="right")
ax.set_xlabel("step")
ax.set_ylabel("prompt tokens (approx, cl100k)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_title("Exam 20 — all 20 questions in one run: append-only vs folded")
ax.legend(loc="center left", fontsize=8.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "stress20.png", dpi=160)

# ---- 4. Phase H: gap scales with forced search, not question count -------
fig, ax = plt.subplots(figsize=(5.6, 3.8))
sizes = ["4 questions", "8 questions", "20 questions"]
gaps = [6232 / 3064, 7880 / 5712, 40529 / 5992]  # baseline peak / folded peak
bars = ax.bar(sizes, gaps, color=ORANGE, width=0.55)
for b, g in zip(bars, gaps):
    ax.text(b.get_x() + b.get_width() / 2, g + 0.15, f"{g:.1f}x", ha="center", fontsize=10)
ax.set_ylabel("baseline peak / folded peak")
ax.set_title("Peak-context gap grows with task size\n(8-question run: baseline answered mostly from memory — Finding 12)", fontsize=9.5)
ax.set_ylim(0, 8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS / "gap_growth.png", dpi=160)
print("wrote plots/stress20.png and plots/gap_growth.png")
