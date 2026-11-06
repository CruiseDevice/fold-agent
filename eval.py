"""Metrics: exact match + word-level F1 (MEM1 / HotpotQA conventions)."""

import json
import re
import string
import time
from collections import Counter
from pathlib import Path

from agent import run

DATA_PATH = Path(__file__).parent / "data" / "hotpot20.json"
RESULTS_PATH = Path(__file__).parent / "results_baseline.json"

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip articles/punct, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = _ARTICLES.sub(" ", text)
    text = "".join(ch for ch in text if ch not in _PUNCT)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> int:
    """1 if normalized strings are identical, else 0."""
    return int(normalize(pred) == normalize(gold))


def word_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between normalized strings. 0.0 on no overlap.
    (both empty -> 1.0, MEM1 convention)"""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(max_steps: int = 20, limit: int = 20) -> dict:
    """Run the agent on each question; write per-question results JSON.
    Resumable: skip ids already in RESULTS (crashes mid-run lose nothing).
    Returns aggregate summary over everything in the results file.
    """
    data = json.loads(DATA_PATH.read_text())[:limit]
    results = (
        json.loads(RESULTS_PATH.read_text())
        if RESULTS_PATH.exists()
        else []
    )
    done = {row["id"] for row in results}

    for i, q in enumerate(data, 1):
        if q["id"] in done:
            continue

        t0 = time.time()
        r = run(q["question"], max_steps)

        row = {
            "id": q["id"],
            "question": q["question"],
            "gold": q["answer"],
            "pred": r.answer,
            "em": exact_match(r.answer, q["answer"]),
            "f1": word_f1(r.answer, q["answer"]),
            "steps": len(r.steps),
            "peak_tokens": r.peak_tokens,
            "parse_failures": r.parse_failures,
            "elapsed_s": round(time.time() - t0, 1),
        }
        results.append(row)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

        print(
            f"[{i}/{len(data)}] em={row['em']} f1={row['f1']:.2f} "
            f"steps={row['steps']} pred={row['pred']!r} gold={row['gold']!r}"
        )

    n = len(results)
    summary = {
        "n": n,
        "em": sum(r["em"] for r in results) / n,
        "f1": sum(r["f1"] for r in results) / n,
        "steps": sum(r["steps"] for r in results) / n,
        "peak_tokens": sum(r["peak_tokens"] for r in results) / n,
        "parse_failures": sum(r["parse_failures"] for r in results),
        "elapsed_s": sum(r["elapsed_s"] for r in results),
    }
    print(
        f"\nn={summary['n']}  EM={summary['em']:.2f}  F1={summary['f1']:.2f}  "
        f"avg steps={summary['steps']:.1f}  "
        f"avg peak tokens={summary['peak_tokens']:.0f}  "
        f"parse failures={summary['parse_failures']}"
    )
    return summary


if __name__ == "__main__":
    evaluate()