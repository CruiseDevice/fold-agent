"""Run one question end-to-end; dump the full trace to traces/ as JSON.

Usage: python run.py <question>     (no arg -> default question)
"""
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from agent import run
from llm import MODEL

TRACES = Path("traces")

DEFAULT_QUESTION = (
    "What government position was held by the woman who portrayed "
    "Jyn Erso in Rogue One?"
)


def slug(text: str) -> str:
    """Filename-safe short slug of a question."""
    words = re.sub(r"[^a-z0-9 ]", "", text.lower()).split()[:6]
    return "-".join(words) if words else "q"


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION

    start = time.time()
    result = run(question)
    elapsed = time.time() - start

    trace = {
        "model": MODEL,
        "question": question,
        "answer": result.answer,
        "num_steps": len(result.steps),
        "peak_tokens": result.peak_tokens,
        "parse_failures": result.parse_failures,
        "elapsed_s": round(elapsed, 1),
        "steps": [asdict(s) for s in result.steps],
    }

    TRACES.mkdir(exist_ok=True)
    path = TRACES / f"{datetime.now():%Y%m%d-%H%M%S}-{slug(question)}.json"
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"QUESTION: {question}")
    print(f"ANSWER:   {result.answer!r}")
    print(f"steps={len(result.steps)}  peak_tokens={result.peak_tokens}  "
          f"parse_failures={result.parse_failures}  elapsed={elapsed:.1f}s")
    print(f"trace:    {path}")
    for s in result.steps:
        print(f"{s.n:>2}. [{s.prompt_tokens:>5} tok] {s.action[:72]}")


if __name__ == "__main__":
    main()
