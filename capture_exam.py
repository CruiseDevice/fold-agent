"""Run one exam on both arms and save full per-step traces for plotting.

Usage: python capture_exam.py exam1
Writes traces/exam1_baseline.json and traces/exam1_folded.json
(~5-8 min; search cache is warm, only LLM calls cost time)."""

import json
import sys
import time
from pathlib import Path

from eval import EXAMS_PATH
from agent import SYSTEM_MULTI, SYSTEM_MULTI_BASELINE, run, run_folded

TRACE_DIR = Path(__file__).parent / "traces"


def exam_prompt(exam):
    return "Answer each question. " + " ".join(
        f"{j}. {q['question']}" for j, q in enumerate(exam["questions"], 1)
    )


def main(exam_id: str, max_steps: int = 48) -> None:
    exam = next(e for e in json.loads(EXAMS_PATH.read_text()) if e["id"] == exam_id)
    prompt = exam_prompt(exam)
    TRACE_DIR.mkdir(exist_ok=True)

    for arm, agent_run, system in [
        ("baseline", run, SYSTEM_MULTI_BASELINE),
        ("folded", run_folded, SYSTEM_MULTI),
    ]:
        t0 = time.time()
        r = agent_run(prompt, max_steps, system=system)
        out = {
            "arm": arm,
            "id": exam_id,
            "answer": r.answer,
            "golds": [q["answer"] for q in exam["questions"]],
            "steps": [
                {
                    "n": s.n,
                    "prompt_tokens": s.prompt_tokens,
                    "action": s.action,
                    "observation_chars": len(s.observation),
                    "fold": s.fold,
                    "fold_applied": s.fold_applied,
                    "report": s.report,
                }
                for s in r.steps
            ],
        }
        path = TRACE_DIR / f"{exam_id}_{arm}.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"{arm}: {len(r.steps)} steps, peak {r.peak_tokens}, "
              f"{r.folds_applied} folds, {round(time.time() - t0)}s -> {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "exam1")
