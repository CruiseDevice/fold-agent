"""Capture one exam on both arms with full workspace snapshots (trace v2).

Usage: python capture_exam.py exam1
Writes traces/<id>_<arm>.json with per-step: prompt tokens, action,
observation preview, fold directive + applied flag, report, and the
workspace block state AFTER each step (post-fold, post-append)."""

import json
import sys
import time
from pathlib import Path

from eval import EXAMS_PATH
from agent import SYSTEM_MULTI, SYSTEM_MULTI_BASELINE, run, run_folded

TRACE_DIR = Path(__file__).parent / "traces"


def main(exam_id: str, max_steps: int = 48) -> None:
    exam = next(e for e in json.loads(EXAMS_PATH.read_text()) if e["id"] == exam_id)
    prompt = "Answer each question. " + " ".join(
        f"{j}. {q['question']}" for j, q in enumerate(exam["questions"], 1)
    )
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
            "question": prompt,
            "answer": r.answer,
            "golds": [q["answer"] for q in exam["questions"]],
            "steps": [
                {
                    "n": s.n,
                    "prompt_tokens": s.prompt_tokens,
                    "action": s.action,
                    "observation_chars": len(s.observation),
                    "observation_preview": s.observation[:300],
                    "fold": s.fold,
                    "fold_applied": s.fold_applied,
                    "fold_malformed": s.fold_malformed,
                    "report": s.report,
                    "workspace_after": [
                        {"start": b.start, "end": b.end, "text": b.text}
                        for b in s.workspace_after
                    ],
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
