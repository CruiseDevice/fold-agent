# fold-agent — plan

> Flagship from `ai-agent-project-roadmap.md` Phase 3. Untrained, prompt-only
> implementation of the context-folding paradigm (AgentFold mechanism +
> IterResearch report + MEM1 eval). See DESIGN.md for the evidence behind
> every decision.

## Phases

### Phase B — Baseline (naive ReAct, append-everything) — ~6h
- [x] 1. `llm.py` — `chat(messages) -> str` via `ollama.chat`, `MODEL` constant swap seam (~30 lines)
- [x] 2. `tools.py` — `web_search(queries: list[str])` batch (DDGS, top-k each) + `visit(url, goal)` (requests + bs4 + truncate); disk cache on both (`cache/`, keyed by hash); honest failure strings; "no results" ≠ "search failed" (~80 lines)
- [x] 3. `agent.py` — hand-rolled ReAct loop: Thought/Action/Observation appended verbatim; regex parse; parse failure → observation telling the model so; `max_steps`; **turns-left hint each step** (~110 lines) ← the loop lesson
  - 📖 before typing: read **ReAct** (Yao 2022, ~9 pp) — this paper is the spec
- [x] 4. `run.py` — one question end-to-end, full trace to disk (steps, tokens per step)
- [x] 5. `eval.py` + `data/hotpot20.json` (HotpotQA dev slice, 20 questions) — EM + word-F1, peak context tokens (tiktoken, marked approx), steps, parse-failure count → baseline table in `eval.md`. Expect visible degradation >15 steps
  - 📖 before writing eval.md: **IterResearch §3.1 + §4.5** (~4 pp)

### Phase F — Folding (AgentFold mechanism) — ~6h
- [x] 6. Output contract upgrade: `<think>/<fold>/<action>` tags; fold = one-line JSON `{"range":[k,t-1],"summary":"..."}`. Context becomes block list `[(start,end,text)]` (ordered partition), rendered per call
- [x] 7. Fold application: retract blocks intersecting [k,t-1], insert new block. Granular (k=t-1) + deep consolidation (k<t-1). Malformed fold → no-op, counted
  - 📖 before typing: **AgentFold §3.2–3.5** (~5 pp) — you've now watched the baseline suffocate; it'll read differently
- [x] 8. Re-run eval vs baseline. Gate: same-or-better accuracy, peak tokens cut ≥3×, parse-failure rate <20% of steps. Iterate prompt (when to deep-consolidate: sub-investigation done or dead end)

### Phase L — Long-horizon — ~5h
- [x] 9. Evolving report section (IterResearch): one line per question, rewritten by the model each step, riding in the output contract + prompt (`SYSTEM_MULTI`); parse_output → 5-tuple. Smoke: 3/3 exam, peak DROPPED 2645→1820 after deep fold
- [x] 10. Multi-objective eval: 5 exams × 4 questions, semicolon answers, per-part EM (MEM1 protocol), 48-step budget. DoD #1 MET (exam1: baseline 0/4 @ 9,471 tok, folded 2/4 @ 3,312). Folded EM 0.60 vs baseline 0.50, peak −51%
- [x] 11. Plots: context-vs-steps curve with fold markers (`plots/context_curve.png`) + peak/accuracy summary (`plots/peak_accuracy.png`); `capture_exam.py` instrumented re-run — DoD claim survived re-run

### Phase P — Polish — ~4h
- [x] 12. Ablations (report-only, fold-only, fixed full) + README. Finding 8: report = accuracy mechanism, folding = context mechanism. Finding 9: 1-example prompt cut format errors 26%→8%. Best arm: fold+report-fixed, 0.65 EM / 0.76 F1 / 3,227 peak tok

> **STATUS: COMPLETE** — all 12 steps, 18 commits, final `b8068f5`.
> 9 findings in `eval.md`; roadmap L3 flagship done.

## DoD

1. [x] Agent solves ≥1 multi-objective task the naive version cannot (exam1: 2/4 vs 0/4; baseline drowned at 9.5K tokens; survived re-run)
2. [x] Peak context grows sub-linearly with steps (context_curve.png: folded 1.9–2.4K flat vs baseline up to 9.5K; −48–51% avg)
3. [x] `eval.md`: baseline vs folded tables + parse-failure accounting (9 findings, gates tracked, 26% breach flagged then fixed to 8%)

## Working rule

User types all logic (One Rule). AI designs, reviews, runs, writes docs/eval.
Readings are JIT — each paper arrives only when its code slot opens (~22 pp total).
