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

> **STATUS: Phase B/F/L/P COMPLETE + PUBLISHED. Phase H COMPLETE —
> steps 13-15 done: 6.8x gap at 20 questions (40,529 vs 5,992 peak), EM tied.
> Findings 10-12 in `eval.md`. Remaining lead: echo fix, untested.**

## DoD

1. [x] Agent solves ≥1 multi-objective task the naive version cannot (exam1: 2/4 vs 0/4; baseline drowned at 9.5K tokens; survived re-run)
2. [x] Peak context grows sub-linearly with steps (context_curve.png: folded 1.9–2.4K flat vs baseline up to 9.5K; −48–51% avg)
3. [x] `eval.md`: baseline vs folded tables + parse-failure accounting (9 findings, gates tracked, 26% breach flagged then fixed to 8%)

## Working rule

User types all logic (One Rule). AI designs, reviews, runs, writes docs/eval.
Readings are JIT — each paper arrives only when its code slot opens (~22 pp total).

---

# Phase H — True Horizon (~1 week, post-publish)

> Phase 3's original goal said *"50–100+ step tasks, context under ~10K tokens."*
> We never tested that: exams ran 7–10 steps (Finding 4 — nothing to compress at
> step 4 applies doubly at step 7). Phase H closes the gap: prove (or disprove)
> the flat line at the horizon the project was named for.

### Step 13 — 8-question exams (~2h + ~40 min runs)
- [x] Slice 16 fresh questions from `data/hotpot_dev.parquet` (same deterministic
  every-250th rule, offset shifted, same ≤4-word filter) → `data/hotpot16.json`
  (frozen, committed). Build 2 exams × 8 questions → `data/hotpot_exams8.json`
- [x] Extend `evaluate_multi` with an `exams_path` param (mechanical)
- [x] Run both arms at `max_steps=64` (expect 25–45 steps/run). Gate: folded
  EM ≥ baseline EM; peak ratio should now show ≥2× (Finding 4 predicts the
  gap widens with horizon — this is the first direct test)
  **Result: gates FAILED honestly — Finding 10 (apostrophe thrash: 18 of 21
  errors were one loop, `answer('...women's...')` rejected identically;
  2-line fix halves errors 21→10) + Finding 11 (baseline answered 8q mostly
  from prior knowledge in 5-8 steps, never drowned; cut 1.1-1.4x, not 2x)**

### Step 14 — the 100-step stress run (~1h typing-free, ~1.5h runs)
- [x] `exam20`: ALL 20 questions of hotpot20 in ONE run, `max_steps=100`.
  Expected 60–80 steps. This is the roadmap's advertised horizon, n=1 per arm —
  labeled a **demo, not statistics** (single-run variance is huge)
  **Result: 6.8x gap — baseline 40,529 peak / 39 steps vs folded 5,992 / 24
  steps; EM tied 0.55, F1 up 0.68→0.71; 22 folds, 0 malformed, 1 format error**
- [x] Predictions on record before running (falsifiable): baseline peak >25K
  tokens and/or EM collapse via question mix-ups; folded peak <10K and EM ≥ 0.4
  **>25K ✓ (40.5K) · <10K ✓ (6.0K) · EM≥0.4 ✓ (0.55) · collapse ✗ (0.55 held)**
- [x] Capture both arms with `capture_exam.py` (needs exam20 in its loader)

### Step 15 — prove it on paper (~2h)
- [x] Plots v2: `stress20.png` (step-axis, fold markers, official peaks dashed)
  + `gap_growth.png` (2.0x → 1.4x → 6.8x by task size)
- [x] README + eval.md update (Findings 10-12); roadmap progress log tick
- [x] Trace-viewer link in README; push

### Phase H DoD — settled honestly
1. ≥50-step run: NOT MET literally (the exam FINISHED at 39/24 steps); spirit
   met — **6.8x gap, folded under 6K the whole run**
2. 8q folded EM ≥ baseline, cut ≥2×: NOT MET (0.50-0.62 vs 0.62; cut
   1.1-1.4x) — Finding 11 explains why (baseline shortcuts on prior knowledge)
3. Format errors <20% at horizon: **MET on exam20 (1/24, 4%)**, not on exam8
   (10/25, 40%) — run-to-run variance; echo fix remains untested

### Honest risks
- n=2 exams at 8q and n=1 at 20q — statistical power is demo-grade only
- 100-step runs amplify any drift (answer leakage across questions, report
  staleness); if folded fails here, that's a FINDING, not a bug
- Baseline may pass 20q if the model is just good — then Finding 4 was wrong
  at this scale, also a finding
