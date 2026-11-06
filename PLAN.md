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
- [ ] 4. `run.py` — one question end-to-end, full trace to disk (steps, tokens per step)
- [ ] 5. `eval.py` + `data/hotpot20.json` (HotpotQA dev slice, 20 questions) — EM + word-F1, peak context tokens (tiktoken, marked approx), steps, parse-failure count → baseline table in `eval.md`. Expect visible degradation >15 steps
  - 📖 before writing eval.md: **IterResearch §3.1 + §4.5** (~4 pp)

### Phase F — Folding (AgentFold mechanism) — ~6h
- [ ] 6. Output contract upgrade: `<think>/<fold>/<action>` tags; fold = one-line JSON `{"range":[k,t-1],"summary":"..."}`. Context becomes block list `[(start,end,text)]` (ordered partition), rendered per call
- [ ] 7. Fold application: retract blocks intersecting [k,t-1], insert new block. Granular (k=t-1) + deep consolidation (k<t-1). Malformed fold → no-op, counted
  - 📖 before typing: **AgentFold §3.2–3.5** (~5 pp) — you've now watched the baseline suffocate; it'll read differently
- [ ] 8. Re-run eval vs baseline. Gate: same-or-better accuracy, peak tokens cut ≥3×, parse-failure rate <20% of steps. Iterate prompt (when to deep-consolidate: sub-investigation done or dead end)

### Phase L — Long-horizon — ~5h
- [ ] 9. Evolving report section (IterResearch): findings / open questions / next steps, rewritten by the model each step, riding in the output contract
- [ ] 10. Multi-objective eval: 4–8 HotpotQA questions per task, semicolon-separated answers, per-answer EM (MEM1 §3.3). Budget 64 steps
  - 📖 before typing: **MEM1 §3.1, §3.3, Algorithm 1** (~4 pp)
- [ ] 11. Plots: peak-context-vs-steps + block-count-vs-steps (matplotlib). Flat-ish line = win

### Phase P — Polish — ~4h
- [ ] 12. Ablations (no-fold vs granular-only vs full) + README: architecture, benchmark table, token plot, one annotated fold trace, the convergent-evolution note (AlphaEvolve)

## DoD

1. Agent solves ≥1 multi-objective task the naive version cannot (context overflow)
2. Peak context grows sub-linearly with steps (plot proves it)
3. `eval.md`: baseline vs folded tables + parse-failure accounting

## Working rule

User types all logic (One Rule). AI designs, reviews, runs, writes docs/eval.
Readings are JIT — each paper arrives only when its code slot opens (~22 pp total).
