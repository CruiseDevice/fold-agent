# fold-agent — context-folding long-horizon research agent

> Flagship project from `ai-agent-project-roadmap.md` Phase 3.
> Builds on AgentFold (arXiv:2510.24699), IterResearch (2511.07327), MEM1 (2506.15841) — engineering only, no training.

## Goal

A web-research agent that completes 50–100+ step tasks while keeping its context
under ~10K tokens — by **folding** history instead of appending it — proven with
measured baselines and ablations.

## Phases (the roadmap's weeks 5–8)

### Phase B — Baseline (naive, append-everything)
- [ ] 1. `llm.py` — raw `ollama.chat` wrapper, no frameworks (~30 lines)
- [ ] 2. `tools.py` — `web_search` (DDGS + disk cache), `visit(url, goal)` (requests + bs4), honest failure strings (~70 lines)
- [ ] 3. `agent.py` — hand-rolled ReAct loop: Thought/Action/Observation cycle, append everything (~110 lines) ← the loop lesson
- [ ] 4. `run.py` — one question end-to-end, trace to disk
- [ ] 5. `eval.py` + 20 multi-hop questions (`data/hotpot20.json` from HotpotQA dev) → baseline table in `eval.md`: accuracy, peak tokens, steps. Expect failure >25 steps

### Phase F — Folding (the AgentFold mechanism)
- [ ] 6. Fold directive in output contract: `{"fold_range": [k, t-1], "fold_summary": "..."}` before each action
- [ ] 7. Context rewrite: retract summary blocks in range, replace with one; granular (last step) vs deep consolidation (whole sub-investigation)
- [ ] 8. Re-run eval. Iterate prompt. Target: same/better accuracy, peak tokens cut 3–5×

### Phase L — Long-horizon
- [ ] 9. Evolving report section (IterResearch): findings/open questions/next steps, updated each step
- [ ] 10. Multi-objective eval: 4–8 questions concatenated into one 50–100 step task
- [ ] 11. Plot peak-context vs steps; flat-ish line = win

### Phase P — Polish
- [ ] 12. Ablations: no-fold vs granular-only vs full; README with architecture + benchmark table + example fold traces

## DoD

1. Agent solves ≥1 multi-objective task the naive version cannot (context overflow)
2. Peak context grows sub-linearly with steps
3. `eval.md` with baseline vs folded tables + one plot

## Working rule

User types all logic (One Rule); AI designs, reviews, runs, writes docs/eval scaffolding.
Stack: `glm-5.3-flash:cloud` via raw ollama chat · ddgs · requests/bs4 · tiktoken (approx counts, stated in eval) · no agent frameworks.
