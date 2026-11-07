# fold-agent — eval

## Baseline (Phase B, naive append-only ReAct)

Run 2026-11-06, `data/hotpot20.json` (frozen slice, distractor validation),
max_steps=20, glm-5.3-flash:cloud, temperature 0.2. Warm disk cache.

| metric | value |
|---|---|
| EM | **0.60** (12/20) |
| word-F1 | **0.70** |
| avg steps | 3.5 |
| avg peak context tokens | 2,786 |
| runs with ≥1 parse failure | 4/20 |
| unanswered (max_steps) | 0/20 |

Per-question rows: `results_baseline.json` (committed).

## Finding 1 — format was the floor, not research (v0 → v1)

First full run (kept as `results_baseline_v0.json`) scored EM 0.05 / F1 0.14
while ~7 answers visibly contained the gold entity buried in a sentence —
30-word preds cap word-F1 near 0.06 regardless of correctness. Adding one
SYSTEM rule ("answer() must contain ONLY the minimal answer phrase") moved
EM 0.05 → 0.60, F1 0.14 → 0.70 with zero loop changes. Lesson recorded:
**before tuning research ability, check that the metric is measuring
research.**

## Finding 2 — near-misses are entity-granularity, not errors

`Istanbul` vs `Istanbul, Turkey` (F1 0.67), `Danny Elfman` vs `Elfman`
(0.67), `Ghana's national football team` vs `Ghanaian national team`
(0.44) — semantically right, string-penalized. EM understates ability;
F1 is the honest headline. 5/20 are genuine research misses (wrong entity
or refused ambiguity), 1 of which answered after only 2 steps (premature).

## Finding 3 — parse failures are stable, not fatal

4/20 runs had ≥1 FORMAT_ERROR; all recovered and finished. Failure mode:
clusters right after large observations (the model mirrors messy input
format). Rate is under the 20% gate; watch it in Phase F where the output
contract gets the fold tag added.

## Caveats

- n=20, one seed — differences <15pp between arms are noise at this n
- tokens are tiktoken cl100k approximations (relative comparisons only)
- live web under disk cache: arms must share the same cache so
  observations are identical across baseline/folded comparisons
- HotpotQA gold includes numeric answers ("12,272") — brutal for EM,
  kept anyway (it's the benchmark's nature)

## Next arm

Phase F (folding) targets: EM ≥ 0.60, peak tokens ≤ ~900 avg (≥3x cut),
parse-failure rate still <20% of steps.

## Folded arm (Phase F, steps 6-8)

Same 20 questions, same cache, max_steps=20. `run_folded()` + SYSTEM_FOLD
(tagged contract, fold directive each turn). Per-question rows:
`results_folded.json`.

| metric | baseline | folded | delta |
|---|---|---|---|
| EM | 0.60 | **0.60** | held |
| word-F1 | 0.70 | **0.71** | +0.01 |
| avg steps | 3.5 | 3.8 | +0.3 |
| avg peak tokens | 2,786 | **1,858** | **-33%** |
| FORMAT_ERRORs (total steps) | 4 | 9 | watch |
| FORMAT_ERROR rate (of 76 steps) | ~6% | ~12% | under 20% gate |

Same-question agreement: 19/20 preds identical or near-identical to baseline
(Q1 `Istanbul` both arms; Q15 slightly longer tail). The one flip: Q2
(`a scholar`) folded got RIGHT (baseline missed). Q17 `Ghana` vs
`Ghanaian national team` — folded answered tighter but EM-unequal.

### Finding 4 — the ≥3x token gate was miscalibrated for this horizon

Gate said ≤900 avg peak tokens. Folded achieved 1858 (-33%). Why the gate
missed: these 2-hop questions finish in ~3.8 steps — history is only 1-2
observations when the run ends, so there is almost nothing to compress.
Folding's ceiling here is the fixed overhead (SYSTEM_FOLD prompt + question
+ current observation), which no context manager can shrink. AgentFold's
92%-smaller-context claim is measured at turn 100, not turn 4.

Where the 3x cut CAN appear: long horizons (30-64 steps) = Phase L
multi-objective tasks, which is exactly the project DoD #1. The 2-hop eval
remains the accuracy-parity check: folding held EM and F1 while active,
with 33% less context — the mechanism does not hurt, which is the
prerequisite for scaling it.

### Finding 5 — the 3-tag contract costs format reliability

FORMAT_ERRORs rose 4 → 9 (12% of steps, still under the 20% gate). Cost of
the richer contract; all runs recovered. If Phase L pushes it over 20%, the
fix candidates are: fewer worked examples of the fold format, or a
FORMAT_ERROR observation that echoes the model's own malformed attempt.

