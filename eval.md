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
