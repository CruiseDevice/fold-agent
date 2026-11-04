# fold-agent DESIGN

## Architecture (target state, Phase F)

```
                 ┌─────────────────────────────────────┐
                 │  CONTEXT (bounded, ~10K tokens)     │
                 │  ┌─────────┐  ┌──────────────────┐  │
 user question → │  │ Question │  │ Multi-Scale      │  │
                 │  │ (fixed)  │  │ State Summaries  │  │
                 │  └─────────┘  │ [s1][s2-5][s6-20]│  │
                 │               └──────────────────┘  │
                 │  ┌──────────────────────────────┐   │
                 │  │ Latest Interaction (verbatim) │   │
                 │  │ last action + last obs        │   │
                 │  └──────────────────────────────┘   │
                 └──────────────────┬──────────────────┘
                                    │
                          LLM decides TWO things each step:
                          1. FOLD DIRECTIVE: {"range":[k,t], "summary":"..."}
                          2. NEXT ACTION: tool call or final answer
                                    │
                          ┌─────────┴─────────┐
                          │ web_search(q)     │
                          │ visit(url, goal)  │
                          └───────────────────┘
```

Baseline (Phase B) is the same loop with NO bounded context: every
Thought/Action/Observation appended verbatim. We measure it dying, then fix it.

## Decisions

1. **Zero frameworks.** Hand-rolled ReAct loop. The loop is the lesson; crewai
   is not invited. `while True: parse → dispatch → observe → append`, break on
   Final Answer or max_steps.
2. **Raw `ollama.chat`, glm-5.3-flash:cloud.** Verified working. `/api/generate`
   returns empty on this model — always use chat.
3. **ReAct wire format.** Assistant emits `Thought:/Action:/Action Input:`;
   loop appends `Observation: ...` as the next user turn. Parse with regex;
   parse failure = observation saying so (honest strings, model self-corrects).
4. **Search: DDGS with disk cache from day one.** Eval reruns × ablations ×
   20 questions will rate-limit bare DDGS (news-agent lesson). `cache/`
   (gitignored) keyed by query hash.
5. **Token counting: tiktoken (cl100k), stated as approximate** in eval.
   Relative comparisons (baseline vs folded) are what matter; approximation is
   self-consistent across arms.
6. **Context as data structure, not string.** Phase F context = [question] +
   [summary blocks s_i..s_j] + [latest interaction]. Folding = retract blocks
   in fold_range, insert one new block. Never string-splice transcript text.
7. **Fold directive rides in the same output** as the action (one JSON object
   per step), so folding costs no extra LLM calls.
8. **Eval before optimize.** No folding code until the baseline table exists.
   A mechanism without a baseline is a demo, not a result.

## Failure modes to expect (named in advance)

- Model emits malformed fold JSON → treat as no-fold, count parse failures in eval
- Dead-end loops (same query repeatedly) → deep-consolidation prompt hint + repeated-query detection in code
- DDGS empty results ≠ failure → distinguish "no results" from "search failed: ..." (news-agent lesson)
- Multi-objective answers drift → require semicolon-separated answers in exact format (MEM1 trick)
