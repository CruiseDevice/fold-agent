
import ast
import json
import re
from dataclasses import dataclass, field

import tiktoken

from llm import chat
from tools import visit, web_search

MAX_STEPS = 100
ENCODING = tiktoken.get_encoding("cl100k_base") # approx counts, by design

def count_tokens(text: str) -> int:
    """Approximate token count (cl100k). Relative comparisons are the claims."""
    return len(ENCODING.encode(text))

SYSTEM = """
You are a research assistant that solves questions step by step.

You have three available actions:

1. web_search(["query1", "query2"])
    Search the web for information

2. visit("url", "goal")
    Visit a webpage to find specific information.

3. answer("final answer")
    Provide the final answer and stop.

You MUST respond using exactly two labeled lines:

Thought: <brief explanation of your next action>
Action: <one tool call>

Examples:

Thought: I need to identify the actress first.
Action: web_search(["who played Jyn Erso in Rogue One"])

Thought: I should inspect this source for more details.
Action: visit("https://example.com", "Find the actress's biography")

Thought: I have enough information to answer.
Action: answer("The answer is ...")

Rules:
- Use only one action per step.
- Use Python literal arguments, not JSON objects.
- Do not use Markdown code fences
- Never invent search results or sources
- answer() must contain ONLY the minimal answer phrase (a name, number,
  or short entity) - no sentences, no explanation, no restating the question.
- Use answer() when you have sufficient
- If you cannot verify the answer, acknowledge the unvertainty.
"""

SYSTEM_FOLD = """
You are a research assistant that solves questions step by step.

You have three available actions:

1. web_search(["query1", "query2"])
    Search the web for information

2. visit("url", "goal")
    Visit a webpage to find specific information.

3. answer("final answer")
    Provide the final answer and stop.

You MUST respond using exactly three tagged sections:

<reasoning>
Brief reasoning: what you know, what is still missing, and why this action comes next.
</reasoning>
<fold>
{"range": [k, t-1], "summary": "one dense paragraph covering steps k..t-1"}
</fold>
<action>one tool call, e.g. web_search(["query"])</action>

Example (first turn - nothing to fold yet, omit <fold>):

<reasoning>
The question asks for a government position, so I must first identify the person. I will search for the actress.
</reasoning>
<action>web_search(["who portrayed Jyn Erso in Rogue One"])</action>

Example (later turn - folding the previous step):

<reasoning>
The search confirmed the actress is Felicity Jones. Now I need to find the government position.
</reasoning>
<fold>
{"range": [2, 2], "summary": "Step 2 searched the Rogue One cast; confirmed Felicity Jones played Jyn Erso (Wikipedia)."}
</fold>
<action>web_search(["Felicity Jones government position"])</action>

Folding rules (the <fold> section compresses your own history):
- Your history is shown as blocks labeled [steps k-t]. Your fold REPLACES
  every block that intersects your stated range with one new block whose
  text is your summary.
- Fold EVERY turn. Normally fold just the previous step: if the last block
  is [steps 6-6] and you are on step 7, write {"range": [6, 6], ...}.
- DEEP-CONSOLIDATE when a sub-investigation is finished or a lead was a
  dead end: fold a wider range ([k, t-1] with k < t-1) into a shorter,
  denser summary.
- Summaries must preserve every fact you may still need: names, dates,
  numbers, URLs, answers to sub-questions. Drop raw search noise and
  boilerplate; note dead ends in one line ("dead end: <query or url>").
- Omit the <fold> section only on your first turn (nothing to fold yet).

Rules:
- Use only one action per step.
- Use Python literal arguments, not JSON objects.
- Do not use Markdown code fences.
- Never invent search results or sources.
- answer() must contain ONLY the minimal answer phrase (a name, number,
  or short entity) - no sentences, no explanation, no restating the question.
- Use answer() when you have sufficient information.
- If you cannot verify the answer, acknowledge the uncertainty.
"""

@dataclass
class Block:
    start: int  # first step this block covers
    end: int  # last step it covers
    text: str # the (possibly summarized) content

@dataclass
class Step:
    n: int
    thought: str
    action: str
    observation: str
    prompt_tokens: int # tokens of the prompt that produced this step
    fold: dict | None = None # well-formed fold emitted this step, if any
    fold_applied: bool = False # False + fold set -> range rejected (no-op)
    fold_malformed: bool = False # <fold> tag present but unparseable/invalid
    raw: str = "" # verbatim model output (run_folded only; prompt-iteration aid)


@dataclass
class RunResult:
    question: str
    answer: str # "" if max_steps exhausted
    steps: list = field(default_factory=list)

    @property
    def peak_tokens(self) -> int:
        return max((s.prompt_tokens for s in self.steps), default=0)

    @property
    def parse_failures(self) -> int:
        return sum(1 for s in self.steps if s.observation.startswith("FORMAT_ERROR"))

    @property
    def folds_applied(self) -> int:
        return sum(1 for s in self.steps if s.fold_applied)

    @property
    def fold_noops(self) -> int:
        # well-formed fold whose range was rejected at application time
        return sum(1 for s in self.steps if s.fold is not None and not s.fold_applied)


_REASON_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
_FOLD_RE = re.compile(r"<fold>(.*?)</fold>", re.DOTALL)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_CALL_RE = re.compile(r"([a-zA-Z_]\w*)\((.*)\)", re.DOTALL)


def _parse_args(tool_name: str, args_text: str):
    """Literal-eval a tool call's arguments and validate per tool.
    Return the args list, or None if not evaluable/invalid. Unknown tools
    pass through (the run loop reports 'unknown action', as in baseline).
    """
    try:
        # Wrap arguments as a tuple of positional arguments.
        args = list(ast.literal_eval(f"({args_text},)"))
    except (ValueError, SyntaxError, TypeError):
        return None

    if tool_name == "web_search":
        if (
            len(args) != 1
            or not isinstance(args[0], list)
            or not all(isinstance(q, str) for q in args[0])
        ):
            return None

    elif tool_name == "visit":
        if not (1 <= len(args) <= 2):
            return None
        if not all(isinstance(arg, str) for arg in args):
            return None

    elif tool_name == "answer":
        if len(args) != 1 or not isinstance(args[0], str):
            return None

    return args


def _parse_fold(raw: str):
    """Validate the JSON inside a <fold> tag.
    Return {"range": [k, t-1], "summary": str} or None if malformed.
    (lenient on extra keys; bool is rejected even though bool <: int)
    """
    try:
        candidate = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if (
        isinstance(candidate, dict)
        and {"range", "summary"} <= candidate.keys()
        and isinstance(candidate["range"], list)
        and len(candidate["range"]) == 2
        and all(
            isinstance(x, int) and not isinstance(x, bool)
            for x in candidate["range"]
        )
        and isinstance(candidate["summary"], str)
    ):
        return candidate

    return None


def parse_output(text: str):
    """Return (think, fold_or_None, tool_name, args) or None if unparseable.
    Contract - the model replies with tagged sections (fold optional):
        <reasoning>reasoning</reasoning>
        <fold>{"range": [k, t-1], "summary": "..."}</fold>
        <action>web_search(["q1"]) / visit("url", "goal") / answer("text")</action>
       (tag is <reasoning>, NOT <think>: the GLM serving layer strips native
       <think> blocks from content before we see them.)
       Malformed fold -> fold=None (recorded, never fatal). Missing/unparseable
       action -> None (FORMAT_ERROR, same as baseline).
    """
    think_match = _REASON_RE.search(text)
    fold_match = _FOLD_RE.search(text)
    action_match = _ACTION_RE.search(text)

    # Action is the only REQUIRED section.
    if not action_match:
        return None

    think = think_match.group(1).strip() if think_match else ""

    fold = None
    if fold_match:
        fold = _parse_fold(fold_match.group(1))

    # Action tag content parsed exactly like baseline: tool call + literals.
    call = _CALL_RE.search(action_match.group(1))
    if not call:
        return None

    tool_name, args_text = call.groups()
    args = _parse_args(tool_name, args_text)
    if args is None:
        return None

    return think, fold, tool_name, args

def parse_action(text: str):
    """Return (thought, tool_name, args_list) or None if unparseable.
    Expected model output (exactly two label lines, args are Python
    literals):
        Thought: <why this action>
        Action: web_search(["q1", "q2"]) | visit("url", "goal") | answer("text")
    """
    pattern = (
        r"Thought:[ \t]*([^\r\n]*)\r?\n"
        r"Action:[ \t]*([a-zA-Z_]\w*)\((.*)\)"
    )

    match = re.search(
        pattern,
        text.strip(),
        flags=re.DOTALL
    )

    if not match:
        return None

    thought, tool_name, args_text = match.groups()

    args = _parse_args(tool_name, args_text)
    if args is None:
        return None

    return thought.strip(), tool_name, args

def run(question: str, max_steps: int = MAX_STEPS) -> RunResult:
    # TODO: messages = [system, user question]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question}
    ]

    # TODO: result = RunResult(question "")
    result = RunResult(question=question, answer="")

    for t in range(1, max_steps + 1):
        # Step 1: Count prompt tokens
        prompt = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        )

        tokens = count_tokens(prompt)

        # Step 2: Ask LLM what to do next
        out = chat(messages)

        # Step 3: Parse the model's response
        parsed = parse_action(out)

        finished = False

        if parsed is None:
            thought = ""
            action = ""

            obs = (
                "FORMAT_ERROR: Respond using exactly two lines:\n"
                "Thought: <reason>\n"
                "Action: web_search([\"query\"]) | "
                "visit(\"url\", \"goal\") | "
                "answer(\"text\")"
            )
        else:
            thought, tool_name, args = parsed

            action = (
                out.strip()
                .partition("\nAction:")[2]
                .strip()
            )

            obs = ""

            # Step 4: Execute action
            try:
                if tool_name == "web_search":
                    obs = str(web_search(args[0]))

                elif tool_name == "visit":
                    goal = args[1] if len(args) > 1 else ""
                    obs = str(visit(args[0], goal))

                elif tool_name == "answer":
                    result.answer = args[0]
                    finished = True

                else:
                    obs = (
                        f"unknown action '{tool_name}'. "
                        "valid: web_search, visit, answer"
                    )

            except Exception as e:
                obs = f"TOOL_ERROR: {type(e).__name__}: {e}"

        # Step 5: Record the step
        result.steps.append(
            Step(
                n=t,
                thought=thought,
                action=action,
                observation=obs,
                prompt_tokens=tokens
            )
        )

        if finished:
            break

        # Step 6: Feed the observation back to the LLM
        messages.extend([
            {
                "role": "assistant",
                "content": out
            },
            {
                "role": "user",
                "content": (
                    f"[HINT: {max_steps - t} steps left]\n"
                    f"Observation: {obs}"
                )
            }
        ])

    return result


def apply_fold(blocks: list[Block], fold: dict, t: int):
    """Retract blocks intersecting the fold range, insert one summary block.
    Called at step t, so a contract-conforming fold [k, t-1] covers exactly
    the history the model just saw. Returns (new_blocks, applied);
    range outside [1, t-1] -> unchanged (no-op, counted by the caller).

    The replacement block spans the UNION of the retracted blocks (never
    just [k, last]): blocks are the model's own consolidation units, and
    this keeps the labels an exact ordered partition of 1..t-1 even when
    the model folds a sub-range of an already-folded block.
    """
    k, last = fold["range"]
    if not (1 <= k <= last <= t - 1):
        return blocks, False

    retracted = [b for b in blocks if b.start <= last and b.end >= k]
    if not retracted:
        return blocks, False

    kept = [b for b in blocks if not (b.start <= last and b.end >= k)]
    start = min(k, retracted[0].start)
    end = max(last, retracted[-1].end)
    # insert at the position of the first retracted block
    idx = sum(1 for b in kept if b.end < k)
    return kept[:idx] + [Block(start=start, end=end, text=fold["summary"])] + kept[idx:], True


def render_messages(question: str, blocks: list[Block], hint: str,
                    system: str = SYSTEM_FOLD) -> list[dict]:
    """AgentFold shape: system / question / history-as-blocks / latest + hint.
    Empty blocks (first step) -> no HISTORY section, just question + hint."""
    if blocks:
        history = "\n\n".join(f"[steps {b.start}-{b.end}]\n{b.text}" for b in blocks)
        user = question + "\n\nHISTORY:\n" + history + "\n\n" + hint
    else:
        user = question + "\n\n" + hint
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run_folded(question: str, max_steps: int = MAX_STEPS) -> RunResult:
    """Phase F agent: same tools and loop shape as run(), but history lives
    as an ordered partition of blocks; each turn the model may fold
    (compress) part of its own history before acting."""
    blocks: list[Block] = []
    result = RunResult(question=question, answer="")

    for t in range(1, max_steps + 1):
        # Step 1: render the folded context and count its tokens
        hint = (
            f"[HINT: step {t} of {max_steps}; "
            f"{max_steps - t} steps remain after this one]"
        )
        messages = render_messages(question, blocks, hint)

        prompt = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        )
        tokens = count_tokens(prompt)

        # Step 2: Ask LLM what to do next
        out = chat(messages)

        # Step 3: Parse the model's response
        parsed = parse_output(out)

        finished = False
        fold = None

        if parsed is None:
            thought = ""
            action = ""

            obs = (
                "FORMAT_ERROR: Respond using exactly three tagged sections:\n"
                "<reasoning>reasoning</reasoning>\n"
                '<fold>{"range": [k, t-1], "summary": "..."}</fold>\n'
                '<action>web_search(["query"]) | visit("url", "goal") | '
                'answer("text")</action>'
            )
        else:
            thought, fold, tool_name, args = parsed

            action_match = _ACTION_RE.search(out)
            action = action_match.group(1).strip() if action_match else ""

            obs = ""

            # Step 4: Execute action
            try:
                if tool_name == "web_search":
                    obs = str(web_search(args[0]))

                elif tool_name == "visit":
                    goal = args[1] if len(args) > 1 else ""
                    obs = str(visit(args[0], goal))

                elif tool_name == "answer":
                    result.answer = args[0]
                    finished = True

                else:
                    obs = (
                        f"unknown action '{tool_name}'. "
                        "valid: web_search, visit, answer"
                    )

            except Exception as e:
                obs = f"TOOL_ERROR: {type(e).__name__}: {e}"

        # Step 5: Apply the fold BEFORE appending step t — a contract fold
        # [k, t-1] covers history the model saw, not the step it just made.
        fold_applied = False
        if fold is not None:
            blocks, fold_applied = apply_fold(blocks, fold, t)

        blocks.append(
            Block(
                start=t,
                end=t,
                text=(
                    f"Thought: {thought}\n"
                    f"Action: {action}\n"
                    f"Observation: {obs}"
                ),
            )
        )

        # Step 6: Record the step
        result.steps.append(
            Step(
                n=t,
                thought=thought,
                action=action,
                observation=obs,
                prompt_tokens=tokens,
                fold=fold,
                fold_applied=fold_applied,
                fold_malformed=fold is None and _FOLD_RE.search(out) is not None,
                raw=out,
            )
        )

        if finished:
            break

    return result


if __name__ == "__main__":
    # smoke: one 2-hop HotpotQA-style question, folded agent
    r = run_folded("What government position was held by the woman who portrated " \
    "Jyn Erso in Rogue One?")

    print("QUESTION:", r.question)
    print("ANSWER:", r.answer)

    print("STEPS:", len(r.steps))
    print("PEAK TOKENS:", r.peak_tokens)
    print("PARSE FAILURES:", r.parse_failures)
    print("FOLDS APPLIED:", r.folds_applied)
    print("FOLD NO-OPS:", r.fold_noops)

    for step in r.steps:
        print(f"\n--- Step {step.n} ---")
        print("Thought:", step.thought)
        if step.fold is not None:
            print("Fold:", step.fold["range"],
                  "(applied)" if step.fold_applied else "(NO-OP)")
        elif step.fold_malformed:
            print("Fold: <malformed>")
        print("Action:", step.action)
        print("Observation:", step.observation)
        print("Prompt tokens:", step.prompt_tokens)