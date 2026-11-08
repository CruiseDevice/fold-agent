
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

SYSTEM_MULTI = """
You are a research assistant solving SEVERAL numbered questions in one
session. Answer them one at a time, in order. Solve each sub-question with
web tools, record it in your report, then move to the next.

You have three available actions:

1. web_search(["query1", "query2"])
    Search the web for information

2. visit("url", "goal")
    Visit a webpage to find specific information.

3. answer("final answer")
    Submit answers for ALL questions and stop.

You MUST respond using exactly four tagged sections:

<reasoning>
What you know, what is still missing, and why this action comes next.
</reasoning>
<fold>
{"range": [k, t-1], "summary": "one dense paragraph covering steps k..t-1"}
</fold>
<report>
One line per question, always all of them:
Q1: <answer> (done)   or   Q1: pending - <what is missing>
</report>
<action>one tool call</action>

Example (3-question session, step 5):

<reasoning>
Q1 is answered (Istanbul). Q2 needs the shared profession of two writers;
I have Mencken's page cached in history, now searching Camus.
</reasoning>
<fold>
{"range": [2, 4], "summary": "Steps 2-4 established Q1: both landmarks are
in Istanbul (done). Began Q2: Mencken was a journalist/editor; need Camus."}
</fold>
<report>
Q1: Istanbul (done)
Q2: pending - checking Camus's profession
Q3: pending
</report>
<action>web_search(["Albert Camus profession journalist"])</action>

Folding rules:
- Your history is shown as blocks labeled [steps k-t]. Your fold REPLACES
  every intersecting block with one new block containing your summary.
- Fold EVERY turn (normally just the previous step).
- DEEP-CONSOLIDATE each question's investigation once its answer is in the
  report: compress that whole question's steps to 1-2 lines ("Q1: Istanbul,
  confirmed via [source]") and discard the raw search noise.
- Summaries must preserve facts you may still need: names, dates, numbers,
  URLs, answers to sub-questions. Note dead ends in one line.
- Omit <fold> only on your first turn.

Answer format (submit ONLY when every question is answered or clearly
stuck): answer("ans1; ans2; ans3") - exactly one part per question, in
order, separated by "; ". Use "?" for a part you could not resolve.
Each part: ONLY the minimal answer phrase - no sentences.

Rules:
- Use only one action per step.
- Use Python literal arguments, not JSON objects.
- Do not use Markdown code fences.
- Never invent search results or sources.
- Keep the report current every turn; it is your only durable memory of
  finished questions (history gets folded away).
"""

SYSTEM_MULTI_BASELINE = """
You are a research assistant solving SEVERAL numbered questions in one
session. Answer them one at a time, in order, using web tools.

You have three available actions:

1. web_search(["query1", "query2"])
    Search the web for information

2. visit("url", "goal")
    Visit a webpage to find specific information.

3. answer("final answer")
    Submit answers for ALL questions and stop.

You MUST respond using exactly two labeled lines:

Thought: <brief explanation - you may note which questions are already
answered and their answers, to keep track>
Action: <one tool call>

Rules:
- Use only one action per step.
- answer() must submit ALL questions at once: answer("ans1; ans2; ...") -
  exactly one part per question, in order, separated by "; ". Use "?" for
  a part you could not resolve. Each part: ONLY the minimal answer phrase
  (a name, number, or short entity) - no sentences.
- Use Python literal arguments, not JSON objects.
- Do not use Markdown code fences.
- Never invent search results or sources.
"""

SYSTEM_REPORT_ONLY = """
You are a research assistant solving SEVERAL numbered questions in one
session. Answer them one at a time, in order. Solve each sub-question with
web tools, record it in your report, then move to the next.

You have three available actions:

1. web_search(["query1", "query2"])
    Search the web for information

2. visit("url", "goal")
    Visit a webpage to find specific information.

3. answer("final answer")
    Submit answers for ALL questions and stop.

You MUST respond using exactly three tagged sections:

<reasoning>
What you know, what is still missing, and why this action comes next.
</reasoning>
<report>
One line per question, always all of them:
Q1: <answer> (done)   or   Q1: pending - <what is missing>
</report>
<action>one tool call</action>

Your full history stays visible to you; use the report to keep track of
answers across questions.

Answer format (submit ONLY when every question is answered or clearly
stuck): answer("ans1; ans2; ans3") - exactly one part per question, in
order, separated by "; ". Use "?" for a part you could not resolve.
Each part: ONLY the minimal answer phrase - no sentences.

Rules:
- Use only one action per step.
- Use Python literal arguments, not JSON objects.
- Do not use Markdown code fences.
- Never invent search results or sources.
"""

SYSTEM_FOLD_ONLY = """
You are a research assistant solving SEVERAL numbered questions in one
session. Answer them one at a time, in order. Solve each sub-question with
web tools, then move to the next.

You have three available actions:

1. web_search(["query1", "query2"])
    Search the web for information

2. visit("url", "goal")
    Visit a webpage to find specific information.

3. answer("final answer")
    Submit answers for ALL questions and stop.

You MUST respond using exactly three tagged sections:

<reasoning>
What you know, what is still missing, and why this action comes next.
</reasoning>
<fold>
{"range": [k, t-1], "summary": "one dense paragraph covering steps k..t-1"}
</fold>
<action>one tool call</action>

Folding rules:
- Your history is shown as blocks labeled [steps k-t]. Your fold REPLACES
  every intersecting block with one new block containing your summary.
- Fold EVERY turn (normally just the previous step).
- DEEP-CONSOLIDATE each question's investigation once it is answered:
  compress that whole question's steps to 1-2 lines and discard the raw
  search noise.
- Summaries must preserve facts you may still need: names, dates, numbers,
  URLs, answers to sub-questions. Note dead ends in one line.
- Omit <fold> only on your first turn.

Answer format (submit ONLY when every question is answered or clearly
stuck): answer("ans1; ans2; ans3") - exactly one part per question, in
order, separated by "; ". Use "?" for a part you could not resolve.
Each part: ONLY the minimal answer phrase - no sentences.

Rules:
- Use only one action per step.
- Use Python literal arguments, not JSON objects.
- Do not use Markdown code fences.
- Never invent search results or sources.
"""

SYSTEM_MULTI_V2 = """
You are a research assistant solving SEVERAL numbered questions in one
session. Answer them one at a time, in order. Solve each sub-question with
web tools, record it in your report, then move to the next.

You have three available actions:

1. web_search(["query1", "query2"])
    Search the web for information

2. visit("url", "goal")
    Visit a webpage to find specific information.

3. answer("final answer")
    Submit answers for ALL questions and stop.

FORMAT CHECKLIST - all four tags, every turn, in this exact order:
<reasoning>...</reasoning>
<fold>{"range": [k, t-1], "summary": "..."}</fold>
<report>...</report>
<action>one tool call</action>
(Omit only <fold> on your first turn.)

<report> is one line per question, always all of them:
Q1: <answer> (done)   or   Q1: pending - <what is missing>

Example (3-question session, step 5):

<reasoning>
Q1 is answered (Istanbul). Q2 needs the shared profession of two writers;
searching Camus now.
</reasoning>
<fold>
{"range": [2, 4], "summary": "Steps 2-4 established Q1: both landmarks are
in Istanbul (done). Began Q2: Mencken was a journalist/editor; need Camus."}
</fold>
<report>
Q1: Istanbul (done)
Q2: pending - checking Camus's profession
Q3: pending
</report>
<action>web_search(["Albert Camus profession journalist"])</action>

Folding rules:
- Your history is shown as blocks labeled [steps k-t]. Your fold REPLACES
  every intersecting block with one new block containing your summary.
- Fold EVERY turn (normally just the previous step).
- DEEP-CONSOLIDATE each question's investigation once its answer is in the
  report: compress that question's steps to 1-2 lines, discard raw noise.
- Summaries must preserve facts you may still need: names, dates, numbers,
  URLs, answers to sub-questions. Note dead ends in one line.

Answer format: answer("ans1; ans2; ans3") - one part per question, in order,
separated by "; ", "?" for unresolved. Each part: ONLY the minimal answer
phrase - no sentences. Use double quotes for the whole argument (single
quotes break on answers containing apostrophes).

Rules:
- Use only one action per step.
- Use Python literal arguments, not JSON objects.
- Do not use Markdown code fences.
- Never invent search results or sources.
- Keep the report current every turn; it is your only durable memory of
  finished questions (history gets folded away).
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
    report: str = "" # the evolving report as of this step ("" until first emitted)
    workspace_after: list = field(default_factory=list) # block snapshot after this step (post-fold, post-append)


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
_REPORT_RE = re.compile(r"<report>(.*?)</report>", re.DOTALL)
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
        # model quoted a string containing the quote char, e.g.
        # answer('...women's basketball...') - recover via outer-quote match
        m = re.fullmatch(r"\s*(['\"])(.*)\1\s*", args_text, re.DOTALL)
        if not m:
            return None
        args = [m.group(2)]

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
    report_match = _REPORT_RE.search(text)

    # Action is the only REQUIRED section.
    if not action_match:
        return None

    think = think_match.group(1).strip() if think_match else ""

    fold = None
    if fold_match:
        fold = _parse_fold(fold_match.group(1))

    report = report_match.group(1).strip() if report_match else ""

    # Action tag content parsed exactly like baseline: tool call + literals.
    call = _CALL_RE.search(action_match.group(1))
    if not call:
        return None

    tool_name, args_text = call.groups()
    args = _parse_args(tool_name, args_text)
    if args is None:
        return None

    return think, fold, tool_name, args, report

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

def run(question: str, max_steps: int = MAX_STEPS,
        system: str = SYSTEM) -> RunResult:
    # TODO: messages = [system, user question]
    messages = [
        {"role": "system", "content": system},
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
                    system: str = SYSTEM_FOLD, report: str = "") -> list[dict]:
    """AgentFold shape: system / question / report / history-as-blocks / hint.
    Empty blocks (first step) -> no HISTORY section. Empty report -> omitted
    (single-question mode never needs one)."""
    parts = [question]
    if report:
        parts.append("YOUR REPORT (rewrite it every turn; history gets folded away):\n" + report)
    if blocks:
        history = "\n\n".join(f"[steps {b.start}-{b.end}]\n{b.text}" for b in blocks)
        parts.append("HISTORY:\n" + history)
    parts.append(hint)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def run_folded(question: str, max_steps: int = MAX_STEPS,
               system: str = SYSTEM_FOLD) -> RunResult:
    """Phase F/L agent: history lives as an ordered partition of blocks;
    each turn the model may fold (compress) part of its own history, and
    keeps an evolving <report> (multi-question memory, if the contract asks
    for one)."""
    blocks: list[Block] = []
    report = ""
    result = RunResult(question=question, answer="")

    for t in range(1, max_steps + 1):
        # Step 1: render the folded context and count its tokens
        hint = (
            f"[HINT: step {t} of {max_steps}; "
            f"{max_steps - t} steps remain after this one]"
        )
        messages = render_messages(question, blocks, hint, system=system,
                                    report=report)

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
            thought, fold, tool_name, args, new_report = parsed
            if new_report:
                report = new_report

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
                report=report,
                workspace_after=[*blocks],
            )
        )

        if finished:
            break

    return result


if __name__ == "__main__":
    # smoke: 3-question multi-objective exam with the evolving report
    exam = (
        "Answer each question. 1. In what city are the Nusretiye Clock Tower "
        "and the Eski Imaret Mosque? 2. What profession do H. L. Mencken and "
        "Albert Camus have in common? 3. Who was the lead character in the "
        "1960s sitcom Get Smart?"
    )
    r = run_folded(exam, max_steps=24, system=SYSTEM_MULTI)

    print("EXAM:", exam)
    print("\nANSWER:", r.answer)
    print("steps:", len(r.steps), "| peak:", r.peak_tokens,
          "| parse failures:", r.parse_failures,
          "| folds applied:", r.folds_applied, "| no-ops:", r.fold_noops)
    print("\nFINAL REPORT:\n" + (r.steps[-1].report if r.steps else "(none)"))
    for s in r.steps:
        print(f"{s.n:>2}. [{s.prompt_tokens:>5} tok] {s.action[:60]}")