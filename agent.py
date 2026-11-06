
import ast
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

@dataclass
class Step:
    n: int
    thought: str
    action: str
    observation: str
    prompt_tokens: int # tokens of the prompt that produced this step


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

    try:
        # Wrap arguments as a tuple of positional arguments.
        args = list(ast.literal_eval(f"({args_text},)"))
    except (ValueError, SyntaxError, TypeError):
        return None

    # Validate arguments for supported tools.
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
    

if __name__ == "__main__":
    # smoke: one 2-hop HotpotQA-style question
    r = run("What government position was held by the woman who portrated " \
    "Jyn Erso in Rogue One?")

    print("QUESTION:", r.question)
    print("ANSWER:", r.answer)

    print("STEPS:", len(r.steps))
    print("PEAK TOKENS:", r.peak_tokens)
    print("PARSE FAILURES:", r.parse_failures)

    for step in r.steps:
        print(f"\n--- Step {step.n} ---")
        print("Thought:", step.thought)
        print("Action:", step.action)
        print("Observation:", step.observation)
        print("Prompt tokens:", step.prompt_tokens)