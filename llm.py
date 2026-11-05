import ollama

MODEL = "glm-5.3-flash:cloud"


def chat(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """One model call. Returns the assistant's text, nothing else.
    messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    Exceptions crash loudly on purpose — the agent loop decides what to do.
    """
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    text = response["message"]["content"]
    return text.strip()

if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Reply with exactly: OK"}]))
