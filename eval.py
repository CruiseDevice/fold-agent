"""Metrics: exact match + word-level F1 (MEM1 / HotpotQA conventions)."""

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip articles/punct, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = _ARTICLES.sub(" ", text)
    text = "".join(ch for ch in text if ch not in _PUNCT)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> int:
    """1 if normalized strings are identical, else 0."""
    return int(normalize(pred) == normalize(gold))


def word_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between normalized strings. 0.0 on no overlap.
    (both empty -> 1.0, MEM1 convention)"""
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    # smoke
    assert normalize("The Best-Known Cat!!") == "bestknown cat"
    assert normalize("a an the") == ""
    assert exact_match("The Answer.", "answer") == 1
    assert exact_match("answers", "answer") == 0
    assert word_f1("", "") == 1.0
    assert word_f1("paris", "london") == 0.0
    f1 = word_f1("the cat sat on a mat", "a cat sat on the hat")
    assert abs(f1 - 0.75) < 1e-9, f1  # articles gone: 4 vs 4 tokens, 3 shared
    print("eval.py smoke: OK")
