"""Tools for fold-agent: batch web search + page visit, both disk-cached."""
import hashlib
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:  # package was renamed from duckduckgo_search -> ddgs
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS
 


CACHE_DIR = Path("cache")
SEARCH_CACHE = CACHE_DIR / "search"
TOP_K = 5
PAGE_CACHE = CACHE_DIR / "pages"
PAGE_CHARS = 6000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _key(s: str) -> str:
    """Stable filename-safe cache key."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]



def web_search(queries: list[str]) -> str:
    """
    Batch search (IterResearch schema — one call, several queries).

    Returns ONE formatted string, one block per query:
        ## query
        title | url | snippet     (x TOP_K)
    Rules:
        - cache hit in SEARCH_CACHE/_key(q).json -> reuse, no network
        - "no results" is a valid outcome, printed as-is
        - "search failed: <reason>" only for real errors (exceptions)
    """
    out = []

    for query in queries:
        path = SEARCH_CACHE / (_key(query) + ".json")
        if path.exists():
            results = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                results = list(DDGS().text(query, max_results=TOP_K) or [])
            except Exception as e:
                out.append(f"## {query}\nsearch failed: {type(e).__name__}: {e}")
                continue

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

        block = [f"## {query}"]
        if not results:
            block.append("no results")
        for result in results:
            title = (result.get("title") or "").replace("\n", " ").strip()
            href = (result.get("href") or result.get("url") or "").strip()
            body = (result.get("body") or "").replace("\n", " ").strip()
            block.append(f"{title} | {href} | {body}")

        out.append("\n".join(block))

    return "\n".join(out)

def visit(url: str, goal: str = "") -> str:
    """Fetch a page, extract readable text, truncate. Cached by url hash.

    goal is accepted but NOT in the cache key (same page, any goal = one
    fetch). v1.5 upgrade: goal-conditioned LLM digest. Not yet.
    """
    path = PAGE_CACHE / (_key(url) + ".txt")
    if path.exists():
        return path.read_text(encoding="utf-8")

    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except Exception as e:
        return f"visit failed: {type(e).__name__}: {e}" # not cathed: may be transiet

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > PAGE_CHARS:
        text = text[:PAGE_CHARS] + f"\n[truncated at {PAGE_CHARS} chars]"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    # smoke: 1) one search (cached on 2nd run), 2) one visit
    print(web_search(["AgentFold paper arxiv"]))
    print("---")
    print(visit("https://arxiv.org/abs/2510.24699"))