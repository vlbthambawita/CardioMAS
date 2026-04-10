"""
LangGraph nodes — each function receives the full AgentState,
performs one job, and returns a partial state update.
"""

import os
import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState

# Shared LLM — all extract nodes reuse this instance
_llm = ChatOllama(model="gemma4:latest")

EXTRACT_SYSTEM = (
    "You are a precise data extraction agent. "
    "You receive raw scraped text from a website and extract structured information. "
    "Always respond in valid Markdown format as instructed — nothing else."
)


# ── Node 1: Scrape ────────────────────────────────────────────────────────────

def scrape_node(state: AgentState) -> dict:
    """Fetch the URL, strip noise, return clean text + metadata."""
    print(f"\n[ScrapeNode] Fetching: {state['url']}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(state["url"], headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ScrapeNode] ERROR: {e}")
        return {"error": f"Scrape failed: {e}"}

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else "Untitled"

    chunks = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 20:
            prefix = "\n## " if tag.name in ("h1", "h2", "h3", "h4") else ""
            chunks.append(f"{prefix}{text}")

    raw_text = "\n".join(chunks)[:12000]

    links = [
        {"label": a.get_text(strip=True), "url": a["href"]}
        for a in soup.find_all("a", href=True)
        if a["href"].startswith("http") and a.get_text(strip=True)
    ][:30]

    print(f"[ScrapeNode] Done — {len(raw_text)} chars, {len(links)} links")
    return {"title": title, "raw_text": raw_text, "links": links, "error": None}


# ── Node 2: Extract ───────────────────────────────────────────────────────────

def _llm_call(user_prompt: str) -> str:
    messages = [
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
    return _llm.invoke(messages).content.strip()


def extract_node(state: AgentState) -> dict:
    """Use Gemma 4 to extract summary, sections, and key facts."""
    print(f"[ExtractNode] Running Gemma 4 extraction...")

    title = state["title"]
    text = state["raw_text"]

    summary = _llm_call(
        f"Website title: {title}\n\nContent:\n{text[:4000]}\n\n"
        "Write a concise 3-5 sentence summary of this page. Return only the summary."
    )
    print(f"[ExtractNode] Summary done")

    sections = _llm_call(
        f"From the following website content, identify the main sections/topics covered.\n"
        f"Format as a Markdown bullet list with a one-line description for each.\n\n"
        f"Content:\n{text[:6000]}\n\nReturn only the Markdown list."
    )
    print(f"[ExtractNode] Sections done")

    key_facts = _llm_call(
        f"Extract up to 10 key facts or important data points from the following content.\n"
        f"Format as a Markdown bullet list.\n\nContent:\n{text[:6000]}\n\n"
        "Return only the bullet list."
    )
    print(f"[ExtractNode] Key facts done")

    return {"summary": summary, "sections": sections, "key_facts": key_facts}


# ── Node 3: Write ─────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:50]


def write_node(state: AgentState) -> dict:
    """Format the extracted data and save to a .md file."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    slug = _slugify(state["title"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{slug}_{timestamp}.md")

    lines = [
        f"# {state['title']}",
        "",
        f"> **Source:** {state['url']}",
        f"> **Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        state["summary"],
        "",
        "## Sections / Topics",
        "",
        state["sections"],
        "",
        "## Key Facts",
        "",
        state["key_facts"],
        "",
    ]

    if state.get("links"):
        lines += ["## Links Found", ""]
        for link in state["links"][:20]:
            lines.append(f"- [{link['label']}]({link['url']})")
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[WriteNode] Saved: {filepath}")
    return {"output_path": filepath}
