from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def search_arxiv(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search arXiv for papers matching the query.
    Returns list of results with title, abstract, url, authors."""
    try:
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        base = "https://export.arxiv.org/api/query?"
        params = urllib.parse.urlencode({
            "search_query": query,
            "max_results": max_results,
            "sortBy": "relevance",
        })
        url = base + params
        with urllib.request.urlopen(url, timeout=15) as resp:
            content = resp.read().decode()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(content)
        results = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]  # type: ignore
            results.append({
                "title": title_el.text.strip() if title_el is not None else "",
                "abstract": summary_el.text.strip()[:500] if summary_el is not None else "",
                "url": id_el.text.strip() if id_el is not None else "",
                "authors": authors[:5],
            })
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}


@tool
def fetch_webpage(url: str) -> dict[str, Any]:
    """Fetch and extract text content from a web page.
    Returns dict with 'title' and 'text'."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (compatible; CardioMAS/1.0)"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "iframe"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title else ""
        text = " ".join(soup.get_text(separator=" ").split())[:8000]
        return {"title": title, "text": text, "url": url}
    except Exception as e:
        return {"title": "", "text": "", "error": str(e)}


@tool
def read_pdf(path_or_url: str) -> dict[str, Any]:
    """Extract text from a PDF file (local path or URL).
    Returns dict with 'text' and 'num_pages'."""
    import io
    try:
        from pypdf import PdfReader
        import requests

        if path_or_url.startswith("http"):
            resp = requests.get(path_or_url, timeout=60)
            resp.raise_for_status()
            data = io.BytesIO(resp.content)
        else:
            data = open(path_or_url, "rb")  # type: ignore

        reader = PdfReader(data)
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts)
        return {
            "text": text[:12000],
            "num_pages": len(reader.pages),
            "full_length": len(text),
        }
    except Exception as e:
        return {"text": "", "num_pages": 0, "error": str(e)}
