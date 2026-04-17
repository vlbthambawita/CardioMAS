from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from cardiomas.schemas.tools import ToolResult


def fetch_webpage(url: str) -> ToolResult:
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "CardioMAS/2.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "iframe"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else url
        text = " ".join(soup.get_text(separator=" ").split())
        return ToolResult(
            tool_name="fetch_webpage",
            ok=True,
            summary=f"Fetched webpage: {title}",
            data={"url": url, "title": title, "text": text[:4000]},
        )
    except Exception as exc:
        return ToolResult(
            tool_name="fetch_webpage",
            ok=False,
            summary="",
            error=str(exc),
        )
