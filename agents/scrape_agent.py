import requests
from bs4 import BeautifulSoup
from .base_agent import BaseAgent


class ScrapeAgent(BaseAgent):
    """Fetches a URL and extracts clean text content + metadata."""

    def __init__(self):
        super().__init__(name="ScrapeAgent")
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    def run(self, url: str) -> dict:
        print(f"[{self.name}] Fetching: {url}")
        resp = requests.get(url, headers=self.headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else "Untitled"

        # Collect headings and paragraphs as structured text
        chunks = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 20:
                if tag.name in ("h1", "h2", "h3", "h4"):
                    chunks.append(f"\n## {text}\n")
                else:
                    chunks.append(text)

        # Collect external links
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            label = a.get_text(strip=True)
            if href.startswith("http") and label:
                links.append({"label": label, "url": href})

        raw_text = "\n".join(chunks)

        print(f"[{self.name}] Scraped {len(raw_text)} chars, {len(links)} links")
        return {
            "url": url,
            "title": title,
            "raw_text": raw_text[:12000],  # cap to keep within LLM context
            "links": links[:30],
        }
