from .base_agent import BaseAgent

SYSTEM_PROMPT = """You are a precise data extraction agent.
You receive raw scraped text from a website and extract structured information.
Always respond in valid Markdown format as instructed — nothing else."""


class ExtractAgent(BaseAgent):
    """Uses Gemma 4 to extract structured information from raw scraped content."""

    def __init__(self):
        super().__init__(name="ExtractAgent")

    def run(self, scrape_result: dict) -> dict:
        url = scrape_result["url"]
        title = scrape_result["title"]
        raw_text = scrape_result["raw_text"]
        links = scrape_result["links"]

        print(f"[{self.name}] Extracting structured data...")

        summary = self._extract_summary(title, raw_text)
        sections = self._extract_sections(raw_text)
        key_facts = self._extract_key_facts(raw_text)

        return {
            "url": url,
            "title": title,
            "summary": summary,
            "sections": sections,
            "key_facts": key_facts,
            "links": links,
        }

    def _extract_summary(self, title: str, text: str) -> str:
        prompt = f"""Website title: {title}

Content:
{text[:4000]}

Write a concise 3-5 sentence summary of what this website/page is about.
Return only the summary text, no preamble."""
        return self.think(prompt, system=SYSTEM_PROMPT)

    def _extract_sections(self, text: str) -> str:
        prompt = f"""From the following website content, identify the main sections/topics covered.
Format as a Markdown list with a one-line description for each section.

Content:
{text[:6000]}

Return only the Markdown list, no preamble."""
        return self.think(prompt, system=SYSTEM_PROMPT)

    def _extract_key_facts(self, text: str) -> str:
        prompt = f"""Extract up to 10 key facts, data points, or important pieces of information
from the following website content. Format as a Markdown bullet list.

Content:
{text[:6000]}

Return only the bullet list, no preamble."""
        return self.think(prompt, system=SYSTEM_PROMPT)
