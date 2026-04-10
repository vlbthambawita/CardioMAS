import os
import re
from datetime import datetime
from .base_agent import BaseAgent


class WriterAgent(BaseAgent):
    """Writes extracted data to structured .md files."""

    def __init__(self, output_dir: str = "output"):
        super().__init__(name="WriterAgent")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, extracted: dict) -> str:
        slug = self._slugify(extracted["title"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{slug}_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)

        content = self._build_markdown(extracted)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[{self.name}] Saved: {filepath}")
        return filepath

    def _build_markdown(self, data: dict) -> str:
        lines = []
        lines.append(f"# {data['title']}")
        lines.append("")
        lines.append(f"> **Source:** {data['url']}")
        lines.append(f"> **Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append(data["summary"])
        lines.append("")

        lines.append("## Sections / Topics")
        lines.append("")
        lines.append(data["sections"])
        lines.append("")

        lines.append("## Key Facts")
        lines.append("")
        lines.append(data["key_facts"])
        lines.append("")

        if data.get("links"):
            lines.append("## Links Found")
            lines.append("")
            for link in data["links"][:20]:
                lines.append(f"- [{link['label']}]({link['url']})")
            lines.append("")

        return "\n".join(lines)

    def _slugify(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_-]+", "_", text)
        return text[:50]
