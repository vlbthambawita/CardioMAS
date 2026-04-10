from agents.scrape_agent import ScrapeAgent
from agents.extract_agent import ExtractAgent
from agents.writer_agent import WriterAgent


class Orchestrator:
    """Coordinates all agents to scrape a URL and produce a .md file."""

    def __init__(self, output_dir: str = "output"):
        self.scraper = ScrapeAgent()
        self.extractor = ExtractAgent()
        self.writer = WriterAgent(output_dir=output_dir)

    def run(self, url: str) -> str:
        print("\n" + "=" * 60)
        print(f"  Multi-Agent Web Extractor")
        print(f"  URL: {url}")
        print("=" * 60 + "\n")

        # Step 1: Scrape
        scrape_result = self.scraper.run(url)

        # Step 2: Extract with Gemma 4
        extracted = self.extractor.run(scrape_result)

        # Step 3: Write to .md
        output_path = self.writer.run(extracted)

        print(f"\n[Orchestrator] Done. Output -> {output_path}\n")
        return output_path
