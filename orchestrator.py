<<<<<<< HEAD
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
=======
from graph.graph import build_graph


class Orchestrator:
    """Runs the LangGraph-based multi-agent pipeline."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.graph = build_graph()

    def run(self, url: str) -> str:
        print("\n" + "=" * 60)
        print(f"  LangGraph Multi-Agent Web Extractor")
        print(f"  URL: {url}")
        print("=" * 60)

        initial_state = {
            "url": url,
            "title": "",
            "raw_text": "",
            "links": [],
            "summary": "",
            "sections": "",
            "key_facts": "",
            "ecg_analysis": "",
            "output_path": "",
            "error": None,
        }

        final_state = self.graph.invoke(initial_state)

        if final_state.get("error"):
            raise RuntimeError(final_state["error"])

        return final_state["output_path"]
>>>>>>> 9eb58f0d93e89e69838ab82fa775a1f194452183
