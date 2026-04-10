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
