from __future__ import annotations

import json

from cardiomas.knowledge_department.models import PageKnowledge, SourceProvenance
from cardiomas.knowledge_department.pipeline import build_knowledge_bundle
from cardiomas.knowledge_department.scraper import parse_page


def test_parse_page_extracts_key_sections():
    html = """
    <html>
      <head>
        <title>Demo Dataset</title>
        <meta name="description" content="Synthetic ECG dataset" />
      </head>
      <body>
        <main>
          <h1>Dataset Card</h1>
          <p>This synthetic ECG dataset contains labeled examples for testing.</p>
          <table>
            <tr><th>Field</th><th>Value</th></tr>
            <tr><td>Records</td><td>4</td></tr>
          </table>
          <a href="/paper">Paper</a>
        </main>
      </body>
    </html>
    """
    page = parse_page(
        "https://example.org/demo",
        html,
        SourceProvenance(url="https://example.org/demo", fetch_method="test"),
    )

    assert page.title == "Demo Dataset"
    assert "Dataset Card" in page.headings
    assert page.tables[0].headers == ["Field", "Value"]
    assert "https://example.org/paper" in page.links


def test_build_knowledge_bundle_writes_expected_files(tmp_path, monkeypatch):
    def fake_fetch(url: str, rate_limit_seconds: float = 0.0) -> PageKnowledge:
        return PageKnowledge(
            url=url,
            title="Demo Dataset",
            description="Synthetic ECG dataset",
            headings=["Overview"],
            content_blocks=["This dataset includes a tiny ECG sample."],
            links=[url],
            provenance=SourceProvenance(url=url, fetch_method="test"),
        )

    monkeypatch.setattr("cardiomas.knowledge_department.pipeline.fetch_and_parse_page", fake_fetch)
    bundle, output_paths, notes = build_knowledge_bundle("Demo Dataset", ["https://example.org/demo"], str(tmp_path))

    assert bundle.dataset_name == "Demo Dataset"
    assert notes == []
    assert json.loads((tmp_path / "knowledge" / "datasets" / "demo-dataset" / "overview.json").read_text())["dataset_name"] == "Demo Dataset"
    assert output_paths["notes.md"].endswith("notes.md")
