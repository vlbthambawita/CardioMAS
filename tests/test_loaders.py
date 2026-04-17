from __future__ import annotations

from cardiomas.knowledge.loaders import load_source
from cardiomas.schemas.config import KnowledgeSource


def test_dataset_loader_skips_license_and_enriches_csv_and_html(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "LICENSE.txt").write_text("license text", encoding="utf-8")
    (dataset_dir / "metadata.csv").write_text("record_id,label\n1,NORM\n2,AFIB\n", encoding="utf-8")
    (dataset_dir / "index.html").write_text(
        "<html><head><title>Dataset Index</title></head><body><h1>records100</h1></body></html>",
        encoding="utf-8",
    )
    nested_dir = dataset_dir / "records100" / "00000"
    nested_dir.mkdir(parents=True)
    (nested_dir / "index.html").write_text(
        "<html><head><title>Nested Index</title></head><body><h1>deep</h1></body></html>",
        encoding="utf-8",
    )

    docs = load_source(KnowledgeSource(kind="dataset_dir", path=str(dataset_dir), label="dataset"))

    titles = {doc.title for doc in docs}
    assert "LICENSE.txt" not in titles
    assert "metadata.csv" in titles
    assert "index.html" in titles
    assert "records100/00000/index.html" not in titles

    csv_doc = next(doc for doc in docs if doc.title == "metadata.csv")
    assert "Columns: record_id, label" in csv_doc.content
    html_doc = next(doc for doc in docs if doc.title == "index.html")
    assert "Dataset Index" in html_doc.content
