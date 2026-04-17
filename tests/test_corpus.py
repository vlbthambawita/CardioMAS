from __future__ import annotations

from cardiomas.knowledge.corpus import build_corpus, load_corpus
from cardiomas.retrieval.hybrid import retrieve
from cardiomas.schemas.config import KnowledgeSource, RuntimeConfig


def test_build_corpus_and_retrieve(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text(
        "record_id,patient_id,label\nrec_1,pat_1,NORM\nrec_2,pat_2,AFIB\n",
        encoding="utf-8",
    )
    notes_path = tmp_path / "notes.md"
    notes_path.write_text("The dataset contains NORM and AFIB rhythm labels.", encoding="utf-8")

    config = RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        sources=[
            KnowledgeSource(kind="dataset_dir", path=str(dataset_dir), label="dataset"),
            KnowledgeSource(kind="local_file", path=str(notes_path), label="notes"),
        ],
    )

    manifest = build_corpus(config)
    chunks = load_corpus(config)
    hits = retrieve(chunks, "Which labels are present?", config.retrieval)

    assert manifest.document_count == 2
    assert manifest.chunk_count >= 2
    assert any("AFIB" in hit.content or "NORM" in hit.content for hit in hits)
