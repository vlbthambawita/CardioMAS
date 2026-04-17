from __future__ import annotations

from cardiomas.agentic.runtime import AgenticRuntime
from cardiomas.schemas.config import KnowledgeSource, RuntimeConfig


def test_runtime_query_returns_grounded_answer(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text(
        "record_id,patient_id,label\nrec_1,pat_1,NORM\nrec_2,pat_2,AFIB\nrec_3,pat_3,MI\n",
        encoding="utf-8",
    )
    notes = tmp_path / "notes.md"
    notes.write_text("Known labels in this dataset are NORM, AFIB, and MI.", encoding="utf-8")

    config = RuntimeConfig(
        output_dir=str(tmp_path / "output"),
        sources=[
            KnowledgeSource(kind="dataset_dir", path=str(dataset_dir), label="dataset"),
            KnowledgeSource(kind="local_file", path=str(notes), label="notes"),
        ],
    )

    result = AgenticRuntime(config).query("What labels are present in the dataset?", force_rebuild=True)

    assert "NORM" in result.answer
    assert result.citations
    assert any(call.tool_name == "retrieve_corpus" for call in result.tool_calls)


def test_runtime_query_can_use_calculator(tmp_path):
    config = RuntimeConfig(output_dir=str(tmp_path / "output"), sources=[])

    result = AgenticRuntime(config).query("Calculate 7 * 6", force_rebuild=True)

    assert "42" in result.answer
