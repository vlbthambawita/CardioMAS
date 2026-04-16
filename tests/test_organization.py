from __future__ import annotations

from pathlib import Path

from cardiomas.knowledge_department.models import PageKnowledge, SourceProvenance
from cardiomas.organization import build_default_organization


def _make_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text(
        "record_id,patient_id,label\nrec_1,pat_1,NORM\nrec_2,pat_2,AFIB\n",
        encoding="utf-8",
    )
    return dataset_dir


def test_organization_workflow_routes_through_departments(tmp_path, monkeypatch):
    dataset_dir = _make_dataset(tmp_path)

    def fake_fetch(url: str, rate_limit_seconds: float = 0.0) -> PageKnowledge:
        return PageKnowledge(
            url=url,
            title="Dataset card",
            description="Demo ECG dataset",
            headings=["Overview"],
            content_blocks=["Contains ECG examples and labels."],
            links=[url],
            provenance=SourceProvenance(url=url, fetch_method="test"),
        )

    monkeypatch.setattr("cardiomas.knowledge_department.pipeline.fetch_and_parse_page", fake_fetch)

    result = build_default_organization().run(
        goal="Prepare dataset understanding assets",
        dataset_name="demo-ecg",
        dataset_dir=str(dataset_dir),
        knowledge_urls=["https://example.org/demo"],
        output_dir=str(tmp_path / "org_output"),
        approve=True,
    )

    assert result.status == "approved"
    assert [message.department for message in result.communication_log] == [
        "knowledge",
        "coding",
        "cardiology",
        "testing",
    ]
    assert all(approval.status.value == "approved" for approval in result.approvals)
    assert any(artifact.name == "tool_validation_json" for artifact in result.final_artifacts)
