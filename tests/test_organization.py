from __future__ import annotations

from pathlib import Path

from cardiomas.knowledge_department.models import PageKnowledge, SourceProvenance
from cardiomas.organization import build_default_organization, resolve_organization_config


def _make_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
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


def test_resolve_organization_config_from_yaml_with_aliases(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    config_path = tmp_path / "organize.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_name: config-demo",
                f"local_data_path: {dataset_dir}",
                "knowledge_links:",
                "  - https://example.org/dataset",
                "goal: Config-driven run",
                "output_dir: custom_output",
                "approve: true",
            ]
        ),
        encoding="utf-8",
    )

    config = resolve_organization_config(config_path=str(config_path))

    assert config.dataset_name == "config-demo"
    assert config.resolved_dataset_dir == str(dataset_dir)
    assert config.knowledge_urls == ["https://example.org/dataset"]
    assert config.goal == "Config-driven run"
    assert config.output_dir == str((tmp_path / "custom_output").resolve())
    assert config.approve is True


def test_resolve_organization_config_uses_config_relative_paths(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    dataset_dir = config_dir / "demo_dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text("record_id,label\nr1,NORM\n", encoding="utf-8")

    config_path = config_dir / "organize.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_name: relative-demo",
                "local_data_path: demo_dataset",
                "output_dir: outputs/demo",
            ]
        ),
        encoding="utf-8",
    )

    config = resolve_organization_config(config_path=str(config_path))

    assert config.resolved_dataset_dir == str(dataset_dir.resolve())
    assert config.output_dir == str((config_dir / "outputs" / "demo").resolve())


def test_cli_values_override_config_dataset_path(tmp_path):
    config_path = tmp_path / "organize.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_name: config-demo",
                "local_data_path: /tmp/original",
                "knowledge_urls:",
                "  - https://example.org/dataset",
            ]
        ),
        encoding="utf-8",
    )
    override_dir = _make_dataset(tmp_path / "override_root")

    config = resolve_organization_config(
        config_path=str(config_path),
        dataset_dir=str(override_dir),
    )

    assert config.resolved_dataset_dir == str(override_dir)
