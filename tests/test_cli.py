from __future__ import annotations

from typer.testing import CliRunner

from cardiomas.cli.main import app
from cardiomas.knowledge_department.models import PageKnowledge, SourceProvenance

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "cardiomas" in result.output


def test_list_command():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    # Should show at least one known dataset
    assert "ptb-xl" in result.output or "Known ECG" in result.output


def test_config_show():
    result = runner.invoke(app, ["config", "--show"])
    assert result.exit_code == 0
    assert "OLLAMA" in result.output or "ollama" in result.output.lower()


def test_status_not_found(monkeypatch):
    """Status check for a non-existent dataset should not crash."""
    from unittest.mock import MagicMock

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = {"exists": False}
    monkeypatch.setattr("cardiomas.tools.publishing_tools.check_hf_repo", mock_tool)
    result = runner.invoke(app, ["status", "fake-dataset-xyz"])
    assert result.exit_code == 0
    assert "not yet published" in result.output or "fake-dataset-xyz" in result.output


def test_organize_command(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text("record_id,patient_id,label\nr1,p1,NORM\n", encoding="utf-8")

    def fake_fetch(url: str, rate_limit_seconds: float = 0.0) -> PageKnowledge:
        return PageKnowledge(
            url=url,
            title="Dataset card",
            description="Demo ECG dataset",
            headings=["Overview"],
            content_blocks=["Contains ECG examples."],
            links=[url],
            provenance=SourceProvenance(url=url, fetch_method="test"),
        )

    monkeypatch.setattr("cardiomas.knowledge_department.pipeline.fetch_and_parse_page", fake_fetch)
    result = runner.invoke(
        app,
        [
            "organize",
            str(dataset_dir),
            "--dataset-name",
            "demo-ecg",
            "--knowledge-url",
            "https://example.org/demo",
            "--approve",
        ],
    )

    assert result.exit_code == 0
    assert "Organization workflow" in result.output
    assert "approved" in result.output


def test_organize_command_from_config(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text("record_id,patient_id,label\nr1,p1,NORM\n", encoding="utf-8")
    config_path = tmp_path / "organize.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset_name: config-ecg",
                f"local_data_path: {dataset_dir}",
                "knowledge_urls:",
                "  - https://example.org/demo",
                "approve: true",
            ]
        ),
        encoding="utf-8",
    )

    def fake_fetch(url: str, rate_limit_seconds: float = 0.0) -> PageKnowledge:
        return PageKnowledge(
            url=url,
            title="Dataset card",
            description="Demo ECG dataset",
            headings=["Overview"],
            content_blocks=["Contains ECG examples."],
            links=[url],
            provenance=SourceProvenance(url=url, fetch_method="test"),
        )

    monkeypatch.setattr("cardiomas.knowledge_department.pipeline.fetch_and_parse_page", fake_fetch)
    result = runner.invoke(app, ["organize", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "config-ecg" in result.output
    assert "approved" in result.output
