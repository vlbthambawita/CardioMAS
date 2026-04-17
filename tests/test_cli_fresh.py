from __future__ import annotations

from typer.testing import CliRunner

from cardiomas.cli.main import app

runner = CliRunner()


def test_build_corpus_and_query_cli(tmp_path):
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text(
        "record_id,patient_id,label\nrec_1,pat_1,NORM\nrec_2,pat_2,AFIB\n",
        encoding="utf-8",
    )
    notes_path = tmp_path / "notes.md"
    notes_path.write_text("The dataset contains NORM and AFIB labels.", encoding="utf-8")
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "output_dir: output",
                "sources:",
                "  - kind: dataset_dir",
                "    path: data",
                "    label: demo-dataset",
                "  - kind: local_file",
                "    path: notes.md",
                "    label: demo-notes",
            ]
        ),
        encoding="utf-8",
    )

    build_result = runner.invoke(app, ["build-corpus", "--config", str(config_path)])
    query_result = runner.invoke(app, ["query", "What labels are present?", "--config", str(config_path)])
    tools_result = runner.invoke(app, ["inspect-tools", "--config", str(config_path)])

    assert build_result.exit_code == 0
    assert "Corpus built" in build_result.output
    assert query_result.exit_code == 0
    assert "AFIB" in query_result.output or "NORM" in query_result.output
    assert tools_result.exit_code == 0
    assert "retrieve_corpus" in tools_result.output


def test_check_ollama_cli(tmp_path, monkeypatch):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "output_dir: output",
                "llm:",
                "  provider: ollama",
                "  model: llama3.2",
                "embeddings:",
                "  provider: ollama",
                "  model: embeddinggemma",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "cardiomas.cli.main.CardioMAS.check_ollama",
        lambda self: {
            "llm": {"configured": True, "ok": True, "models": [{"name": "llama3.2"}], "error": ""},
            "embeddings": {"configured": True, "ok": True, "models": [{"name": "embeddinggemma"}], "error": ""},
        },
    )

    result = runner.invoke(app, ["check-ollama", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "llama3.2" in result.output
    assert "embeddinggemma" in result.output


def test_inspect_tools_shows_autonomous_tools_when_enabled(tmp_path):
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text("record_id,label\nrec_1,NORM\n", encoding="utf-8")
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "\n".join(
            [
                "output_dir: output",
                "sources:",
                "  - kind: dataset_dir",
                "    path: data",
                "    label: demo-dataset",
                "autonomy:",
                "  enable_code_agents: true",
                "  allow_tool_codegen: true",
                "  allow_script_codegen: true",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect-tools", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "dataset_statistics" in result.output
    assert "read_dataset_file" in result.output
    assert "generate_shell_script" in result.output
