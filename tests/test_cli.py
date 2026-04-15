from __future__ import annotations

from typer.testing import CliRunner

from cardiomas.cli.main import app

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
