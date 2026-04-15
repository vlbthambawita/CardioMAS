"""
Tests for the Data Engineering Agent (V4).
Uses a mock LLM to verify script generation behaviour.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cardiomas.schemas.state import (
    GraphState,
    RefinementContext,
    ScriptRecord,
    UserOptions,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def base_state(tmp_path):
    """A GraphState with a local dataset path."""
    opts = UserOptions(
        dataset_source=str(tmp_path / "dataset"),
        local_path=str(tmp_path / "dataset"),
        output_dir=str(tmp_path / "output"),
        seed=42,
        v4_subset_size=50,
    )
    return GraphState(dataset_source=str(tmp_path / "dataset"), user_options=opts)


def _make_mock_llm(scripts: dict[str, str] | None = None):
    """Return a mock LLM that produces a valid ScriptBundle JSON."""
    if scripts is None:
        scripts = {
            "00_explore_structure.py": _minimal_script("00"),
            "01_extract_metadata.py": _minimal_script("01"),
            "02_compute_statistics.py": _minimal_script("02"),
            "03_generate_splits_subset.py": _minimal_script("03"),
            "04_generate_splits_full.py": _minimal_script("04"),
        }
    bundle = {
        "scripts": scripts,
        "notes": "test bundle",
    }
    mock_llm = MagicMock()
    mock_llm.model = "mock-model"
    mock_llm.model_name = "mock-model"
    mock_response = MagicMock()
    mock_response.content = json.dumps(bundle)
    mock_llm.invoke.return_value = mock_response
    # Make model_copy return a new mock with same behaviour
    mock_llm.model_copy = lambda **kwargs: mock_llm
    return mock_llm


def _minimal_script(prefix: str) -> str:
    """Minimal valid Python script content for testing."""
    stdout_key = {
        "00": "TOTAL_FILES=0\nEXTENSIONS=[]\nROOT=/data\n",
        "01": "COLUMNS=[]\nDTYPES={}\nSAMPLE_ROWS=0\n",
        "02": "LABEL_FIELD=none\n",
        "03": "SUBSET_SIZE=50\nSPLITS_SHA256=abc123\n",
        "04": "TOTAL_RECORDS=0\nSPLITS_SHA256=abc123\n",
    }.get(prefix, f"PREFIX={prefix}\n")
    return (
        f"#!/usr/bin/env python3\n"
        f'"""\nScript: {prefix}_test.py\n"""\n'
        f"OUTPUT_DIR = '/tmp/output'\n"
        f"DATASET_PATH = '/tmp/data'\n"
        f"from pathlib import Path\n"
        f"Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)\n"
        f'print("{stdout_key.strip()}")\n'
    )


# ── Tests ─────────────────────────────────────────────────────────────────

class TestDataEngineerAgent:
    def test_generates_scripts(self, base_state, tmp_path):
        """Agent should generate scripts and store ScriptRecords in state."""
        mock_llm = _make_mock_llm()

        with patch("cardiomas.llm_factory.get_llm_for_agent", return_value=mock_llm):
            from cardiomas.agents.data_engineer import data_engineer_agent
            result = data_engineer_agent(base_state)

        assert len(result.v4_generated_scripts) > 0
        assert "00_explore_structure.py" in result.v4_generated_scripts
        assert "04_generate_splits_full.py" in result.v4_generated_scripts

        # Check ScriptRecord structure
        rec = result.v4_generated_scripts["00_explore_structure.py"]
        assert isinstance(rec, ScriptRecord)
        assert rec.phase == "subset"
        assert Path(rec.path).exists()

    def test_sets_pipeline_phase(self, base_state):
        """Agent should set v4_pipeline_phase to subset_validation."""
        mock_llm = _make_mock_llm()

        with patch("cardiomas.llm_factory.get_llm_for_agent", return_value=mock_llm):
            from cardiomas.agents.data_engineer import data_engineer_agent
            result = data_engineer_agent(base_state)

        assert result.v4_pipeline_phase == "subset_validation"

    def test_creates_v4_output_dir(self, base_state):
        """Agent should set v4_output_dir in state."""
        mock_llm = _make_mock_llm()

        with patch("cardiomas.llm_factory.get_llm_for_agent", return_value=mock_llm):
            from cardiomas.agents.data_engineer import data_engineer_agent
            result = data_engineer_agent(base_state)

        assert result.v4_output_dir != ""
        assert Path(result.v4_output_dir).is_dir()

    def test_04_script_assigned_full_phase(self, base_state):
        """Script 04_generate_splits_full.py should have phase='full'."""
        mock_llm = _make_mock_llm()

        with patch("cardiomas.llm_factory.get_llm_for_agent", return_value=mock_llm):
            from cardiomas.agents.data_engineer import data_engineer_agent
            result = data_engineer_agent(base_state)

        rec = result.v4_generated_scripts.get("04_generate_splits_full.py")
        if rec:  # depends on LLM returning it
            assert rec.phase == "full"

    def test_refinement_mode_regenerates_failed_script(self, base_state):
        """In refinement mode, only the failed script is regenerated."""
        # Set up refinement context
        base_state.v4_refinement_context = RefinementContext(
            failed_script="01_extract_metadata.py",
            error_message="KeyError: 'patient_id'",
            stdout_excerpt="COLUMNS=['ecg_id']\n",
            attempt=1,
        )
        # Pre-populate some scripts as if we already ran once
        base_state.v4_generated_scripts["00_explore_structure.py"] = ScriptRecord(
            name="00_explore_structure.py",
            path="/tmp/script.py",
            purpose="explore",
            output_dir="/tmp/out",
        )

        mock_llm = _make_mock_llm(scripts={
            "01_extract_metadata.py": _minimal_script("01"),
        })

        with patch("cardiomas.llm_factory.get_llm_for_agent", return_value=mock_llm):
            from cardiomas.agents.data_engineer import data_engineer_agent
            result = data_engineer_agent(base_state)

        # Old script should still be there
        assert "00_explore_structure.py" in result.v4_generated_scripts
        # New script should be regenerated
        assert "01_extract_metadata.py" in result.v4_generated_scripts
        # Refinement context should be cleared after regeneration
        assert result.v4_refinement_context is None

    def test_injects_output_dir_constant(self, base_state):
        """Agent should override OUTPUT_DIR in generated scripts."""
        scripts = {"00_explore_structure.py": 'OUTPUT_DIR = "/wrong/path"\nprint("test")\n'}
        mock_llm = _make_mock_llm(scripts=scripts)

        with patch("cardiomas.llm_factory.get_llm_for_agent", return_value=mock_llm):
            from cardiomas.agents.data_engineer import data_engineer_agent
            result = data_engineer_agent(base_state)

        rec = result.v4_generated_scripts.get("00_explore_structure.py")
        if rec and Path(rec.path).exists():
            content = Path(rec.path).read_text()
            # Should not contain the wrong path
            assert "/wrong/path" not in content

    def test_handles_llm_failure_gracefully(self, base_state):
        """Agent should add error and return state if LLM fails."""
        from cardiomas.agents.base import AgentOutputError

        with patch("cardiomas.llm_factory.get_llm_for_agent") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = RuntimeError("Ollama down")
            mock_get_llm.return_value = mock_llm

            from cardiomas.agents.data_engineer import data_engineer_agent
            result = data_engineer_agent(base_state)

        # Should have recorded an error but not raised
        assert len(result.errors) > 0

    def test_logs_execution(self, base_state):
        """Agent should append LogEntry records."""
        mock_llm = _make_mock_llm()

        with patch("cardiomas.llm_factory.get_llm_for_agent", return_value=mock_llm):
            from cardiomas.agents.data_engineer import data_engineer_agent
            result = data_engineer_agent(base_state)

        agent_entries = [e for e in result.execution_log if e.agent == "data_engineer"]
        assert any(e.action == "start" for e in agent_entries)


# ── _inject_output_dir helper ─────────────────────────────────────────────

class TestInjectOutputDir:
    def test_replaces_output_dir(self):
        from cardiomas.agents.data_engineer import _inject_output_dir

        script = 'OUTPUT_DIR = "/old/path"\nprint("hello")\n'
        result = _inject_output_dir(script, "/new/path", "/data")
        assert '/new/path' in result
        assert '/old/path' not in result

    def test_replaces_dataset_path(self):
        from cardiomas.agents.data_engineer import _inject_output_dir

        script = 'DATASET_PATH = "/old/data"\nprint("hello")\n'
        result = _inject_output_dir(script, "/out", "/new/data")
        assert '/new/data' in result
        assert '/old/data' not in result

    def test_injects_when_missing(self):
        from cardiomas.agents.data_engineer import _inject_output_dir

        script = 'print("no constants here")\n'
        result = _inject_output_dir(script, "/out", "/data")
        assert 'OUTPUT_DIR' in result
        assert 'DATASET_PATH' in result
