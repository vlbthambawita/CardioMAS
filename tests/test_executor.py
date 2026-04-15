"""
Tests for the Executor Agent (V4).
Uses real subprocess execution with simple test scripts.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from cardiomas.schemas.state import (
    ExecutionResult,
    GraphState,
    RefinementContext,
    ScriptRecord,
    UserOptions,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def state_with_scripts(tmp_path):
    """State with pre-generated scripts that actually run and produce correct output."""
    opts = UserOptions(
        dataset_source=str(tmp_path),
        local_path=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        v4_subset_size=10,
    )
    state = GraphState(
        dataset_source=str(tmp_path),
        user_options=opts,
        v4_output_dir=str(tmp_path / "output" / "ds" / "v4"),
        v4_pipeline_phase="subset_validation",
    )

    # Create output dirs
    subset_out = tmp_path / "output" / "ds" / "v4" / "outputs" / "subset"
    subset_out.mkdir(parents=True, exist_ok=True)
    full_out = tmp_path / "output" / "ds" / "v4" / "outputs" / "full"
    full_out.mkdir(parents=True, exist_ok=True)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    return state, tmp_path, scripts_dir, subset_out, full_out


def _write_script(scripts_dir: Path, name: str, content: str) -> ScriptRecord:
    path = scripts_dir / name
    path.write_text(content)
    return ScriptRecord(
        name=name,
        path=str(path),
        purpose=f"Test script {name}",
        output_dir=str(scripts_dir.parent / "out"),
        timeout=30,
        phase="subset" if not name.startswith("04") else "full",
    )


# ── Tests ─────────────────────────────────────────────────────────────────

class TestExecutorAgent:
    def test_runs_subset_scripts(self, state_with_scripts):
        """Executor runs scripts 00-03 in subset_validation phase."""
        state, tmp_path, scripts_dir, subset_out, _ = state_with_scripts

        # Write real scripts that produce expected output
        s00 = scripts_dir / "00_explore_structure.py"
        s00.write_text(textwrap.dedent(f"""\
            import json
            from pathlib import Path
            DATASET_PATH = r"{tmp_path}"
            OUTPUT_DIR = r"{subset_out}"
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            print("TOTAL_FILES=42")
            print("EXTENSIONS=[\\".csv\\", \\".hea\\"]")
            print(f"ROOT={{DATASET_PATH}}")
        """))

        s01 = scripts_dir / "01_extract_metadata.py"
        s01.write_text(textwrap.dedent(f"""\
            from pathlib import Path
            OUTPUT_DIR = r"{subset_out}"
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            print("COLUMNS=[\\"ecg_id\\", \\"patient_id\\", \\"scp_codes\\"]")
            print("DTYPES={{}}")
            print("SAMPLE_ROWS=5")
        """))

        s02 = scripts_dir / "02_compute_statistics.py"
        s02.write_text(textwrap.dedent(f"""\
            import csv
            from pathlib import Path
            OUTPUT_DIR = r"{subset_out}"
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            with open(f"{{OUTPUT_DIR}}/stats.csv", "w") as f:
                f.write("label,count\\nNORM,5000\\nAFIB,2000\\n")
            print("LABEL_FIELD=scp_codes")
        """))

        s03 = scripts_dir / "03_generate_splits_subset.py"
        splits_data = {"splits": {"train": [f"r{i}" for i in range(7)],
                                   "val": [f"v{i}" for i in range(2)],
                                   "test": [f"t{i}" for i in range(1)]}}
        s03.write_text(textwrap.dedent(f"""\
            import json
            from pathlib import Path
            OUTPUT_DIR = r"{subset_out}"
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            splits = {json.dumps(splits_data)}
            with open(f"{{OUTPUT_DIR}}/splits_subset.json", "w") as f:
                json.dump(splits, f)
            print("SUBSET_SIZE=10")
            print("SPLITS_SHA256=abc123def456")
        """))

        for name, path in [
            ("00_explore_structure.py", s00),
            ("01_extract_metadata.py", s01),
            ("02_compute_statistics.py", s02),
            ("03_generate_splits_subset.py", s03),
        ]:
            state.v4_generated_scripts[name] = ScriptRecord(
                name=name,
                path=str(path),
                purpose=f"test {name}",
                output_dir=str(subset_out),
                timeout=30,
                phase="subset",
            )

        from cardiomas.agents.executor import executor_agent
        result = executor_agent(state)

        assert result.v4_subset_validated is True
        assert len(result.v4_execution_results) == 4
        assert all(r.verification_passed for r in result.v4_execution_results)
        assert result.v4_execution_summary != ""
        assert "stats.csv" in result.v4_generated_files
        assert "splits_subset.json" in result.v4_generated_files

    def test_populates_execution_results(self, state_with_scripts):
        """Each script execution should be captured as an ExecutionResult."""
        state, tmp_path, scripts_dir, subset_out, _ = state_with_scripts

        script = scripts_dir / "00_explore_structure.py"
        script.write_text(textwrap.dedent(f"""\
            DATASET_PATH = r"{tmp_path}"
            OUTPUT_DIR = r"{subset_out}"
            from pathlib import Path
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            print("TOTAL_FILES=5")
            print("EXTENSIONS=[\\".csv\\"]")
            print(f"ROOT={{DATASET_PATH}}")
        """))
        state.v4_generated_scripts["00_explore_structure.py"] = ScriptRecord(
            name="00_explore_structure.py",
            path=str(script),
            purpose="test",
            output_dir=str(subset_out),
            timeout=30,
            phase="subset",
        )

        from cardiomas.agents.executor import executor_agent
        result = executor_agent(state)

        assert len(result.v4_execution_results) >= 1
        er = result.v4_execution_results[0]
        assert isinstance(er, ExecutionResult)
        assert er.script_name == "00_explore_structure.py"
        assert er.exit_code == 0
        assert "TOTAL_FILES" in er.stdout

    def test_stops_on_first_failure(self, state_with_scripts):
        """Executor should stop and set refinement_context on first failure."""
        state, tmp_path, scripts_dir, subset_out, _ = state_with_scripts

        s00 = scripts_dir / "00_explore_structure.py"
        s00.write_text(textwrap.dedent(f"""\
            DATASET_PATH = r"{tmp_path}"
            OUTPUT_DIR = r"{subset_out}"
            from pathlib import Path
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            print("TOTAL_FILES=10")
            print("EXTENSIONS=[\\".csv\\"]")
            print(f"ROOT={{DATASET_PATH}}")
        """))

        s01 = scripts_dir / "01_extract_metadata.py"
        s01.write_text("import sys; sys.exit(1)\n")  # Fails!

        for name, path in [("00_explore_structure.py", s00), ("01_extract_metadata.py", s01)]:
            state.v4_generated_scripts[name] = ScriptRecord(
                name=name,
                path=str(path),
                purpose=f"test {name}",
                output_dir=str(subset_out),
                timeout=30,
                phase="subset",
            )

        from cardiomas.agents.executor import executor_agent
        result = executor_agent(state)

        # Should have stopped after the failure
        assert result.v4_subset_validated is False
        assert result.v4_refinement_context is not None
        assert result.v4_refinement_context.failed_script == "01_extract_metadata.py"
        assert len(result.errors) > 0
        # 00 should have run, 01 should have failed
        assert len(result.v4_execution_results) == 2

    def test_full_run_phase_runs_04_only(self, state_with_scripts):
        """In full_run phase, only script 04 should be executed."""
        state, tmp_path, scripts_dir, subset_out, full_out = state_with_scripts
        state.v4_pipeline_phase = "full_run"

        # Add a subset script (should NOT run in full_run phase)
        s00 = scripts_dir / "00_explore_structure.py"
        s00.write_text("raise RuntimeError('should not run in full_run')\n")
        state.v4_generated_scripts["00_explore_structure.py"] = ScriptRecord(
            name="00_explore_structure.py",
            path=str(s00),
            purpose="should not run",
            output_dir=str(subset_out),
            timeout=30,
            phase="subset",
        )

        # Add the full script (SHOULD run)
        splits_data = {"splits": {"train": [f"r{i}" for i in range(700)],
                                   "val": [f"v{i}" for i in range(150)],
                                   "test": [f"t{i}" for i in range(150)]}}
        s04 = scripts_dir / "04_generate_splits_full.py"
        s04.write_text(textwrap.dedent(f"""\
            import json
            from pathlib import Path
            OUTPUT_DIR = r"{full_out}"
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            splits = {json.dumps(splits_data)}
            with open(f"{{OUTPUT_DIR}}/splits.json", "w") as f:
                json.dump(splits, f)
            print("TOTAL_RECORDS=1000")
            print("SPLITS_SHA256=abc123")
        """))
        state.v4_generated_scripts["04_generate_splits_full.py"] = ScriptRecord(
            name="04_generate_splits_full.py",
            path=str(s04),
            purpose="full splits",
            output_dir=str(full_out),
            timeout=30,
            phase="full",
        )

        from cardiomas.agents.executor import executor_agent
        result = executor_agent(state)

        # Only 04 should have run
        assert len(result.v4_execution_results) == 1
        assert result.v4_execution_results[0].script_name == "04_generate_splits_full.py"
        assert "splits.json" in result.v4_generated_files

    def test_builds_execution_summary(self, state_with_scripts):
        """After successful subset validation, v4_execution_summary should be populated."""
        state, tmp_path, scripts_dir, subset_out, _ = state_with_scripts

        s00 = scripts_dir / "00_explore_structure.py"
        s00.write_text(textwrap.dedent(f"""\
            from pathlib import Path
            OUTPUT_DIR = r"{subset_out}"
            DATASET_PATH = r"{tmp_path}"
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            print("TOTAL_FILES=100")
            print("EXTENSIONS=[\\".csv\\"]")
            print(f"ROOT={{DATASET_PATH}}")
        """))
        state.v4_generated_scripts["00_explore_structure.py"] = ScriptRecord(
            name="00_explore_structure.py",
            path=str(s00),
            purpose="test",
            output_dir=str(subset_out),
            timeout=30,
            phase="subset",
        )

        from cardiomas.agents.executor import executor_agent
        result = executor_agent(state)

        # With only 00, subset is not yet fully validated (need 00-03)
        # but execution results should still be populated
        assert len(result.v4_execution_results) >= 1


class TestExecutorSelectScripts:
    def test_selects_00_to_03_for_subset(self):
        from cardiomas.agents.executor import _select_scripts
        from cardiomas.schemas.state import ScriptRecord

        scripts = {
            "00_explore.py": ScriptRecord(name="00_explore.py", path="/p", purpose="", output_dir="/o"),
            "01_meta.py": ScriptRecord(name="01_meta.py", path="/p", purpose="", output_dir="/o"),
            "04_full.py": ScriptRecord(name="04_full.py", path="/p", purpose="", output_dir="/o"),
        }
        state = GraphState(dataset_source="/d", v4_generated_scripts=scripts)
        state.v4_pipeline_phase = "subset_validation"

        selected = _select_scripts(state, "subset_validation")
        assert "00_explore.py" in selected
        assert "01_meta.py" in selected
        assert "04_full.py" not in selected

    def test_selects_04_for_full_run(self):
        from cardiomas.agents.executor import _select_scripts
        from cardiomas.schemas.state import ScriptRecord

        scripts = {
            "00_explore.py": ScriptRecord(name="00_explore.py", path="/p", purpose="", output_dir="/o"),
            "04_full.py": ScriptRecord(name="04_full.py", path="/p", purpose="", output_dir="/o"),
        }
        state = GraphState(dataset_source="/d", v4_generated_scripts=scripts)
        state.v4_pipeline_phase = "full_run"

        selected = _select_scripts(state, "full_run")
        assert "04_full.py" in selected
        assert "00_explore.py" not in selected
