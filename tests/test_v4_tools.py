"""
Tests for V4 tools:
- v4_output_tools: setup_v4_output_dir, read_generated_file, list_generated_files,
  write_execution_log, collect_generated_files
- script_verification: verify_explore_output, verify_metadata_output,
  verify_stats_output, verify_subset_splits_output, verify_ecg_stats_output,
  verify_script_output (dispatcher)
- code_tools: execute_script_with_env
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# ── v4_output_tools ───────────────────────────────────────────────────────

class TestSetupV4OutputDir:
    def test_creates_all_dirs(self, tmp_path):
        from cardiomas.tools.v4_output_tools import setup_v4_output_dir

        result = setup_v4_output_dir.invoke({
            "dataset_name": "test-dataset",
            "base_dir": str(tmp_path),
        })
        assert result["status"] == "ok"
        assert "root" in result
        dirs = result["dirs"]
        for key in ("scripts_subset", "scripts_full", "scripts_ecg_stats",
                    "outputs_subset", "outputs_full", "outputs_ecg_stats", "logs"):
            assert key in dirs
            assert Path(dirs[key]).is_dir()

    def test_idempotent(self, tmp_path):
        from cardiomas.tools.v4_output_tools import setup_v4_output_dir

        r1 = setup_v4_output_dir.invoke({"dataset_name": "ds", "base_dir": str(tmp_path)})
        r2 = setup_v4_output_dir.invoke({"dataset_name": "ds", "base_dir": str(tmp_path)})
        assert r1["status"] == "ok"
        assert r2["status"] == "ok"
        assert r1["root"] == r2["root"]


class TestReadGeneratedFile:
    def test_reads_csv(self, tmp_path):
        from cardiomas.tools.v4_output_tools import read_generated_file

        csv_file = tmp_path / "stats.csv"
        csv_file.write_text("label,count\nNORM,100\nAFIB,50\n")

        result = read_generated_file.invoke({"file_path": str(csv_file)})
        assert result["status"] == "ok"
        assert "NORM" in result["content"]
        assert result["truncated"] is False

    def test_reads_json(self, tmp_path):
        from cardiomas.tools.v4_output_tools import read_generated_file

        json_file = tmp_path / "splits_subset.json"
        json_file.write_text(json.dumps({"splits": {"train": ["r1", "r2"]}}))

        result = read_generated_file.invoke({"file_path": str(json_file)})
        assert result["status"] == "ok"
        assert "train" in result["content"]

    def test_refuses_signal_files(self, tmp_path):
        from cardiomas.tools.v4_output_tools import read_generated_file

        for ext in (".dat", ".h5", ".hdf5", ".edf", ".npy"):
            sig_file = tmp_path / f"record00001{ext}"
            sig_file.write_bytes(b"\x00\x01\x02\x03")
            result = read_generated_file.invoke({"file_path": str(sig_file)})
            assert result["status"] == "error"
            assert "V4 core constraint" in result["error"] or "signal" in result["error"].lower()

    def test_file_not_found(self):
        from cardiomas.tools.v4_output_tools import read_generated_file

        result = read_generated_file.invoke({"file_path": "/nonexistent/file.csv"})
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_respects_max_bytes(self, tmp_path):
        from cardiomas.tools.v4_output_tools import read_generated_file

        big_file = tmp_path / "big.csv"
        big_file.write_text("x" * 2000)

        result = read_generated_file.invoke({
            "file_path": str(big_file),
            "max_bytes": 100,
        })
        assert result["status"] == "ok"
        assert result["truncated"] is True
        assert len(result["content"]) == 100


class TestListGeneratedFiles:
    def test_lists_files(self, tmp_path):
        from cardiomas.tools.v4_output_tools import list_generated_files

        (tmp_path / "stats.csv").write_text("a,b\n1,2\n")
        (tmp_path / "splits_subset.json").write_text('{"splits":{}}')
        (tmp_path / "table1.md").write_text("# Table 1\n")

        result = list_generated_files.invoke({"output_dir": str(tmp_path)})
        assert result["status"] == "ok"
        names = [f["name"] for f in result["files"]]
        assert "stats.csv" in names
        assert "splits_subset.json" in names
        assert "table1.md" in names

    def test_excludes_signal_files(self, tmp_path):
        from cardiomas.tools.v4_output_tools import list_generated_files

        (tmp_path / "stats.csv").write_text("data")
        (tmp_path / "record00001.dat").write_bytes(b"\x00\x01")

        result = list_generated_files.invoke({"output_dir": str(tmp_path)})
        names = [f["name"] for f in result["files"]]
        assert "stats.csv" in names
        assert "record00001.dat" not in names

    def test_empty_dir(self, tmp_path):
        from cardiomas.tools.v4_output_tools import list_generated_files

        result = list_generated_files.invoke({"output_dir": str(tmp_path)})
        assert result["status"] == "ok"
        assert result["count"] == 0


class TestWriteExecutionLog:
    def test_writes_log(self, tmp_path):
        from cardiomas.tools.v4_output_tools import write_execution_log

        log_dir = str(tmp_path / "logs")
        result = write_execution_log.invoke({
            "log_dir": log_dir,
            "script_name": "00_explore_structure.py",
            "execution_result": {
                "exit_code": 0,
                "stdout": "TOTAL_FILES=42",
                "stderr": "",
                "duration_seconds": 1.5,
                "generated_files": {"stats.csv": "label,count\nNORM,100\n"},
            },
        })
        assert result["status"] == "ok"
        log_file = Path(result["log_file"])
        assert log_file.exists()
        data = json.loads(log_file.read_text())
        assert data["exit_code"] == 0
        # Generated files should be summarised, not full content
        assert "<" in str(data["generated_files"]["stats.csv"])


class TestCollectGeneratedFiles:
    def test_collects_csv_and_json(self, tmp_path):
        from cardiomas.tools.v4_output_tools import collect_generated_files

        (tmp_path / "stats.csv").write_text("label,count\nNORM,100\n")
        (tmp_path / "splits_subset.json").write_text('{"splits": {"train": []}}')

        files = collect_generated_files(str(tmp_path))
        assert "stats.csv" in files
        assert "splits_subset.json" in files
        assert "NORM" in files["stats.csv"]

    def test_excludes_large_files(self, tmp_path):
        from cardiomas.tools.v4_output_tools import collect_generated_files

        (tmp_path / "small.csv").write_text("a,b\n1,2\n")
        big = tmp_path / "big.csv"
        big.write_text("x" * 200_000)  # > 100KB cap

        files = collect_generated_files(str(tmp_path))
        assert "small.csv" in files
        assert "big.csv" not in files


# ── script_verification ───────────────────────────────────────────────────

class TestVerifyExploreOutput:
    def test_valid_output(self):
        from cardiomas.tools.script_verification import verify_explore_output

        stdout = "TOTAL_FILES=1234\nEXTENSIONS=['.csv', '.hea']\nROOT=/data/ptb-xl\n"
        result = verify_explore_output(stdout)
        assert result.passed is True
        assert "1234" in result.notes

    def test_missing_total_files(self):
        from cardiomas.tools.script_verification import verify_explore_output

        stdout = "EXTENSIONS=['.csv']\nROOT=/data\n"
        result = verify_explore_output(stdout)
        assert result.passed is False
        assert any("TOTAL_FILES" in issue for issue in result.issues)

    def test_empty_stdout(self):
        from cardiomas.tools.script_verification import verify_explore_output

        result = verify_explore_output("")
        assert result.passed is False

    def test_missing_structure_info(self):
        from cardiomas.tools.script_verification import verify_explore_output

        stdout = "TOTAL_FILES=100\n"
        result = verify_explore_output(stdout)
        assert result.passed is False


class TestVerifyMetadataOutput:
    def test_valid_output(self):
        from cardiomas.tools.script_verification import verify_metadata_output

        stdout = "COLUMNS=['ecg_id', 'patient_id', 'scp_codes', 'age']\nDTYPES={...}\n"
        result = verify_metadata_output(stdout)
        assert result.passed is True

    def test_missing_columns(self):
        from cardiomas.tools.script_verification import verify_metadata_output

        stdout = "DTYPES={...}\nSAMPLE_ROWS=5\n"
        result = verify_metadata_output(stdout)
        assert result.passed is False

    def test_empty_stdout(self):
        from cardiomas.tools.script_verification import verify_metadata_output

        result = verify_metadata_output("")
        assert result.passed is False


class TestVerifyStatsOutput:
    def test_valid_with_csv(self):
        from cardiomas.tools.script_verification import verify_stats_output

        generated = {"stats.csv": "label,count\nNORM,5000\nAFIB,2000\n"}
        result = verify_stats_output("LABEL_FIELD=scp_codes\n", generated)
        assert result.passed is True

    def test_csv_missing_but_stdout_mentions_it(self):
        from cardiomas.tools.script_verification import verify_stats_output

        result = verify_stats_output("Wrote stats.csv successfully", {})
        assert result.passed is True

    def test_nothing_at_all(self):
        from cardiomas.tools.script_verification import verify_stats_output

        result = verify_stats_output("some output with no mention", {})
        assert result.passed is False


class TestVerifySubsetSplitsOutput:
    def test_valid_splits(self):
        from cardiomas.tools.script_verification import verify_subset_splits_output

        splits_data = {"splits": {"train": [f"r{i}" for i in range(70)],
                                  "val": [f"v{i}" for i in range(15)],
                                  "test": [f"t{i}" for i in range(15)]}}
        generated = {"splits_subset.json": json.dumps(splits_data)}
        result = verify_subset_splits_output("SUBSET_SIZE=100\n", generated, 100)
        assert result.passed is True

    def test_missing_file(self):
        from cardiomas.tools.script_verification import verify_subset_splits_output

        result = verify_subset_splits_output("", {}, 100)
        assert result.passed is False
        assert any("splits_subset.json" in issue for issue in result.issues)

    def test_invalid_json(self):
        from cardiomas.tools.script_verification import verify_subset_splits_output

        generated = {"splits_subset.json": "this is not json {{{"}
        result = verify_subset_splits_output("", generated, 100)
        assert result.passed is False

    def test_empty_splits(self):
        from cardiomas.tools.script_verification import verify_subset_splits_output

        generated = {"splits_subset.json": json.dumps({"splits": {"train": [], "val": []}})}
        result = verify_subset_splits_output("", generated, 100)
        assert result.passed is False


class TestVerifyEcgStatsOutput:
    @pytest.mark.parametrize("prefix,expected_files", [
        ("10_class_distribution.py", ["class_dist.csv"]),
        ("11_per_lead_statistics.py", ["lead_stats.csv"]),
        ("12_signal_quality.py", ["quality_report.csv"]),
        ("13_clinical_plausibility.py", ["clinical_flags.csv"]),
        ("14_publication_table.py", ["table1.md", "table1.tex"]),
    ])
    def test_passes_with_expected_file(self, prefix, expected_files):
        from cardiomas.tools.script_verification import verify_ecg_stats_output

        generated = {fname: "some content" for fname in expected_files}
        result = verify_ecg_stats_output(prefix, "output\n", generated)
        assert result.passed is True

    def test_fails_missing_expected_file(self):
        from cardiomas.tools.script_verification import verify_ecg_stats_output

        result = verify_ecg_stats_output("10_class_dist.py", "output", {})
        assert result.passed is False

    def test_unknown_script_passes_with_stdout(self):
        from cardiomas.tools.script_verification import verify_ecg_stats_output

        result = verify_ecg_stats_output("99_custom.py", "some output line\n", {})
        assert result.passed is True


class TestVerifyScriptOutputDispatcher:
    def test_nonzero_exit_always_fails(self):
        from cardiomas.tools.script_verification import verify_script_output

        result = verify_script_output(
            script_name="00_explore.py",
            stdout="TOTAL_FILES=42\nROOT=/data\nEXTENSIONS=['.csv']\n",
            stderr="SomeError",
            exit_code=1,
            generated_files={},
        )
        assert result.passed is False
        assert any("exit" in issue.lower() for issue in result.issues)

    def test_dispatches_to_00_verifier(self):
        from cardiomas.tools.script_verification import verify_script_output

        result = verify_script_output(
            script_name="00_explore.py",
            stdout="TOTAL_FILES=100\nEXTENSIONS=['.csv']\nROOT=/data\n",
            stderr="",
            exit_code=0,
            generated_files={},
        )
        assert result.passed is True

    def test_dispatches_to_03_verifier(self):
        from cardiomas.tools.script_verification import verify_script_output

        splits_data = {"splits": {"train": [f"r{i}" for i in range(70)],
                                  "val": [f"v{i}" for i in range(15)],
                                  "test": [f"t{i}" for i in range(15)]}}
        generated = {"splits_subset.json": json.dumps(splits_data)}
        result = verify_script_output(
            script_name="03_generate_splits_subset.py",
            stdout="SUBSET_SIZE=100\n",
            stderr="",
            exit_code=0,
            generated_files=generated,
            subset_size=100,
        )
        assert result.passed is True


# ── execute_script_with_env ───────────────────────────────────────────────

class TestExecuteScriptWithEnv:
    def test_simple_success(self, tmp_path):
        from cardiomas.tools.code_tools import execute_script_with_env

        script = tmp_path / "test_script.py"
        script.write_text('print("TOTAL_FILES=42")\n')

        result = execute_script_with_env.invoke({
            "script_path": str(script),
            "timeout": 30,
        })
        assert result["exit_code"] == 0
        assert "TOTAL_FILES=42" in result["stdout"]
        assert result["duration_seconds"] >= 0

    def test_env_vars_passed(self, tmp_path):
        from cardiomas.tools.code_tools import execute_script_with_env

        script = tmp_path / "env_script.py"
        script.write_text(
            'import os\nprint("MY_VAR=" + os.environ.get("MY_VAR", "not_set"))\n'
        )

        result = execute_script_with_env.invoke({
            "script_path": str(script),
            "env_vars": {"MY_VAR": "hello_v4"},
            "timeout": 30,
        })
        assert result["exit_code"] == 0
        assert "MY_VAR=hello_v4" in result["stdout"]

    def test_captures_files(self, tmp_path):
        from cardiomas.tools.code_tools import execute_script_with_env

        script = tmp_path / "write_file.py"
        script.write_text(
            f'from pathlib import Path\n'
            f'Path("{tmp_path}/output.csv").write_text("label,count\\nNORM,100\\n")\n'
            f'print("DONE")\n'
        )

        result = execute_script_with_env.invoke({
            "script_path": str(script),
            "timeout": 30,
            "working_dir": str(tmp_path),
            "capture_files": ["output.csv"],
        })
        assert result["exit_code"] == 0
        assert "output.csv" in result["captured_files"]
        assert "NORM" in result["captured_files"]["output.csv"]

    def test_nonexistent_script(self):
        from cardiomas.tools.code_tools import execute_script_with_env

        result = execute_script_with_env.invoke({
            "script_path": "/nonexistent/script.py",
        })
        assert result["exit_code"] == -1
        assert "error" in result

    def test_timeout(self, tmp_path):
        from cardiomas.tools.code_tools import execute_script_with_env

        script = tmp_path / "slow.py"
        script.write_text("import time\ntime.sleep(60)\n")

        result = execute_script_with_env.invoke({
            "script_path": str(script),
            "timeout": 2,  # 2 second timeout
        })
        assert result["exit_code"] == -1
        assert result["status"] == "timeout"

    def test_script_with_failure(self, tmp_path):
        from cardiomas.tools.code_tools import execute_script_with_env

        script = tmp_path / "fail.py"
        script.write_text('import sys\nsys.exit(1)\n')

        result = execute_script_with_env.invoke({
            "script_path": str(script),
            "timeout": 30,
        })
        assert result["exit_code"] == 1
        assert result["status"] == "failed"
