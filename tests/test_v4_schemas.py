"""
Tests for V4 Pydantic models: ScriptRecord, ExecutionResult,
RefinementContext, ApprovalSummary, and new UserOptions/GraphState fields.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from cardiomas.schemas.state import (
    ApprovalSummary,
    ExecutionResult,
    GraphState,
    LogEntry,
    RefinementContext,
    ScriptRecord,
    UserOptions,
)


# ── ScriptRecord ──────────────────────────────────────────────────────────

class TestScriptRecord:
    def test_minimal_construction(self):
        rec = ScriptRecord(
            name="00_explore_structure.py",
            path="/tmp/v4/scripts/subset/00_explore_structure.py",
            purpose="Walk directory tree",
            output_dir="/tmp/v4/outputs/subset",
        )
        assert rec.name == "00_explore_structure.py"
        assert rec.phase == "subset"
        assert rec.timeout == 300
        assert rec.sha256 == ""

    def test_full_construction(self):
        rec = ScriptRecord(
            name="04_generate_splits_full.py",
            path="/tmp/v4/scripts/full/04_generate_splits_full.py",
            purpose="Full dataset splits",
            output_dir="/tmp/v4/outputs/full",
            timeout=600,
            phase="full",
            sha256="abc123",
        )
        assert rec.phase == "full"
        assert rec.timeout == 600
        assert rec.sha256 == "abc123"

    def test_serialization_roundtrip(self):
        rec = ScriptRecord(
            name="00_explore.py",
            path="/tmp/script.py",
            purpose="test",
            output_dir="/tmp/out",
        )
        data = rec.model_dump(mode="json")
        assert "generated_at" in data
        rec2 = ScriptRecord(**data)
        assert rec2.name == rec.name


# ── ExecutionResult ───────────────────────────────────────────────────────

class TestExecutionResult:
    def test_success_result(self):
        result = ExecutionResult(
            script_name="00_explore_structure.py",
            exit_code=0,
            stdout="TOTAL_FILES=42\nEXTENSIONS=['.csv', '.dat']\nROOT=/data",
            stderr="",
            duration_seconds=1.23,
            verification_passed=True,
            verification_notes="TOTAL_FILES=42",
        )
        assert result.exit_code == 0
        assert result.verification_passed is True
        assert result.duration_seconds == 1.23

    def test_failure_result(self):
        result = ExecutionResult(
            script_name="01_extract_metadata.py",
            exit_code=1,
            stdout="",
            stderr="FileNotFoundError: /data/metadata.csv",
            duration_seconds=0.01,
            verification_passed=False,
            verification_notes="Script exited with code 1",
        )
        assert result.exit_code == 1
        assert result.verification_passed is False

    def test_generated_files_dict(self):
        result = ExecutionResult(
            script_name="02_compute_statistics.py",
            exit_code=0,
            generated_files={
                "stats.csv": "label,count\nNORM,5000\nAFIB,2000\n",
            },
            verification_passed=True,
        )
        assert "stats.csv" in result.generated_files
        assert "NORM" in result.generated_files["stats.csv"]

    def test_serialization_roundtrip(self):
        result = ExecutionResult(
            script_name="03_generate_splits_subset.py",
            exit_code=0,
        )
        data = result.model_dump(mode="json")
        result2 = ExecutionResult(**data)
        assert result2.script_name == result.script_name


# ── RefinementContext ─────────────────────────────────────────────────────

class TestRefinementContext:
    def test_construction(self):
        ctx = RefinementContext(
            failed_script="01_extract_metadata.py",
            error_message="KeyError: 'patient_id'",
            stdout_excerpt="COLUMNS=['ecg_id', 'label']\n",
            suggested_fix="Use 'ecg_id' as the ID field",
            attempt=1,
        )
        assert ctx.failed_script == "01_extract_metadata.py"
        assert ctx.attempt == 1

    def test_defaults(self):
        ctx = RefinementContext(
            failed_script="00_explore.py",
            error_message="something broke",
            stdout_excerpt="",
        )
        assert ctx.suggested_fix == ""
        assert ctx.attempt == 1

    def test_serialization(self):
        ctx = RefinementContext(
            failed_script="02_stats.py",
            error_message="Error",
            stdout_excerpt="partial output",
            attempt=2,
        )
        data = ctx.model_dump(mode="json")
        ctx2 = RefinementContext(**data)
        assert ctx2.attempt == 2


# ── ApprovalSummary ───────────────────────────────────────────────────────

class TestApprovalSummary:
    def test_construction(self):
        summary = ApprovalSummary(
            dataset_name="ptb-xl",
            subset_size=100,
            records_found=21837,
            columns_found=["ecg_id", "patient_id", "scp_codes", "age", "sex"],
            label_distribution_excerpt="NORM,5000\nAFIB,2000",
            split_sizes={"train": 70, "val": 15, "test": 15},
            scripts_passed=["00_explore_structure.py", "01_extract_metadata.py"],
            scripts_failed=[],
            output_dir="/tmp/output/ptb-xl/v4",
        )
        assert summary.dataset_name == "ptb-xl"
        assert summary.records_found == 21837
        assert len(summary.columns_found) == 5
        assert summary.split_sizes["train"] == 70

    def test_defaults(self):
        summary = ApprovalSummary(
            dataset_name="test",
            subset_size=100,
            records_found=0,
        )
        assert summary.columns_found == []
        assert summary.scripts_failed == []

    def test_serialization(self):
        summary = ApprovalSummary(
            dataset_name="test",
            subset_size=50,
            records_found=1000,
        )
        data = summary.model_dump(mode="json")
        summary2 = ApprovalSummary(**data)
        assert summary2.dataset_name == "test"


# ── UserOptions V4 fields ─────────────────────────────────────────────────

class TestUserOptionsV4:
    def test_v4_defaults(self):
        opts = UserOptions(dataset_source="https://example.com/ecg")
        assert opts.v4_auto_approve is False
        assert opts.v4_subset_size == 100
        assert opts.v4_max_refinements == 2
        assert opts.v4_skip_ecg_stats is False
        assert opts.v4_plot_format == "png"

    def test_v4_overrides(self):
        opts = UserOptions(
            dataset_source="/data/ecg",
            v4_auto_approve=True,
            v4_subset_size=500,
            v4_max_refinements=3,
            v4_skip_ecg_stats=True,
            v4_plot_format="pdf",
        )
        assert opts.v4_auto_approve is True
        assert opts.v4_subset_size == 500
        assert opts.v4_skip_ecg_stats is True
        assert opts.v4_plot_format == "pdf"

    def test_backwards_compatible_with_v2_fields(self):
        opts = UserOptions(
            dataset_source="https://example.com",
            requirement="80/10/10 split",
            agent_llm_map={"coder": "deepseek-coder:6.7b"},
        )
        assert opts.requirement == "80/10/10 split"
        assert opts.v4_auto_approve is False  # default


# ── GraphState V4 fields ──────────────────────────────────────────────────

class TestGraphStateV4:
    def test_v4_defaults(self):
        state = GraphState(dataset_source="/data/ecg")
        assert state.v4_output_dir == ""
        assert state.v4_subset_size == 100
        assert state.v4_generated_scripts == {}
        assert state.v4_execution_results == []
        assert state.v4_execution_summary == ""
        assert state.v4_generated_files == {}
        assert state.v4_pipeline_phase == "subset_validation"
        assert state.v4_subset_validated is False
        assert state.v4_approval_status == "pending"
        assert state.v4_approval_summary is None
        assert state.v4_refinement_context is None
        assert state.v4_refinement_rounds == {}
        assert state.v4_ecg_stats_dir == ""
        assert state.v4_ecg_stats_scripts == {}

    def test_v4_script_records(self):
        rec = ScriptRecord(
            name="00_explore.py",
            path="/tmp/script.py",
            purpose="explore",
            output_dir="/tmp/out",
        )
        state = GraphState(
            dataset_source="/data",
            v4_generated_scripts={"00_explore.py": rec},
        )
        assert "00_explore.py" in state.v4_generated_scripts
        assert state.v4_generated_scripts["00_explore.py"].name == "00_explore.py"

    def test_v4_execution_results_list(self):
        r1 = ExecutionResult(script_name="00_explore.py", exit_code=0, verification_passed=True)
        r2 = ExecutionResult(script_name="01_metadata.py", exit_code=0, verification_passed=True)
        state = GraphState(
            dataset_source="/data",
            v4_execution_results=[r1, r2],
        )
        assert len(state.v4_execution_results) == 2

    def test_v4_refinement_context(self):
        ctx = RefinementContext(
            failed_script="01_metadata.py",
            error_message="Error",
            stdout_excerpt="",
        )
        state = GraphState(
            dataset_source="/data",
            v4_refinement_context=ctx,
        )
        assert state.v4_refinement_context.failed_script == "01_metadata.py"

    def test_full_serialization_roundtrip(self):
        """GraphState with V4 fields should survive model_dump/reconstruction."""
        ctx = RefinementContext(
            failed_script="00.py",
            error_message="err",
            stdout_excerpt="partial",
        )
        summary = ApprovalSummary(
            dataset_name="test",
            subset_size=100,
            records_found=500,
        )
        state = GraphState(
            dataset_source="/data/ecg",
            user_options=UserOptions(dataset_source="/data/ecg"),
            v4_pipeline_phase="subset_validation",
            v4_subset_validated=False,
            v4_approval_status="pending",
            v4_refinement_context=ctx,
            v4_approval_summary=summary,
            v4_generated_files={"stats.csv": "label,count\nNORM,100\n"},
        )
        data = state.model_dump(mode="json")
        state2 = GraphState(**data)
        assert state2.v4_pipeline_phase == "subset_validation"
        assert state2.v4_refinement_context.failed_script == "00.py"
        assert state2.v4_approval_summary.dataset_name == "test"
        assert "stats.csv" in state2.v4_generated_files

    def test_v4_fields_do_not_break_v3_fields(self):
        """Ensure V4 fields coexist with V3 fields."""
        from cardiomas.schemas.dataset import DatasetInfo, DatasetSource
        info = DatasetInfo(
            name="ptb-xl",
            source_type=DatasetSource.PHYSIONET,
            description="PTB-XL",
        )
        state = GraphState(
            dataset_source="https://physionet.org/ptb-xl",
            dataset_info=info,
            v4_output_dir="/tmp/output/ptb-xl/v4",
            v4_subset_size=200,
        )
        assert state.dataset_info.name == "ptb-xl"
        assert state.v4_output_dir == "/tmp/output/ptb-xl/v4"
        assert state.dataset_map is None  # V3 field still defaults to None
