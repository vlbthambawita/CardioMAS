"""
Tests for the Approval Gate (V4) and V4 orchestrator routing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cardiomas.schemas.state import (
    ApprovalSummary,
    ExecutionResult,
    GraphState,
    RefinementContext,
    ScriptRecord,
    UserOptions,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _state(
    approval_status: str = "pending",
    auto_approve: bool = False,
    subset_validated: bool = True,
    pipeline_phase: str = "subset_validation",
) -> GraphState:
    opts = UserOptions(
        dataset_source="/data/ptb-xl",
        v4_auto_approve=auto_approve,
        v4_subset_size=100,
    )
    return GraphState(
        dataset_source="/data/ptb-xl",
        user_options=opts,
        v4_approval_status=approval_status,
        v4_pipeline_phase=pipeline_phase,
        v4_subset_validated=subset_validated,
        v4_output_dir="/tmp/v4_output",
    )


# ── Approval Gate Tests ───────────────────────────────────────────────────

class TestApprovalGateNode:
    def test_approved_status_routes_to_executor(self):
        """With approval_status='approved', gate should set next_agent='executor'."""
        from cardiomas.agents.approval_gate import approval_gate_node

        state = _state(approval_status="approved")
        result = approval_gate_node(state)

        assert result.next_agent == "executor"
        assert result.v4_pipeline_phase == "full_run"

    def test_rejected_routes_to_end_saved(self):
        """With approval_status='rejected', gate should set next_agent='end_saved'."""
        from cardiomas.agents.approval_gate import approval_gate_node

        state = _state(approval_status="rejected")
        result = approval_gate_node(state)

        assert result.next_agent == "end_saved"

    def test_pending_with_auto_approve(self):
        """With auto_approve=True and pending, gate should auto-approve."""
        from cardiomas.agents.approval_gate import approval_gate_node

        state = _state(approval_status="pending", auto_approve=True)
        result = approval_gate_node(state)

        assert result.v4_approval_status == "approved"
        assert result.next_agent == "executor"
        assert result.v4_pipeline_phase == "full_run"

    def test_pending_without_auto_approve_saves_checkpoint(self, tmp_path):
        """With pending status and no auto-approve, gate saves checkpoint and ends."""
        from cardiomas.agents.approval_gate import approval_gate_node

        state = _state(approval_status="pending", auto_approve=False)
        state.v4_output_dir = str(tmp_path / "v4")
        Path(state.v4_output_dir).mkdir(parents=True, exist_ok=True)

        result = approval_gate_node(state)

        assert result.next_agent == "end_saved"
        assert result.v4_approval_summary is not None

    def test_approved_sets_full_run_phase(self):
        """Approval must set v4_pipeline_phase to full_run."""
        from cardiomas.agents.approval_gate import approval_gate_node

        state = _state(approval_status="approved")
        result = approval_gate_node(state)

        assert result.v4_pipeline_phase == "full_run"

    def test_approved_logs_event(self):
        """Gate should log the approval decision."""
        from cardiomas.agents.approval_gate import approval_gate_node

        state = _state(approval_status="approved")
        result = approval_gate_node(state)

        gate_entries = [e for e in result.execution_log if e.agent == "approval_gate"]
        assert any(e.action == "approved" for e in gate_entries)

    def test_rejected_logs_event(self):
        from cardiomas.agents.approval_gate import approval_gate_node

        state = _state(approval_status="rejected")
        result = approval_gate_node(state)

        gate_entries = [e for e in result.execution_log if e.agent == "approval_gate"]
        assert any(e.action == "rejected" for e in gate_entries)


class TestBuildApprovalSummary:
    def test_extracts_records_from_explore_output(self):
        """Summary should extract TOTAL_FILES from script 00 stdout."""
        from cardiomas.agents.approval_gate import _build_approval_summary

        r00 = ExecutionResult(
            script_name="00_explore_structure.py",
            exit_code=0,
            stdout="TOTAL_FILES=21837\nEXTENSIONS=['.csv']\nROOT=/data\n",
            verification_passed=True,
        )
        state = _state()
        state.v4_execution_results = [r00]
        state.dataset_source = "/data/ptb-xl"

        summary = _build_approval_summary(state)
        assert summary.records_found == 21837

    def test_extracts_columns_from_metadata_output(self):
        """Summary should parse COLUMNS= from script 01 stdout."""
        from cardiomas.agents.approval_gate import _build_approval_summary

        r01 = ExecutionResult(
            script_name="01_extract_metadata.py",
            exit_code=0,
            stdout='COLUMNS=["ecg_id", "patient_id", "scp_codes"]\nDTYPES={}\n',
            verification_passed=True,
        )
        state = _state()
        state.v4_execution_results = [r01]

        summary = _build_approval_summary(state)
        assert "ecg_id" in summary.columns_found

    def test_extracts_split_sizes_from_json(self):
        """Summary should parse split sizes from splits_subset.json."""
        from cardiomas.agents.approval_gate import _build_approval_summary

        splits_data = {"splits": {"train": [f"r{i}" for i in range(70)],
                                   "val": [f"v{i}" for i in range(15)],
                                   "test": [f"t{i}" for i in range(15)]}}
        state = _state()
        state.v4_generated_files = {"splits_subset.json": json.dumps(splits_data)}

        summary = _build_approval_summary(state)
        assert summary.split_sizes["train"] == 70
        assert summary.split_sizes["val"] == 15
        assert summary.split_sizes["test"] == 15

    def test_tracks_passed_and_failed_scripts(self):
        """Summary should categorise scripts by pass/fail."""
        from cardiomas.agents.approval_gate import _build_approval_summary

        results = [
            ExecutionResult(script_name="00_explore.py", exit_code=0, verification_passed=True),
            ExecutionResult(script_name="01_metadata.py", exit_code=1, verification_passed=False),
        ]
        state = _state()
        state.v4_execution_results = results

        summary = _build_approval_summary(state)
        assert "00_explore.py" in summary.scripts_passed
        assert "01_metadata.py" in summary.scripts_failed


# ── Orchestrator V4 routing ───────────────────────────────────────────────

class TestOrchestratorV4Routing:
    def _make_state(self, last_agent: str, **kwargs) -> GraphState:
        opts = UserOptions(
            dataset_source="/data/ptb-xl",
            local_path="/data/ptb-xl",
            **{k: v for k, v in kwargs.items() if k in UserOptions.model_fields},
        )
        state = GraphState(
            dataset_source="/data/ptb-xl",
            user_options=opts,
            last_completed_agent=last_agent,
            **{k: v for k, v in kwargs.items() if k not in UserOptions.model_fields and k in GraphState.model_fields},
        )
        return state

    def test_paper_routes_to_data_engineer(self):
        from cardiomas.agents.orchestrator import _route

        state = self._make_state("paper")
        next_a, reason = _route(state, "paper")
        assert next_a == "data_engineer"

    def test_data_engineer_routes_to_executor(self):
        from cardiomas.agents.orchestrator import _route

        state = self._make_state("data_engineer")
        next_a, reason = _route(state, "data_engineer")
        assert next_a == "executor"

    def test_executor_subset_validated_routes_to_analysis(self):
        from cardiomas.agents.orchestrator import _route

        state = self._make_state(
            "executor",
            v4_pipeline_phase="subset_validation",
            v4_subset_validated=True,
        )
        next_a, reason = _route(state, "executor")
        assert next_a == "analysis"

    def test_executor_subset_failed_routes_to_data_engineer(self):
        from cardiomas.agents.orchestrator import _route

        ctx = RefinementContext(
            failed_script="01_meta.py",
            error_message="error",
            stdout_excerpt="",
        )
        state = self._make_state(
            "executor",
            v4_pipeline_phase="subset_validation",
            v4_subset_validated=False,
        )
        state.errors.append("executor: script '01_meta.py' failed")
        state.v4_refinement_context = ctx
        state.retry_counts = {}

        next_a, reason = _route(state, "executor")
        assert next_a == "data_engineer"

    def test_analysis_subset_routes_to_approval_gate(self):
        from cardiomas.agents.orchestrator import _route

        state = self._make_state(
            "analysis",
            v4_pipeline_phase="subset_validation",
            v4_subset_validated=True,
        )
        next_a, reason = _route(state, "analysis")
        assert next_a == "approval_gate"

    def test_analysis_full_run_routes_to_ecg_stats(self):
        from cardiomas.agents.orchestrator import _route

        state = self._make_state(
            "analysis",
            v4_pipeline_phase="full_run",
        )
        # skip_ecg_stats defaults to False
        next_a, reason = _route(state, "analysis")
        assert next_a == "ecg_stats"

    def test_analysis_full_run_skip_ecg_stats(self):
        from cardiomas.agents.orchestrator import _route

        opts = UserOptions(
            dataset_source="/data",
            local_path="/data",
            v4_skip_ecg_stats=True,
        )
        state = GraphState(
            dataset_source="/data",
            user_options=opts,
            last_completed_agent="analysis",
            v4_pipeline_phase="full_run",
        )
        next_a, reason = _route(state, "analysis")
        assert next_a == "splitter"

    def test_ecg_stats_routes_to_executor(self):
        from cardiomas.agents.orchestrator import _route

        state = self._make_state("ecg_stats")
        next_a, reason = _route(state, "ecg_stats")
        assert next_a == "executor"

    def test_executor_ecg_stats_run_routes_to_splitter(self):
        from cardiomas.agents.orchestrator import _route

        state = self._make_state(
            "executor",
            v4_pipeline_phase="ecg_stats_run",
        )
        next_a, reason = _route(state, "executor")
        assert next_a == "splitter"


# ── Splitter V4: read from splits.json ───────────────────────────────────

class TestSplitterV4Loading:
    def test_loads_from_v4_splits_json(self):
        """Splitter should use splits.json from v4_generated_files."""
        from cardiomas.agents.splitter import _load_record_ids

        splits_data = {
            "splits": {
                "train": [f"r{i}" for i in range(700)],
                "val": [f"v{i}" for i in range(150)],
                "test": [f"t{i}" for i in range(150)],
            }
        }
        state = GraphState(
            dataset_source="/data",
            v4_generated_files={"splits.json": json.dumps(splits_data)},
        )

        ids = _load_record_ids(state, {})
        assert len(ids) == 1000
        assert "r0" in ids
        assert "v0" in ids

    def test_deduplicates_ids(self):
        """Splitter should deduplicate IDs from all splits."""
        from cardiomas.agents.splitter import _load_record_ids

        splits_data = {
            "splits": {
                "train": ["r1", "r2", "r3"],
                "val": ["r4", "r5"],
                # deliberate dup
                "test": ["r5", "r6"],
            }
        }
        state = GraphState(
            dataset_source="/data",
            v4_generated_files={"splits.json": json.dumps(splits_data)},
        )

        ids = _load_record_ids(state, {})
        assert len(ids) == 6  # r5 deduped
        assert ids == sorted(set(ids))  # should be sorted

    def test_falls_back_to_v3_when_no_v4_data(self):
        """Without V4 data, splitter falls back to V3 DatasetMap."""
        from cardiomas.agents.splitter import _load_record_ids

        dataset_map = {"all_record_ids": ["rec1", "rec2", "rec3"]}
        analysis = {"dataset_map": dataset_map}
        state = GraphState(
            dataset_source="/data",
            v4_generated_files={},  # no V4 data
        )

        ids = _load_record_ids(state, analysis)
        assert ids == ["rec1", "rec2", "rec3"]


# ── Security V4: patient_map.json ─────────────────────────────────────────

class TestSecurityV4PatientMapping:
    def test_reads_patient_map_from_v4_files(self):
        from cardiomas.agents.security import _get_patient_mapping_v4

        patient_map = {
            "patient_001": ["record_001", "record_002"],
            "patient_002": ["record_003"],
        }
        state = GraphState(
            dataset_source="/data",
            v4_generated_files={"patient_map.json": json.dumps(patient_map)},
        )

        mapping = _get_patient_mapping_v4(state)
        assert mapping is not None
        assert "patient_001" in mapping
        assert len(mapping["patient_001"]) == 2

    def test_returns_none_when_no_patient_map(self):
        from cardiomas.agents.security import _get_patient_mapping_v4

        state = GraphState(
            dataset_source="/data",
            v4_generated_files={},  # no patient_map.json
        )

        mapping = _get_patient_mapping_v4(state)
        assert mapping is None

    def test_returns_none_on_invalid_json(self):
        from cardiomas.agents.security import _get_patient_mapping_v4

        state = GraphState(
            dataset_source="/data",
            v4_generated_files={"patient_map.json": "not json {{{{"},
        )

        mapping = _get_patient_mapping_v4(state)
        assert mapping is None

    def test_prefers_v4_over_v3(self):
        """_get_patient_mapping should prefer V4 data over V3 DatasetMap."""
        from cardiomas.agents.security import _get_patient_mapping

        v4_map = {"p1": ["r1", "r2"]}
        v3_map = {"old_patient": ["old_record"]}
        state = GraphState(
            dataset_source="/data",
            v4_generated_files={"patient_map.json": json.dumps(v4_map)},
            analysis_report={"dataset_map": {"patient_record_map": v3_map}},
        )

        mapping = _get_patient_mapping(state)
        assert "p1" in mapping
        assert "old_patient" not in mapping
