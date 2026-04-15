from __future__ import annotations

import logging

from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.security.audit import run_security_audit
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def security_agent(state: GraphState) -> GraphState:
    """Run security and data-leak audit on proposed splits."""
    state.execution_log.append(LogEntry(agent="security", action="start"))
    vprint("security", "running PII, raw-data, and leakage audit…")

    if state.proposed_splits is None:
        state.errors.append("security_agent: no proposed_splits to audit")
        return state

    splits = state.proposed_splits.splits
    total_ids = sum(len(v) for v in splits.values())
    vprint("security", f"auditing {total_ids} record IDs across {list(splits.keys())}")

    # ── Phase 2: use real patient_record_map from DatasetMap ──────────────
    patient_mapping = _get_patient_mapping(state)
    if patient_mapping:
        vprint("security", f"using real patient mapping ({len(patient_mapping)} patients)")
    else:
        vprint("security", "no patient mapping available — leakage check limited to record IDs")

    audit = run_security_audit(splits, patient_mapping=patient_mapping)
    state.security_audit = audit

    if not audit.passed:
        for issue in audit.blocking_issues:
            state.errors.append(f"Security: {issue}")
            vprint("security", f"[red]BLOCKED: {issue}[/red]")
        state.execution_log.append(
            LogEntry(agent="security", action="FAILED", detail=str(audit.blocking_issues))
        )
    else:
        vprint("security", "audit PASSED — no PII, no raw data, no leakage")
        state.execution_log.append(LogEntry(agent="security", action="PASSED"))

    return state


def _get_patient_mapping(state: GraphState) -> dict[str, list[str]] | None:
    """Extract patient_record_map from DatasetMap if available."""
    # Check state.dataset_map directly (set by Phase 2 DatasetMapper)
    dataset_map = getattr(state, "dataset_map", None)
    if dataset_map is None:
        # Fall back to analysis_report dict (analysis agent stores it there too)
        analysis = state.analysis_report or {}
        dataset_map = analysis.get("dataset_map")

    if dataset_map is None:
        return None

    mapping = None
    if hasattr(dataset_map, "patient_record_map"):
        mapping = dataset_map.patient_record_map
    elif isinstance(dataset_map, dict):
        mapping = dataset_map.get("patient_record_map")

    return mapping if mapping else None
