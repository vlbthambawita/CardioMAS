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

    audit = run_security_audit(splits)
    state.security_audit = audit

    if not audit.passed:
        for issue in audit.blocking_issues:
            state.errors.append(f"Security: {issue}")
            vprint("security", f"BLOCKED: {issue}")
        state.execution_log.append(LogEntry(agent="security", action="FAILED", detail=str(audit.blocking_issues)))
    else:
        vprint("security", "audit PASSED — no PII, no raw data, no leakage")
        state.execution_log.append(LogEntry(agent="security", action="PASSED"))

    return state
