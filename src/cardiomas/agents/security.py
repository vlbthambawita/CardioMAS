from __future__ import annotations

import logging

from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.security.audit import run_security_audit

logger = logging.getLogger(__name__)


def security_agent(state: GraphState) -> GraphState:
    """Run security and data-leak audit on proposed splits."""
    state.execution_log.append(LogEntry(agent="security", action="start"))

    if state.proposed_splits is None:
        state.errors.append("security_agent: no proposed_splits to audit")
        return state

    splits = state.proposed_splits.splits
    audit = run_security_audit(splits)
    state.security_audit = audit

    if not audit.passed:
        for issue in audit.blocking_issues:
            state.errors.append(f"Security: {issue}")
        state.execution_log.append(LogEntry(agent="security", action="FAILED", detail=str(audit.blocking_issues)))
    else:
        state.execution_log.append(LogEntry(agent="security", action="PASSED"))

    return state
