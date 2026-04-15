from __future__ import annotations

import logging

from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.publishing.hf_publisher import publish_to_hf
from cardiomas.publishing.github_updater import update_github_page
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def publisher_agent(state: GraphState) -> GraphState:
    """Push split manifest to HuggingFace and update GitHub README."""
    state.execution_log.append(LogEntry(agent="publisher", action="start"))
    dry = state.user_options.dry_run
    vprint("publisher", f"{'[dry-run] ' if dry else ''}publishing to HF: {state.proposed_splits.dataset_name if state.proposed_splits else '?'}")

    if state.proposed_splits is None:
        state.errors.append("publisher_agent: no proposed_splits to publish")
        return state

    if state.security_audit and not state.security_audit.passed:
        state.errors.append("publisher_agent: blocked by security audit failure")
        state.publish_status = "blocked"
        return state

    dry_run = state.user_options.dry_run
    result = publish_to_hf(state.proposed_splits, dry_run=dry_run)

    if result["status"] in ("ok", "dry_run"):
        hf_url = result.get("url", "")
        state.publish_status = result["status"]
        gh_result = update_github_page(state.proposed_splits.dataset_name, hf_url, dry_run=dry_run)
        vprint("publisher", f"HF upload: {result['status']} — {hf_url}")
        vprint("publisher", f"GitHub README update: {gh_result.get('status', '?')}")
        state.execution_log.append(
            LogEntry(agent="publisher", action="complete", detail=f"HF: {hf_url}")
        )
    else:
        state.errors.append(f"publisher_agent: {result.get('error', 'unknown error')}")
        state.publish_status = "failed"
        vprint("publisher", f"FAILED: {result.get('error', '?')}")
        state.execution_log.append(LogEntry(agent="publisher", action="failed", detail=result.get("error", "")))

    return state
