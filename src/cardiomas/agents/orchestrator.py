from __future__ import annotations

import logging

from cardiomas.schemas.state import GraphState, LogEntry

logger = logging.getLogger(__name__)


def orchestrator_agent(state: GraphState) -> GraphState:
    """Entry node: check HF for existing splits, set up state."""
    from cardiomas import config as cfg
    from cardiomas.tools.publishing_tools import check_hf_repo

    state.execution_log.append(LogEntry(agent="orchestrator", action="start", detail=state.dataset_source))

    if not state.user_options.force_reanalysis:
        # Try to infer dataset name for registry lookup
        source = state.dataset_source
        dataset_name = source.rstrip("/").split("/")[-1].lower()

        result = check_hf_repo.invoke({"repo_id": cfg.HF_REPO_ID, "dataset_name": dataset_name})
        if result.get("exists"):
            state.existing_hf_splits = result.get("metadata")
            state.publish_status = "already_published"
            state.execution_log.append(
                LogEntry(agent="orchestrator", action="cache_hit", detail=dataset_name)
            )

    return state
