from __future__ import annotations

import logging

from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def orchestrator_agent(state: GraphState) -> GraphState:
    """Entry node: check HF for existing splits, set up state."""
    from cardiomas import config as cfg
    from cardiomas.tools.publishing_tools import check_hf_repo

    state.execution_log.append(LogEntry(agent="orchestrator", action="start", detail=state.dataset_source))
    vprint("orchestrator", f"pipeline start — source: {state.dataset_source}")

    if not state.user_options.force_reanalysis:
        source = state.dataset_source
        dataset_name = source.rstrip("/").split("/")[-1].lower()
        vprint("orchestrator", f"checking HF cache for '{dataset_name}'…")

        result = check_hf_repo.invoke({"repo_id": cfg.HF_REPO_ID, "dataset_name": dataset_name})
        if result.get("exists"):
            state.existing_hf_splits = result.get("metadata")
            state.publish_status = "already_published"
            state.execution_log.append(
                LogEntry(agent="orchestrator", action="cache_hit", detail=dataset_name)
            )
            vprint("orchestrator", f"cache hit — {dataset_name} already on HF, skipping pipeline")
        else:
            vprint("orchestrator", "no cache hit — running full pipeline")
    else:
        vprint("orchestrator", "--force-reanalysis set — skipping cache check")

    return state
