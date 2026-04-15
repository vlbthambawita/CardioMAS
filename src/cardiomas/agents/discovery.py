from __future__ import annotations

import logging
from typing import Any

from cardiomas.agents.base import AgentOutputError, run_structured_agent
from cardiomas.schemas.agent_outputs import DiscoveryOutput
from cardiomas.schemas.dataset import DatasetInfo, DatasetSource
from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.tools.research_tools import fetch_webpage
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def discovery_agent(state: GraphState) -> GraphState:
    """Identify dataset type, metadata, source, and official split info."""
    from cardiomas.llm_factory import get_llm_for_agent

    opts = state.user_options
    source = state.dataset_source
    state.execution_log.append(LogEntry(agent="discovery", action="start", detail=source))
    vprint("discovery", f"starting — source: {source}")

    # Try registry first
    from cardiomas.datasets.registry import get_registry
    registry = get_registry()
    for name in registry.list_names():
        info = registry.get(name)
        if info and info.source_url and source in (info.source_url, info.name, name):
            state.dataset_info = info
            state.execution_log.append(LogEntry(agent="discovery", action="registry_hit", detail=name))
            vprint("discovery", f"registry hit → {name}")
            return state

    # Fetch webpage for metadata
    page_data: dict[str, Any] = {}
    if source.startswith("http"):
        vprint("discovery", f"fetching webpage: {source}")
        page_data = fetch_webpage.invoke({"url": source})
        vprint("discovery", f"page title: {page_data.get('title', '(none)')}")

    # Build context for LLM
    context = (
        f"URL/path: {source}\n"
        f"Page title: {page_data.get('title', '')}\n"
        f"Page text excerpt:\n{page_data.get('text', '')[:3000]}"
    )
    prompt = (
        "Analyze this ECG dataset source and extract all available metadata.\n"
        "Be precise: only report what is explicitly stated in the page text.\n"
        "For 'source_type' use exactly one of: physionet, huggingface, local, url, kaggle.\n"
        "For 'ecg_id_field' identify the column that uniquely identifies each ECG recording.\n"
        "Set 'official_splits' to true only if the dataset explicitly provides predefined splits."
    )

    llm = get_llm_for_agent(
        "discovery",
        prefer_cloud=opts.use_cloud_llm,
        agent_llm_map=opts.agent_llm_map,
    )

    try:
        output: DiscoveryOutput = run_structured_agent(
            llm, "discovery", prompt, DiscoveryOutput, extra_context=context
        )
    except AgentOutputError as exc:
        logger.error(f"discovery_agent: structured output failed — {exc}")
        state.errors.append(f"discovery: {exc}")
        # Build minimal DatasetInfo so pipeline can continue
        output = DiscoveryOutput(
            dataset_name=source.rstrip("/").split("/")[-1].lower()[:30] or "unknown",
            source_type="url" if source.startswith("http") else "local",
        )

    try:
        source_type = DatasetSource(output.source_type)
    except ValueError:
        source_type = DatasetSource.URL

    info = DatasetInfo(
        name=output.dataset_name,
        source_type=source_type,
        source_url=source if source.startswith("http") else None,
        description=page_data.get("title", ""),
        paper_url=output.paper_url,
        official_splits=output.official_splits,
        num_records=output.num_records,
        ecg_id_field=output.ecg_id_field,
        sampling_rate=output.sampling_rate_hz,
        num_leads=output.num_leads,
    )
    state.dataset_info = info

    state.execution_log.append(
        LogEntry(agent="discovery", action="complete", detail=f"identified as {output.dataset_name}")
    )
    vprint(
        "discovery",
        f"complete — '{output.dataset_name}' ({source_type.value})"
        f" id_field={output.ecg_id_field}"
        f" official_splits={output.official_splits}",
    )
    return state
