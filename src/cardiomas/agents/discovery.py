from __future__ import annotations

import logging
from typing import Any

from cardiomas.agents.base import run_agent
from cardiomas.schemas.dataset import DatasetInfo, DatasetSource
from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.tools.data_tools import list_dataset_files
from cardiomas.tools.research_tools import fetch_webpage

logger = logging.getLogger(__name__)


def discovery_agent(state: GraphState) -> GraphState:
    """Identify dataset type, metadata, source, and official split info."""
    from cardiomas.llm_factory import get_llm

    opts = state.user_options
    source = state.dataset_source
    state.execution_log.append(LogEntry(agent="discovery", action="start", detail=source))

    # Try registry first
    from cardiomas.datasets.registry import get_registry
    registry = get_registry()
    for name in registry.list_names():
        info = registry.get(name)
        if info and info.source_url and source in (info.source_url, info.name, name):
            state.dataset_info = info
            state.execution_log.append(LogEntry(agent="discovery", action="registry_hit", detail=name))
            return state

    # Fetch webpage for metadata
    page_data: dict[str, Any] = {}
    if source.startswith("http"):
        page_data = fetch_webpage.invoke({"url": source})

    # Ask LLM to structure the discovery
    llm = get_llm(prefer_cloud=opts.use_cloud_llm)
    context = f"URL/path: {source}\nPage title: {page_data.get('title', '')}\nPage text excerpt: {page_data.get('text', '')[:3000]}"
    prompt = (
        "Analyze this ECG dataset source. Extract:\n"
        "1. Dataset name (short slug, e.g. ptb-xl)\n"
        "2. Source type: physionet | huggingface | local | url | kaggle\n"
        "3. Number of records if mentioned\n"
        "4. Whether official train/val/test splits exist (yes/no and where)\n"
        "5. ECG record identifier field name\n"
        "6. Sampling rate and number of leads if mentioned\n"
        "7. Associated paper URL if mentioned\n\n"
        "Respond in structured format. Cite source for every claim."
    )
    response = run_agent(llm, "discovery", prompt, context)

    # Parse LLM response to build DatasetInfo
    name = _extract_field(response, "Dataset name", source.split("/")[-1].lower()[:30])
    source_type_str = _extract_field(response, "Source type", "url").lower().strip()
    try:
        source_type = DatasetSource(source_type_str)
    except ValueError:
        source_type = DatasetSource.URL

    info = DatasetInfo(
        name=name,
        source_type=source_type,
        source_url=source if source.startswith("http") else None,
        description=page_data.get("title", ""),
    )
    state.dataset_info = info
    state.execution_log.append(LogEntry(agent="discovery", action="complete", detail=f"identified as {name}"))
    return state


def _extract_field(text: str, field: str, default: str) -> str:
    for line in text.splitlines():
        if field.lower() in line.lower():
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip().strip("`\"'").split()[0] if parts[1].strip() else default
    return default
