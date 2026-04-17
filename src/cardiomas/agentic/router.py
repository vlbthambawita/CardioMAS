from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from cardiomas.inference.base import ChatClient, ChatRequest
from cardiomas.inference.prompts import router_messages
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.tools import ToolSpec

Route = Literal["code", "retrieval", "web", "orchestrate"]


@dataclass
class RouteDecision:
    route: Route
    reason: str


def route_query(
    query: str,
    config: RuntimeConfig,
    tools: list[ToolSpec],
    chat_client: ChatClient | None = None,
) -> RouteDecision:
    """Classify the query type and choose the best initial route."""
    if chat_client is not None and config.llm is not None and config.llm.resolved_router_model:
        result = _route_with_llm(query, config, tools, chat_client)
        if result is not None:
            return result
    return _route_heuristic(query, config, tools)


def _route_with_llm(
    query: str,
    config: RuntimeConfig,
    tools: list[ToolSpec],
    chat_client: ChatClient,
) -> RouteDecision | None:
    assert config.llm is not None
    messages = router_messages(query, tools)
    try:
        request = ChatRequest(
            model=config.llm.resolved_router_model,
            messages=messages,
            temperature=0.0,
            max_tokens=config.llm.router_max_tokens,
            json_mode=True,
            keep_alive=config.llm.keep_alive,
        )
        response = chat_client.chat(request)
        data = json.loads(response.content)
        route = str(data.get("route", "")).strip().lower()
        reason = str(data.get("reason", ""))
        if route in {"code", "retrieval", "web", "orchestrate"}:
            return RouteDecision(route=route, reason=reason)  # type: ignore[arg-type]
    except Exception:
        pass
    return None


def _route_heuristic(query: str, config: RuntimeConfig, tools: list[ToolSpec]) -> RouteDecision:
    available = {t.name for t in tools}
    lower = query.lower()

    if re.search(r"https?://\S+", query) and "fetch_webpage" in available:
        return RouteDecision(route="web", reason="Query contains a URL.")

    dataset_path = _has_dataset(config)
    compute_keywords = [
        "count", "how many", "unique", "distribution", "average", "mean",
        "statistics", "calculate", "compute", "analyze", "analyse", "summarize",
        "label", "class", "column", "row", "missing", "percentage", "plot",
    ]
    if dataset_path and "generate_python_artifact" in available:
        if any(kw in lower for kw in compute_keywords):
            return RouteDecision(route="code", reason="Computational dataset query detected.")

    complex_keywords = ["compare", "versus", " vs ", "difference between", "both", "and how"]
    if any(kw in lower for kw in complex_keywords) or lower.count("?") > 1:
        return RouteDecision(route="orchestrate", reason="Multi-part query detected.")

    return RouteDecision(route="retrieval", reason="Default: knowledge lookup.")


def _has_dataset(config: RuntimeConfig) -> bool:
    return any(s.path and s.kind in {"dataset_dir", "local_dir"} for s in config.sources)
