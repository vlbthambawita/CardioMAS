from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from cardiomas.inference.base import ChatClient, ChatRequest
from cardiomas.inference.prompts import decomposer_messages
from cardiomas.schemas.config import RuntimeConfig

QueryType = Literal["factual", "computational", "exploratory"]
_COMPLEX_KEYWORDS = ["compare", "versus", " vs ", "difference between", "both", "correlation"]


@dataclass
class SubQuery:
    text: str
    query_type: QueryType = "exploratory"


def decompose(
    query: str,
    config: RuntimeConfig,
    chat_client: ChatClient | None = None,
) -> list[SubQuery]:
    """Split a complex query into atomic sub-queries.

    Returns a single-element list for simple queries.
    Capped at 4 sub-queries.
    """
    if not _is_complex(query):
        return [SubQuery(text=query, query_type=_infer_type(query))]

    if chat_client is not None and config.llm is not None:
        result = _decompose_with_llm(query, config, chat_client)
        if result:
            return result[:4]

    return [SubQuery(text=query, query_type=_infer_type(query))]


def _is_complex(query: str) -> bool:
    lower = query.lower()
    return (
        any(kw in lower for kw in _COMPLEX_KEYWORDS)
        or lower.count("?") > 1
        or (" and " in lower and lower.count(" and ") >= 2)
    )


def _decompose_with_llm(
    query: str,
    config: RuntimeConfig,
    chat_client: ChatClient,
) -> list[SubQuery] | None:
    assert config.llm is not None
    messages = decomposer_messages(query)
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
        raw_sqs = data.get("sub_queries", [])
        result: list[SubQuery] = []
        for item in raw_sqs:
            if isinstance(item, dict) and "text" in item:
                qt = str(item.get("query_type", "exploratory")).lower()
                if qt not in {"factual", "computational", "exploratory"}:
                    qt = "exploratory"
                result.append(SubQuery(text=str(item["text"]), query_type=qt))  # type: ignore[arg-type]
        if result:
            return result
    except Exception:
        pass
    return None


def _infer_type(query: str) -> QueryType:
    lower = query.lower()
    if any(kw in lower for kw in ["count", "how many", "distribution", "calculate", "compute", "analyze", "statistics"]):
        return "computational"
    if any(kw in lower for kw in ["what is", "define", "explain", "describe", "which"]):
        return "factual"
    return "exploratory"
