from __future__ import annotations

import json

from cardiomas.inference.base import ChatMessage
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.tools import ToolSpec


def planner_messages(
    query: str,
    tools: list[ToolSpec],
    dataset_path: str,
    urls: list[str],
    expression: str,
) -> list[ChatMessage]:
    tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools) or "- none"
    context = {
        "query": query,
        "default_dataset_path": dataset_path,
        "urls_in_query": urls,
        "expression_hint": expression,
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You are the CardioMAS planner. Select only from the provided tools. "
                "Return strict JSON with keys: strategy, steps, notes. "
                "Each step must contain tool_name, reason, and args. "
                "Do not invent tools or filesystem paths."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "Available tools:\n"
                f"{tool_lines}\n\n"
                "Planning context:\n"
                f"{json.dumps(context, indent=2)}\n\n"
                "Rules:\n"
                "- Use retrieve_corpus for grounded answers when available.\n"
                "- Use inspect_dataset only for dataset/file/metadata questions.\n"
                "- Use calculate only for explicit arithmetic.\n"
                "- Use fetch_webpage only for direct URLs already present in the query.\n"
                "- Prefer the default_dataset_path when inspect_dataset is needed.\n"
                "- Return compact JSON only."
            ),
        ),
    ]


def responder_messages(
    query: str,
    evidence: list[EvidenceChunk],
    aggregate: dict,
    warnings: list[str],
) -> list[ChatMessage]:
    evidence_lines = []
    for index, chunk in enumerate(evidence, start=1):
        evidence_lines.append(
            json.dumps(
                {
                    "id": index,
                    "chunk_id": chunk.chunk_id,
                    "source_label": chunk.source_label,
                    "locator": chunk.metadata.get("chunk_label") or chunk.title or chunk.uri,
                    "content": _trim(chunk.content, 500),
                },
                ensure_ascii=True,
            )
        )

    tool_context = {
        "dataset_inspection": aggregate.get("dataset_inspection"),
        "calculations": aggregate.get("calculations", []),
        "web_pages": aggregate.get("web_pages", []),
        "warnings": warnings,
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You are the grounded CardioMAS responder. Answer only with support from the provided evidence "
                "and tool outputs. If the evidence is insufficient, say so plainly. "
                "Return strict JSON with keys: answer, citations, warnings. "
                "Citations must be a list of integer evidence ids."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Question:\n{query}\n\n"
                "Evidence items:\n"
                f"{chr(10).join(evidence_lines) if evidence_lines else '(none)'}\n\n"
                "Tool context:\n"
                f"{json.dumps(tool_context, indent=2, default=str)}\n\n"
                "Write a concise grounded answer. Return JSON only."
            ),
        ),
    ]


def prompt_preview(messages: list[ChatMessage], limit: int = 400) -> str:
    text = "\n".join(f"{message.role.upper()}: {message.content}" for message in messages)
    return _trim(text, limit)


def _trim(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    trimmed = compact[:limit].rsplit(" ", 1)[0]
    return f"{trimmed}..."
