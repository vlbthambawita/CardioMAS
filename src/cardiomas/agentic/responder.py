from __future__ import annotations

import json

from pydantic import BaseModel, Field

from cardiomas.inference.base import ChatClient, ChatRequest
from cardiomas.inference.prompts import prompt_preview, responder_messages
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import Citation, EvidenceChunk
from cardiomas.schemas.runtime import LLMTrace


class _ResponderPayload(BaseModel):
    answer: str
    citations: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def compose_answer(
    query: str,
    config: RuntimeConfig,
    evidence: list[EvidenceChunk],
    aggregate: dict,
    warnings: list[str],
    chat_client: ChatClient | None = None,
) -> tuple[str, list[Citation], list[LLMTrace], list[str]]:
    if config.responder_uses_ollama and chat_client is not None and config.llm is not None:
        return _compose_with_ollama(query, config, evidence, aggregate, warnings, chat_client)
    if config.responder_uses_ollama and chat_client is None:
        answer, citations = _compose_deterministic(query, config, evidence, aggregate, warnings)
        return answer, citations, [], ["Ollama responder was requested, but no chat client was available; using deterministic responder."]
    answer, citations = _compose_deterministic(query, config, evidence, aggregate, warnings)
    return answer, citations, [], []


def _compose_with_ollama(
    query: str,
    config: RuntimeConfig,
    evidence: list[EvidenceChunk],
    aggregate: dict,
    warnings: list[str],
    chat_client: ChatClient,
) -> tuple[str, list[Citation], list[LLMTrace], list[str]]:
    assert config.llm is not None
    messages = responder_messages(query, evidence[: config.response.max_citations], aggregate, warnings)
    trace = LLMTrace(
        stage="responder",
        provider=config.llm.provider,
        model=config.llm.resolved_responder_model,
        prompt_preview=prompt_preview(messages),
    )

    try:
        response = chat_client.chat(
            ChatRequest(
                model=config.llm.resolved_responder_model,
                messages=messages,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                json_mode=True,
                keep_alive=config.llm.keep_alive,
            )
        )
        trace.response_preview = _trim(response.content)
        payload = _ResponderPayload.model_validate(json.loads(response.content))
        answer = payload.answer.strip()
        if not answer:
            raise ValueError("Responder returned an empty answer.")
        citations = _citations_from_indexes(payload.citations, evidence, config.response.max_citations)
        return answer, citations, [trace], [warning for warning in payload.warnings if warning]
    except Exception as exc:
        trace.ok = False
        trace.error = str(exc)
        answer, citations = _compose_deterministic(query, config, evidence, aggregate, warnings)
        fallback_warning = f"Ollama responder failed; using deterministic responder. {exc}"
        return answer, citations, [trace], [fallback_warning]


def _compose_deterministic(
    query: str,
    config: RuntimeConfig,
    evidence: list[EvidenceChunk],
    aggregate: dict,
    warnings: list[str],
) -> tuple[str, list[Citation]]:
    lines: list[str] = []
    citations: list[Citation] = []

    calculations = aggregate.get("calculations", [])
    if calculations:
        latest = calculations[-1]
        lines.append(f"Calculation result: `{latest['expression']} = {latest['result']}`.")

    dataset_info = aggregate.get("dataset_inspection")
    if dataset_info:
        lines.append(
            "Dataset inspection: "
            f"{dataset_info['total_files']} file(s), "
            f"extensions={dataset_info['extension_counts']}."
        )
        if dataset_info.get("csv_headers"):
            first_name, headers = next(iter(dataset_info["csv_headers"].items()))
            lines.append(f"Sample CSV schema from `{first_name}`: {', '.join(headers) if headers else '(empty)'}.")

    if evidence:
        lines.append("Retrieved evidence:")
        for chunk in evidence[: config.response.max_citations]:
            snippet = _clean_snippet(chunk.content)
            lines.append(f"- {snippet}")
            citations.append(chunk.citation())

    if aggregate.get("web_pages"):
        latest_page = aggregate["web_pages"][-1]
        lines.append(f"Fetched webpage title: {latest_page.get('title', latest_page.get('url', 'web page'))}.")

    if not lines:
        lines.append("I could not produce a grounded answer from the available knowledge sources and tools.")

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines), citations


def _citations_from_indexes(
    indexes: list[int],
    evidence: list[EvidenceChunk],
    limit: int,
) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[int] = set()
    for index in indexes:
        if not isinstance(index, int) or index < 1 or index > len(evidence) or index in seen:
            continue
        seen.add(index)
        citations.append(evidence[index - 1].citation())
        if len(citations) >= limit:
            break
    return citations


def _clean_snippet(text: str, limit: int = 220) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= limit:
        return snippet
    trimmed = snippet[:limit].rsplit(" ", 1)[0]
    return f"{trimmed}..."


def _trim(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    trimmed = compact[:limit].rsplit(" ", 1)[0]
    return f"{trimmed}..."
