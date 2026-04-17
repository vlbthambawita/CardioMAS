from __future__ import annotations

from collections.abc import Generator
import json

from pydantic import BaseModel, Field

from cardiomas.inference.base import ChatClient, ChatRequest
from cardiomas.inference.prompts import prompt_preview, responder_messages
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import Citation, EvidenceChunk
from cardiomas.schemas.runtime import AgentEvent, LLMTrace


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
    generator = compose_answer_events(query, config, evidence, aggregate, warnings, chat_client=chat_client)
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value


def compose_answer_events(
    query: str,
    config: RuntimeConfig,
    evidence: list[EvidenceChunk],
    aggregate: dict,
    warnings: list[str],
    chat_client: ChatClient | None = None,
) -> Generator[AgentEvent, None, tuple[str, list[Citation], list[LLMTrace], list[str]]]:
    yield AgentEvent(type="status", stage="responder", message="Response synthesis started.")
    if config.responder_uses_ollama and chat_client is not None and config.llm is not None:
        result = yield from _compose_with_ollama_events(query, config, evidence, aggregate, warnings, chat_client)
        yield AgentEvent(type="status", stage="responder", message="Response synthesis finished.")
        return result
    if config.responder_uses_ollama and chat_client is None:
        answer, citations = _compose_deterministic(query, config, evidence, aggregate, warnings)
        warning = "Ollama responder was requested, but no chat client was available; using deterministic responder."
        yield AgentEvent(type="status", stage="responder", message=warning)
        yield AgentEvent(type="status", stage="responder", message="Response synthesis finished.")
        return answer, citations, [], [warning]
    answer, citations = _compose_deterministic(query, config, evidence, aggregate, warnings)
    yield AgentEvent(type="status", stage="responder", message="Response synthesis finished.")
    return answer, citations, [], []


def _compose_with_ollama_events(
    query: str,
    config: RuntimeConfig,
    evidence: list[EvidenceChunk],
    aggregate: dict,
    warnings: list[str],
    chat_client: ChatClient,
) -> Generator[AgentEvent, None, tuple[str, list[Citation], list[LLMTrace], list[str]]]:
    assert config.llm is not None
    messages = responder_messages(query, evidence[: config.response.max_citations], aggregate, warnings)
    trace = LLMTrace(
        stage="responder",
        provider=config.llm.provider,
        model=config.llm.resolved_responder_model,
        prompt_preview=prompt_preview(messages),
    )

    try:
        request = ChatRequest(
            model=config.llm.resolved_responder_model,
            messages=messages,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            json_mode=True,
            keep_alive=config.llm.keep_alive,
        )
        yield AgentEvent(
            type="llm_stream_start",
            stage="responder",
            message="Responder LLM stream started.",
            data={"model": config.llm.resolved_responder_model},
        )
        streamed_content = ""
        for chunk in chat_client.chat_stream(request):
            if chunk.content:
                streamed_content += chunk.content
                yield AgentEvent(
                    type="llm_token",
                    stage="responder",
                    content=chunk.content,
                    data={"model": chunk.model},
                )
        yield AgentEvent(
            type="llm_stream_end",
            stage="responder",
            message="Responder LLM stream ended.",
            data={"model": config.llm.resolved_responder_model},
        )
        trace.response_preview = _trim(streamed_content)
        payload = _ResponderPayload.model_validate(json.loads(streamed_content))
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
        yield AgentEvent(type="status", stage="responder", message=fallback_warning)
        return answer, citations, [trace], [fallback_warning]


def _compose_with_ollama(
    query: str,
    config: RuntimeConfig,
    evidence: list[EvidenceChunk],
    aggregate: dict,
    warnings: list[str],
    chat_client: ChatClient,
) -> tuple[str, list[Citation], list[LLMTrace], list[str]]:
    generator = _compose_with_ollama_events(query, config, evidence, aggregate, warnings, chat_client)
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value


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

    generated_python_artifacts = aggregate.get("generated_python_artifacts", [])
    if generated_python_artifacts:
        latest_artifact = generated_python_artifacts[-1]
        lines.append(
            "Generated analysis artifact: "
            f"`{latest_artifact.get('artifact_entrypoint', '')}` "
            f"mode={latest_artifact.get('mode', latest_artifact.get('value', 'unknown'))}."
        )
        if latest_artifact.get("selected_path"):
            lines.append(f"Selected file: `{latest_artifact['selected_path']}`.")
        if latest_artifact.get("columns"):
            lines.append(f"Detected columns: {', '.join(latest_artifact['columns'][:12])}.")
        if "total_rows" in latest_artifact:
            lines.append(
                "Generated analysis metrics: "
                f"rows={latest_artifact.get('total_rows', 0)}, "
                f"missing_fraction={latest_artifact.get('missing_fraction', 0.0)}."
            )
        class_counts = latest_artifact.get("class_counts") or {}
        if isinstance(class_counts, dict) and class_counts.get("counts"):
            lines.append(f"Detected class counts from `{class_counts.get('column', 'label')}`: {class_counts['counts']}.")
        preview_lines = latest_artifact.get("preview_lines") or []
        if preview_lines:
            lines.append(f"Preview: {' | '.join(preview_lines[:3])}.")
        sample_files = latest_artifact.get("sample_files") or []
        if sample_files:
            lines.append(f"Sample files: {', '.join(sample_files[:8])}.")

    generated_shell_artifacts = aggregate.get("generated_shell_artifacts", [])
    if generated_shell_artifacts:
        latest_script = generated_shell_artifacts[-1]
        execution_note = "executed" if latest_script.get("executed") else "saved"
        lines.append(f"Generated shell artifact ({execution_note}): `{latest_script.get('script_path', '')}`.")

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
