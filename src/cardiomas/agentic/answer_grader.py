from __future__ import annotations

import json
from typing import Literal

from cardiomas.inference.base import ChatClient, ChatRequest
from cardiomas.inference.prompts import answer_grader_messages
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import EvidenceChunk

AnswerVerdict = Literal["grounded", "hallucinated", "incomplete"]


def grade_answer(
    query: str,
    answer: str,
    evidence: list[EvidenceChunk],
    config: RuntimeConfig,
    chat_client: ChatClient | None = None,
) -> AnswerVerdict:
    """Assess whether the answer is grounded in the retrieved evidence.

    Returns 'grounded' if the answer is supported by evidence.
    Returns 'hallucinated' if the answer contradicts or lacks evidence support.
    Returns 'incomplete' if the answer is partially correct but misses information.
    Falls back to 'grounded' when no LLM is configured (don't block pipeline).
    """
    if chat_client is None or config.llm is None:
        return "grounded"

    evidence_text = _format_evidence(evidence)
    messages = answer_grader_messages(query, answer, evidence_text)
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
        verdict = str(data.get("verdict", "grounded")).lower()
        if verdict not in {"grounded", "hallucinated", "incomplete"}:
            verdict = "grounded"
        return verdict  # type: ignore[return-value]
    except Exception:
        return "grounded"


def _format_evidence(evidence: list[EvidenceChunk], limit: int = 5) -> str:
    lines = []
    for chunk in evidence[:limit]:
        snippet = " ".join(chunk.content.split())[:300]
        lines.append(f"- [{chunk.source_label}] {snippet}")
    return "\n".join(lines) or "(no evidence)"
