from __future__ import annotations

import json

from cardiomas.inference.base import ChatClient, ChatRequest
from cardiomas.inference.prompts import retrieval_grader_messages
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.runtime import GradedEvidence


def grade_chunks(
    query: str,
    chunks: list[EvidenceChunk],
    config: RuntimeConfig,
    chat_client: ChatClient | None = None,
) -> GradedEvidence:
    """Grade retrieved chunks for relevance to the query.

    Returns 'sufficient' when evidence clearly answers the query.
    Returns 'partial' when evidence is partially useful.
    Returns 'insufficient' when evidence is irrelevant — the orchestrator should re-try.
    Falls back to 'partial' when no LLM is configured.
    """
    if not chunks:
        return GradedEvidence(verdict="insufficient", relevant_count=0, reason="No chunks retrieved.")

    if chat_client is None or config.llm is None:
        return GradedEvidence(
            verdict="partial",
            relevant_count=len(chunks),
            reason="No LLM configured for grading; accepting retrieved chunks.",
        )

    chunks_text = _format_chunks(chunks)
    messages = retrieval_grader_messages(query, chunks_text)
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
        verdict = str(data.get("verdict", "partial")).lower()
        if verdict not in {"sufficient", "partial", "insufficient"}:
            verdict = "partial"
        return GradedEvidence(
            verdict=verdict,  # type: ignore[arg-type]
            relevant_count=int(data.get("relevant_count", len(chunks))),
            reason=str(data.get("reason", "")),
        )
    except Exception as exc:
        return GradedEvidence(
            verdict="partial",
            relevant_count=len(chunks),
            reason=f"Grader failed ({exc}); accepting chunks.",
        )


def _format_chunks(chunks: list[EvidenceChunk], limit: int = 5) -> str:
    lines = []
    for i, chunk in enumerate(chunks[:limit], 1):
        snippet = " ".join(chunk.content.split())[:200]
        lines.append(f"  [{i}] (score={chunk.score:.2f}) {snippet}")
    if len(chunks) > limit:
        lines.append(f"  ... and {len(chunks) - limit} more chunks.")
    return "\n".join(lines)
