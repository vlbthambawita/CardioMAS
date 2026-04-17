from __future__ import annotations

from cardiomas.retrieval.hybrid import retrieve
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.tools import ToolResult


def retrieve_corpus(
    chunks: list[EvidenceChunk],
    query: str,
    config: RuntimeConfig,
    top_k: int | None = None,
) -> ToolResult:
    retrieval_config = config.retrieval.model_copy(update={"top_k": top_k or config.retrieval.top_k})
    evidence = retrieve(chunks, query, retrieval_config)
    return ToolResult(
        tool_name="retrieve_corpus",
        ok=True,
        summary=f"Retrieved {len(evidence)} chunk(s) for query: {query}",
        data={"query": query, "top_k": retrieval_config.top_k},
        evidence=evidence,
    )
