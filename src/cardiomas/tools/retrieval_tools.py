from __future__ import annotations

from cardiomas.inference.base import EmbeddingClient
from cardiomas.retrieval.hybrid import retrieve
from cardiomas.schemas.config import RetrievalConfig, RuntimeConfig
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.tools import ToolResult


def retrieve_corpus(
    chunks: list[EvidenceChunk],
    query: str,
    config: RuntimeConfig,
    embedding_client: EmbeddingClient | None = None,
    top_k: int | None = None,
    agent_chunks: list[EvidenceChunk] | None = None,
    agent_retrieval: RetrievalConfig | None = None,
) -> ToolResult:
    effective_top_k = top_k or config.retrieval.top_k
    retrieval_cfg = config.retrieval.model_copy(update={"top_k": effective_top_k})
    embed_model = config.embeddings.model if config.embeddings else ""

    global_hits = retrieve(chunks, query, retrieval_cfg, embedding_client=embedding_client, embedding_model=embed_model)

    if agent_chunks:
        agent_cfg = (agent_retrieval or config.retrieval).model_copy(update={"top_k": effective_top_k})
        agent_hits = retrieve(agent_chunks, query, agent_cfg, embedding_client=embedding_client, embedding_model=embed_model)
        evidence = _rrf_merge(global_hits, agent_hits, config.retrieval.rrf_k, effective_top_k)
    else:
        evidence = global_hits

    return ToolResult(
        tool_name="retrieve_corpus",
        ok=True,
        summary=f"Retrieved {len(evidence)} chunk(s) for query: {query}",
        data={"query": query, "top_k": effective_top_k, "mode": retrieval_cfg.mode},
        evidence=evidence,
    )


def _rrf_merge(
    global_hits: list[EvidenceChunk],
    agent_hits: list[EvidenceChunk],
    rrf_k: int,
    top_k: int,
) -> list[EvidenceChunk]:
    """Merge global and agent corpus results with RRF; agent hits get a 1.2× boost."""
    fused: dict[str, float] = {}
    by_id: dict[str, EvidenceChunk] = {}
    for rank, hit in enumerate(global_hits):
        fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        by_id[hit.chunk_id] = hit
    for rank, hit in enumerate(agent_hits):
        fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.2 / (rrf_k + rank + 1)
        by_id[hit.chunk_id] = hit
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [by_id[cid].model_copy(update={"score": score}) for cid, score in ranked[:top_k]]
