from __future__ import annotations

from cardiomas.inference.base import EmbeddingClient
from cardiomas.retrieval.bm25 import retrieve_bm25
from cardiomas.retrieval.dense import retrieve_dense
from cardiomas.schemas.config import RetrievalConfig
from cardiomas.schemas.evidence import EvidenceChunk


def retrieve(
    chunks: list[EvidenceChunk],
    query: str,
    config: RetrievalConfig,
    embedding_client: EmbeddingClient | None = None,
    embedding_model: str = "",
) -> list[EvidenceChunk]:
    if config.mode == "bm25":
        return _filter_min_score(retrieve_bm25(chunks, query, config.top_k), config.min_score)
    if config.mode == "dense":
        return _filter_min_score(
            retrieve_dense(
                chunks,
                query,
                config.top_k,
                embedding_client=embedding_client,
                embedding_model=embedding_model,
            ),
            config.min_score,
        )

    bm25_hits = retrieve_bm25(chunks, query, max(config.top_k * 2, config.top_k))
    dense_hits = retrieve_dense(
        chunks,
        query,
        max(config.top_k * 2, config.top_k),
        embedding_client=embedding_client,
        embedding_model=embedding_model,
    )
    fused: dict[str, float] = {}
    by_id: dict[str, EvidenceChunk] = {}

    for rank, hit in enumerate(bm25_hits):
        fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (config.rrf_k + rank + 1)
        by_id[hit.chunk_id] = hit
    for rank, hit in enumerate(dense_hits):
        fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (config.rrf_k + rank + 1)
        by_id[hit.chunk_id] = hit

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    merged = [by_id[chunk_id].model_copy(update={"score": score}) for chunk_id, score in ranked[: config.top_k]]
    return _filter_min_score(merged, config.min_score)


def _filter_min_score(chunks: list[EvidenceChunk], min_score: float) -> list[EvidenceChunk]:
    return [chunk for chunk in chunks if chunk.score >= min_score]
