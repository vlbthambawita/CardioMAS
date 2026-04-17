from __future__ import annotations

import math
import re
from collections import Counter

from cardiomas.inference.base import EmbeddingClient
from cardiomas.schemas.evidence import EvidenceChunk


def retrieve_dense(
    chunks: list[EvidenceChunk],
    query: str,
    top_k: int,
    embedding_client: EmbeddingClient | None = None,
    embedding_model: str = "",
) -> list[EvidenceChunk]:
    if embedding_client is not None and embedding_model and any(chunk.embedding for chunk in chunks):
        query_embeddings = embedding_client.embed(embedding_model, [query])
        query_vector = query_embeddings[0] if query_embeddings else []
        if query_vector:
            ranked = sorted(
                (
                    (
                        chunk,
                        _vector_cosine(query_vector, chunk.embedding) if chunk.embedding else 0.0,
                    )
                    for chunk in chunks
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            return [chunk.model_copy(update={"score": score}) for chunk, score in ranked[:top_k]]

    query_vector = Counter(_tokens(query))
    ranked = sorted(
        ((chunk, _cosine(query_vector, Counter(_tokens(chunk.content)))) for chunk in chunks),
        key=lambda item: item[1],
        reverse=True,
    )
    return [chunk.model_copy(update={"score": score}) for chunk, score in ranked[:top_k]]


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", value.lower())


def _vector_cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
