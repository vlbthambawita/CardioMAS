from __future__ import annotations

import math
import re

from cardiomas.schemas.evidence import EvidenceChunk


def retrieve_bm25(chunks: list[EvidenceChunk], query: str, top_k: int) -> list[EvidenceChunk]:
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [_tokens(chunk.content) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokens(query))
        ranked = sorted(
            ((index, score) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        return [chunks[index].model_copy(update={"score": float(score)}) for index, score in ranked[:top_k]]
    except Exception:
        ranked = sorted(
            ((chunk, _fallback_score(chunk.content, query)) for chunk in chunks),
            key=lambda item: item[1],
            reverse=True,
        )
        return [chunk.model_copy(update={"score": score}) for chunk, score in ranked[:top_k]]


def _fallback_score(text: str, query: str) -> float:
    query_terms = _tokens(query)
    if not query_terms:
        return 0.0
    text_terms = _tokens(text)
    overlap = sum(text_terms.count(term) for term in query_terms)
    return float(overlap) / math.sqrt(max(len(text_terms), 1))


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", value.lower())
