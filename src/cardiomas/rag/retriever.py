"""
Hybrid retriever — combines BM25 (keyword) and dense (semantic) search
with Reciprocal Rank Fusion (RRF) merging.

Usage:
    evidence = retrieve_evidence(
        paper_text=full_text,
        paper_source=url,
        dataset_name="ptb-xl",
        query="train test split stratification patient level",
        top_k=5,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Keywords used to guide BM25 search toward split-methodology sections
_SPLIT_KEYWORDS = [
    "train", "validation", "test", "split", "partition",
    "stratif", "patient", "ratio", "fold", "cross-validation",
    "exclusion", "criteria", "official", "random", "random seed",
]

_RRF_K = 60  # RRF constant (standard: 60)


def retrieve_evidence(
    paper_text: str,
    paper_source: str,
    dataset_name: str,
    query: str,
    top_k: int = 5,
    output_dir: str = "output",
) -> list[str]:
    """
    Retrieve the top_k most relevant chunks from the paper.

    Strategy:
    1. Build/load the LanceDB index for this paper.
    2. Run BM25 search over all chunks.
    3. Run dense vector search in LanceDB.
    4. Merge with Reciprocal Rank Fusion.
    5. Return top_k chunk texts.

    Falls back to a simple keyword-match extraction if LanceDB is unavailable.
    """
    from cardiomas.rag.paper_indexer import build_index, chunk_text, _index_path, _embed

    index_dir = _index_path(paper_source, dataset_name, output_dir)

    # Build index if not present
    if not (index_dir / "_latest.manifest").exists():
        build_index(paper_text, paper_source, dataset_name, output_dir)

    chunks = chunk_text(paper_text)
    if not chunks:
        return []

    # ── BM25 retrieval ─────────────────────────────────────────────────────
    bm25_results = _bm25_search(chunks, query, top_n=min(20, len(chunks)))

    # ── Dense retrieval (if LanceDB available) ────────────────────────────
    dense_results = _dense_search(index_dir, query, top_n=min(20, len(chunks)))

    # ── RRF merge ─────────────────────────────────────────────────────────
    if dense_results:
        merged = _rrf_merge(bm25_results, dense_results, chunks)
    else:
        # LanceDB not available — fall back to BM25 only
        merged = [chunks[i] for i in bm25_results[:top_k]]

    return merged[:top_k]


def _bm25_search(chunks: list[str], query: str, top_n: int) -> list[int]:
    """Return chunk indices sorted by BM25 score (descending)."""
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [chunk.lower().split() for chunk in chunks]
        tokenized_query = query.lower().split()
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_n]
    except ImportError:
        logger.debug("rank-bm25 not installed — using keyword-match fallback for BM25")
        return _keyword_match_fallback(chunks, top_n)
    except Exception as exc:
        logger.warning(f"BM25 search failed: {exc}")
        return list(range(min(top_n, len(chunks))))


def _keyword_match_fallback(chunks: list[str], top_n: int) -> list[int]:
    """Simple keyword hit count as a BM25 fallback."""
    scores = []
    for i, chunk in enumerate(chunks):
        lower = chunk.lower()
        score = sum(1 for kw in _SPLIT_KEYWORDS if kw in lower)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores[:top_n]]


def _dense_search(index_dir: Path, query: str, top_n: int) -> list[int]:
    """Return chunk indices from LanceDB dense search (ascending rank order)."""
    try:
        import lancedb
        from cardiomas.rag.paper_indexer import _embed

        if not (index_dir / "_latest.manifest").exists():
            return []

        db = lancedb.connect(str(index_dir))
        if "paper_chunks" not in db.table_names():
            return []

        table = db.open_table("paper_chunks")
        query_vec = _embed([query])[0]
        results = (
            table.search(query_vec)
            .limit(top_n)
            .select(["chunk_id"])
            .to_list()
        )
        return [r["chunk_id"] for r in results]
    except ImportError:
        return []
    except Exception as exc:
        logger.debug(f"Dense search failed: {exc}")
        return []


def _rrf_merge(
    bm25_ranks: list[int],
    dense_ranks: list[int],
    chunks: list[str],
) -> list[str]:
    """
    Reciprocal Rank Fusion: score = Σ 1/(k + rank_i) for each retriever.
    Returns deduplicated chunks sorted by descending RRF score.
    """
    scores: dict[int, float] = {}

    for rank, chunk_idx in enumerate(bm25_ranks):
        scores[chunk_idx] = scores.get(chunk_idx, 0.0) + 1.0 / (_RRF_K + rank + 1)

    for rank, chunk_idx in enumerate(dense_ranks):
        scores[chunk_idx] = scores.get(chunk_idx, 0.0) + 1.0 / (_RRF_K + rank + 1)

    ranked = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked if i < len(chunks)]
