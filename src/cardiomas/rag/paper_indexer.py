"""
Paper indexer — chunks a PDF text and stores embeddings in LanceDB.

Architecture:
    PDF text
        → paragraph-aware chunking (≤400 tokens, 50-token overlap)
        → embeddings via nomic-embed-text (Ollama) or sentence-transformers fallback
        → stored in LanceDB at output/<dataset>/rag/<sanitized_url>/

One index is created per paper URL. Subsequent calls reuse the existing index.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Approximate chars per token for chunking (conservative for English text)
_CHARS_PER_TOKEN = 4
_CHUNK_TOKENS = 400
_OVERLAP_TOKENS = 50
_CHUNK_CHARS = _CHUNK_TOKENS * _CHARS_PER_TOKEN      # 1600 chars
_OVERLAP_CHARS = _OVERLAP_TOKENS * _CHARS_PER_TOKEN  # 200 chars


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping paragraph-aware chunks."""
    # Split on paragraph boundaries first
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= _CHUNK_CHARS:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                # Overlap: keep last _OVERLAP_CHARS of previous chunk
                current = current[-_OVERLAP_CHARS:].strip() + "\n\n" + para
                current = current.strip()
            else:
                # Para itself is too long — hard split
                for i in range(0, len(para), _CHUNK_CHARS - _OVERLAP_CHARS):
                    chunks.append(para[i: i + _CHUNK_CHARS])
                current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) > 50]  # drop trivially short chunks


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Tries Ollama nomic-embed-text first, then sentence-transformers."""
    # ── Try Ollama embeddings ──────────────────────────────────────────────
    try:
        from langchain_ollama import OllamaEmbeddings
        import cardiomas.config as cfg
        embedder = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=cfg.OLLAMA_BASE_URL,
        )
        return embedder.embed_documents(texts)
    except Exception as ollama_exc:
        logger.debug(f"Ollama embeddings failed: {ollama_exc} — trying sentence-transformers")

    # ── Fallback: sentence-transformers ───────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vecs]
    except Exception as st_exc:
        logger.warning(f"sentence-transformers also failed: {st_exc}")
        raise RuntimeError(
            "No embedding backend available. "
            "Install 'sentence-transformers' or ensure 'nomic-embed-text' is pulled in Ollama."
        ) from st_exc


def _index_path(paper_source: str, dataset_name: str, output_base: str) -> Path:
    """Deterministic path for this paper's LanceDB index."""
    url_hash = hashlib.md5(paper_source.encode()).hexdigest()[:12]
    return Path(output_base) / dataset_name / "rag" / url_hash


def build_index(
    paper_text: str,
    paper_source: str,
    dataset_name: str,
    output_dir: str = "output",
) -> Optional[Path]:
    """
    Chunk and embed paper_text, store in LanceDB.

    Returns the LanceDB directory path, or None if lancedb is not available.
    """
    try:
        import lancedb
    except ImportError:
        logger.debug("lancedb not installed — RAG indexing skipped")
        return None

    index_dir = _index_path(paper_source, dataset_name, output_dir)

    # Skip re-indexing if already done
    if (index_dir / "_latest.manifest").exists():
        logger.debug(f"RAG index already exists at {index_dir}")
        return index_dir

    chunks = chunk_text(paper_text)
    if not chunks:
        logger.warning("paper_indexer: no chunks produced — paper text may be empty")
        return None

    logger.info(f"paper_indexer: embedding {len(chunks)} chunks from {paper_source}")

    try:
        embeddings = _embed(chunks)
    except RuntimeError as exc:
        logger.warning(f"paper_indexer: embedding failed — {exc}")
        return None

    # Store in LanceDB
    index_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(index_dir))

    data = [
        {"chunk_id": i, "text": chunk, "vector": vec, "source": paper_source}
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings))
    ]

    if "paper_chunks" in db.table_names():
        db.drop_table("paper_chunks")
    db.create_table("paper_chunks", data=data)

    # Write sentinel so we know the index is complete
    (index_dir / "_latest.manifest").write_text(
        f"chunks={len(chunks)}\nsource={paper_source}\n"
    )
    logger.info(f"paper_indexer: indexed {len(chunks)} chunks → {index_dir}")
    return index_dir
