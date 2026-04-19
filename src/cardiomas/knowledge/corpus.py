from __future__ import annotations

import json
from pathlib import Path

from cardiomas.inference.base import EmbeddingClient
from cardiomas.knowledge.chunking import chunk_document
from cardiomas.knowledge.loaders import load_source
from cardiomas.schemas.config import NamedAgentConfig, RuntimeConfig
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.runtime import CorpusManifest


def build_corpus(
    config: RuntimeConfig,
    embedding_client: EmbeddingClient | None = None,
) -> CorpusManifest:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[EvidenceChunk] = []
    document_count = 0
    source_ids: list[str] = []
    warnings: list[str] = []

    for source in config.sources:
        docs = load_source(source)
        document_count += len(docs)
        source_ids.append(source.id)
        for doc in docs:
            chunks.extend(chunk_document(doc, config.retrieval.chunk_size, config.retrieval.chunk_overlap))

    if config.embeddings is not None:
        if embedding_client is None:
            warnings.append("Embeddings configured, but no embedding client was available; using lexical-only corpus.")
        else:
            try:
                _attach_embeddings(chunks, config, embedding_client)
            except Exception as exc:
                warnings.append(f"Embeddings unavailable during corpus build; using lexical-only corpus. {exc}")

    config.corpus_path.write_text(
        "\n".join(json.dumps(chunk.model_dump(mode="json")) for chunk in chunks) + ("\n" if chunks else ""),
        encoding="utf-8",
    )

    manifest = CorpusManifest(
        document_count=document_count,
        chunk_count=len(chunks),
        embedded_chunk_count=sum(1 for chunk in chunks if chunk.embedding),
        embedding_model=config.embeddings.model if config.embeddings and any(chunk.embedding for chunk in chunks) else "",
        output_dir=str(output_dir),
        corpus_path=str(config.corpus_path),
        source_ids=source_ids,
        warnings=warnings,
    )
    config.manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    return manifest


def build_agent_corpus(
    agent: NamedAgentConfig,
    config: RuntimeConfig,
    embedding_client: EmbeddingClient | None = None,
    force: bool = False,
) -> CorpusManifest:
    """Build and persist a private corpus for one named agent."""
    corpus_path = config.agent_corpus_path(agent.name)
    manifest_path = config.agent_manifest_path(agent.name)

    if not force and corpus_path.exists() and manifest_path.exists():
        return CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    retrieval_cfg = agent.knowledge.retrieval or config.retrieval
    chunks: list[EvidenceChunk] = []
    document_count = 0
    source_ids: list[str] = []
    warnings: list[str] = []

    for source in agent.knowledge.sources:
        docs = load_source(source)
        document_count += len(docs)
        source_ids.append(source.id)
        for doc in docs:
            chunks.extend(chunk_document(doc, retrieval_cfg.chunk_size, retrieval_cfg.chunk_overlap))

    if config.embeddings is not None:
        if embedding_client is None:
            warnings.append("Embeddings configured but no embedding client; using lexical-only corpus.")
        else:
            try:
                _attach_embeddings(chunks, config, embedding_client)
            except Exception as exc:
                warnings.append(f"Embeddings unavailable; using lexical-only corpus. {exc}")

    corpus_path.write_text(
        "\n".join(json.dumps(chunk.model_dump(mode="json")) for chunk in chunks) + ("\n" if chunks else ""),
        encoding="utf-8",
    )
    manifest = CorpusManifest(
        document_count=document_count,
        chunk_count=len(chunks),
        embedded_chunk_count=sum(1 for c in chunks if c.embedding),
        embedding_model=config.embeddings.model if config.embeddings and any(c.embedding for c in chunks) else "",
        output_dir=str(corpus_path.parent),
        corpus_path=str(corpus_path),
        source_ids=source_ids,
        warnings=warnings,
    )
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    return manifest


def load_agent_corpus(agent_name: str, config: RuntimeConfig) -> list[EvidenceChunk]:
    """Load a pre-built agent corpus from disk (empty list if not built yet)."""
    path = config.agent_corpus_path(agent_name)
    if not path.exists():
        return []
    chunks: list[EvidenceChunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(EvidenceChunk.model_validate_json(line))
    return chunks


def load_corpus(config: RuntimeConfig) -> list[EvidenceChunk]:
    if not config.corpus_path.exists():
        return []
    chunks: list[EvidenceChunk] = []
    for line in config.corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(EvidenceChunk.model_validate_json(line))
    return chunks


def _attach_embeddings(
    chunks: list[EvidenceChunk],
    config: RuntimeConfig,
    embedding_client: EmbeddingClient,
) -> None:
    assert config.embeddings is not None
    batch_size = max(1, config.embeddings.batch_size)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embedding_client.embed(config.embeddings.model, [chunk.content for chunk in batch])
        if len(embeddings) != len(batch):
            raise ValueError("Embedding response length did not match the chunk batch size.")
        for chunk, vector in zip(batch, embeddings, strict=True):
            chunk.embedding = vector
            chunk.embedding_model = config.embeddings.model
