from __future__ import annotations

import json
from pathlib import Path

from cardiomas.knowledge.chunking import chunk_document
from cardiomas.knowledge.loaders import load_source
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.runtime import CorpusManifest


def build_corpus(config: RuntimeConfig) -> CorpusManifest:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[EvidenceChunk] = []
    document_count = 0
    source_ids: list[str] = []

    for source in config.sources:
        docs = load_source(source)
        document_count += len(docs)
        source_ids.append(source.id)
        for doc in docs:
            chunks.extend(chunk_document(doc, config.retrieval.chunk_size, config.retrieval.chunk_overlap))

    config.corpus_path.write_text(
        "\n".join(json.dumps(chunk.model_dump(mode="json")) for chunk in chunks) + ("\n" if chunks else ""),
        encoding="utf-8",
    )

    manifest = CorpusManifest(
        document_count=document_count,
        chunk_count=len(chunks),
        output_dir=str(output_dir),
        corpus_path=str(config.corpus_path),
        source_ids=source_ids,
    )
    config.manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    return manifest


def load_corpus(config: RuntimeConfig) -> list[EvidenceChunk]:
    if not config.corpus_path.exists():
        return []
    chunks: list[EvidenceChunk] = []
    for line in config.corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(EvidenceChunk.model_validate_json(line))
    return chunks
