from __future__ import annotations

from cardiomas.schemas.evidence import EvidenceChunk, KnowledgeDocument


def chunk_document(document: KnowledgeDocument, chunk_size: int, overlap: int) -> list[EvidenceChunk]:
    text = document.content.strip()
    if not text:
        return []

    paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[EvidenceChunk] = []
    buffer = ""
    chunk_index = 0

    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(_make_chunk(document, buffer, chunk_index))
            chunk_index += 1
            buffer = _tail(buffer, overlap)
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        while len(candidate) > chunk_size:
            current = candidate[:chunk_size].rsplit(" ", 1)[0] or candidate[:chunk_size]
            chunks.append(_make_chunk(document, current, chunk_index))
            chunk_index += 1
            candidate = candidate[max(0, len(current) - overlap):].lstrip()
        buffer = candidate

    if buffer:
        chunks.append(_make_chunk(document, buffer, chunk_index))
    return chunks


def _make_chunk(document: KnowledgeDocument, text: str, chunk_index: int) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=f"{document.doc_id}:chunk-{chunk_index}",
        source_id=document.source_id,
        source_label=document.source_label,
        source_type=document.source_type,
        title=document.title,
        content=text.strip(),
        uri=document.uri,
        metadata={**document.metadata, "chunk_index": str(chunk_index), "chunk_label": f"{document.title}#chunk-{chunk_index}"},
    )


def _tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return text
    return text[-overlap:]
