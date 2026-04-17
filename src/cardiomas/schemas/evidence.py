from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_id: str
    source_label: str
    locator: str
    source_type: str
    score: float = 0.0


class KnowledgeDocument(BaseModel):
    doc_id: str
    source_id: str
    source_label: str
    source_type: str
    uri: str
    title: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class EvidenceChunk(BaseModel):
    chunk_id: str
    source_id: str
    source_label: str
    source_type: str
    title: str
    content: str
    uri: str
    metadata: dict[str, str] = Field(default_factory=dict)
    score: float = 0.0

    def citation(self) -> Citation:
        locator = self.metadata.get("chunk_label") or self.title or self.uri
        return Citation(
            source_id=self.source_id,
            source_label=self.source_label,
            locator=locator,
            source_type=self.source_type,
            score=self.score,
        )
