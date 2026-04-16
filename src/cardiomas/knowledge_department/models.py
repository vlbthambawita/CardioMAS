from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class SourceProvenance(BaseModel):
    url: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    fetch_method: str = "unknown"
    status: str = "ok"
    robots_allowed: bool = True
    rate_limit_seconds: float = 0.0
    error: str = ""


class ExtractedTable(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class PageKnowledge(BaseModel):
    url: str
    title: str = ""
    description: str = ""
    headings: list[str] = Field(default_factory=list)
    content_blocks: list[str] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    text_excerpt: str = ""
    provenance: SourceProvenance


class DatasetKnowledgeBundle(BaseModel):
    dataset_name: str
    slug: str
    pages: list[PageKnowledge] = Field(default_factory=list)
    overview: dict[str, Any] = Field(default_factory=dict)
    references: list[dict[str, Any]] = Field(default_factory=list)
    schema_definition: dict[str, Any] = Field(default_factory=dict)
    notes_markdown: str = ""
