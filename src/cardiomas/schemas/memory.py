from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from cardiomas.schemas.evidence import Citation
from cardiomas.schemas.tools import ToolCallRecord


class MemoryPolicy(BaseModel):
    mode: str = "session"
    max_turns: int = 8


class SessionMemory(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    queries: list[str] = Field(default_factory=list)
    retrieved_citations: list[Citation] = Field(default_factory=list)
    tool_history: list[ToolCallRecord] = Field(default_factory=list)
