from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cardiomas.schemas.evidence import EvidenceChunk


class ToolSpec(BaseModel):
    name: str
    description: str
    category: str
    read_only: bool = True
    requires_approval: bool = False
    generated: bool = False


class ToolResult(BaseModel):
    tool_name: str
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    error: str = ""


class ToolCallRecord(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    summary: str = ""
    error: str = ""
    repaired: bool = False
