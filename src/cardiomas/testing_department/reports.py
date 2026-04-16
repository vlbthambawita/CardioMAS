from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ToolTestReport(BaseModel):
    tool_name: str
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    details: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
