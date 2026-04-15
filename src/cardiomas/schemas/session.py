from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LLMCall(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent: str
    model: str = ""
    system_prompt: str = ""
    user_message: str = ""
    response: str = ""
    duration_ms: int = 0
    token_counts: dict | None = None
    compressed: bool = False          # True if context was compressed before this call
    original_context_len: int = 0    # chars in context before compression


class AgentStep(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent: str
    action: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    reasoning: str = ""
    skipped: bool = False
    skip_reason: str = ""
    duration_ms: int = 0


class SessionLog(BaseModel):
    session_id: str
    cardiomas_version: str = ""
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    dataset_name: str = ""
    raw_requirement: str | None = None
    user_options: dict = Field(default_factory=dict)
    agent_steps: list[AgentStep] = Field(default_factory=list)
    orchestrator_reasoning: list[str] = Field(default_factory=list)
    final_status: str = "pending"
    errors: list[str] = Field(default_factory=list)
