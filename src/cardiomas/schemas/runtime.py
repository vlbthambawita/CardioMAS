from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from cardiomas.schemas.evidence import Citation, EvidenceChunk
from cardiomas.schemas.tools import ToolCallRecord


class PlanStep(BaseModel):
    tool_name: str
    reason: str
    args: dict = Field(default_factory=dict)


class AgentDecision(BaseModel):
    strategy: str
    steps: list[PlanStep] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LLMTrace(BaseModel):
    stage: str
    provider: str = ""
    model: str = ""
    ok: bool = True
    prompt_preview: str = ""
    response_preview: str = ""
    error: str = ""


class AgentEvent(BaseModel):
    type: str
    stage: str = ""
    message: str = ""
    content: str = ""
    data: dict = Field(default_factory=dict)


class RepairTrace(BaseModel):
    tool_name: str
    action: str
    attempt: int = 0
    ok: bool = True
    workspace_path: str = ""
    files_written: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    retry_succeeded: bool = False
    error: str = ""


class CorpusManifest(BaseModel):
    built_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    document_count: int = 0
    chunk_count: int = 0
    embedded_chunk_count: int = 0
    embedding_model: str = ""
    output_dir: str
    corpus_path: str
    source_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    session_id: str
    query: str
    answer: str
    decision: AgentDecision
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_traces: list[LLMTrace] = Field(default_factory=list)
    repair_traces: list[RepairTrace] = Field(default_factory=list)
    standalone_scripts: list[dict] = Field(default_factory=list)
