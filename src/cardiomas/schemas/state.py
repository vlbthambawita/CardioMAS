from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from cardiomas.schemas.dataset import DatasetInfo
from cardiomas.schemas.split import SplitManifest
from cardiomas.schemas.audit import SecurityAudit


class UserOptions(BaseModel):
    dataset_source: str
    local_path: str | None = None
    output_dir: str = "output"
    force_reanalysis: bool = False
    use_cloud_llm: bool = False
    seed: int = 42
    custom_split: dict[str, float] | None = None
    ignore_official: bool = False
    stratify_by: str | None = None
    verbose: bool = False
    push_to_hf: bool = False          # explicit opt-in; requires HF_TOKEN
    requirement: str | None = None    # natural language requirement input (V2)
    agent_llm_map: dict[str, str] = Field(default_factory=dict)  # per-agent LLM overrides (V2)


class LogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent: str
    action: str
    detail: str = ""


class GraphState(BaseModel):
    dataset_source: str
    dataset_info: DatasetInfo | None = None
    download_status: str = "pending"
    paper_findings: dict[str, Any] | None = None
    analysis_report: dict[str, Any] | None = None
    existing_hf_splits: dict | None = None
    proposed_splits: SplitManifest | None = None
    security_audit: SecurityAudit | None = None
    publish_status: str = "pending"
    local_output_dir: str = ""   # set by splitter_agent after saving locally
    user_options: UserOptions = Field(default_factory=UserOptions.model_construct)
    execution_log: list[LogEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # ── V2 orchestrator fields ─────────────────────────────────────────────
    session_id: str = ""
    next_agent: str = ""
    last_completed_agent: str = ""
    agent_skip_reasons: dict[str, str] = Field(default_factory=dict)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    orchestrator_reasoning: list[str] = Field(default_factory=list)

    # ── V2 NL requirement ─────────────────────────────────────────────────
    parsed_requirement: Any | None = None   # ParsedRequirement | None

    # ── V2 coder agent ────────────────────────────────────────────────────
    generated_scripts: dict[str, str] = Field(default_factory=dict)   # {name: path}
    script_execution_log: list[dict] = Field(default_factory=list)
    script_verified: bool = False

    # ── V2 checkpoint / resume ────────────────────────────────────────────
    checkpoint_path: str = ""
    last_completed_agent_at: str = ""  # ISO timestamp of last agent completion
