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
    dry_run: bool = False


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
    user_options: UserOptions = Field(default_factory=UserOptions.model_construct)
    execution_log: list[LogEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
