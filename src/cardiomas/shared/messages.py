from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ArtifactStatus(str, Enum):
    pending = "pending"
    complete = "complete"
    approved = "approved"
    rejected = "rejected"


class ArtifactRef(BaseModel):
    name: str
    path: str = ""
    artifact_type: str = ""
    summary: str = ""
    status: ArtifactStatus = ArtifactStatus.pending
    provenance: list[str] = Field(default_factory=list)


class TaskMessage(BaseModel):
    sender: str
    recipient: str
    department: str
    purpose: str
    instructions: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    approval_required: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DepartmentReport(BaseModel):
    department: str
    summary: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ApprovalRecord(BaseModel):
    artifact_name: str
    artifact_path: str = ""
    status: ArtifactStatus = ArtifactStatus.pending
    approved_by: str = "organization_head"
    notes: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
