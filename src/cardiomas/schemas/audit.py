from __future__ import annotations

from pydantic import BaseModel, Field


class SecurityAudit(BaseModel):
    passed: bool
    pii_detected: bool = False
    raw_data_detected: bool = False
    patient_leakage_detected: bool = False
    suspicious_file_sizes: list[str] = Field(default_factory=list)
    pii_findings: list[str] = Field(default_factory=list)
    leakage_findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    report: str = ""
