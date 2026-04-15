"""
Per-agent structured output DTOs (Phase 1 — Structured Output Contracts).

Each agent's LLM call produces one of these validated Pydantic models
instead of free-form text, eliminating ad-hoc JSON parsing.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from cardiomas.schemas.dataset import DatasetSource


class DiscoveryOutput(BaseModel):
    """Structured output from the discovery agent."""

    dataset_name: str = Field(
        description="Short slug identifier, e.g. 'ptb-xl' or 'mimic-iv-ecg'"
    )
    source_type: str = Field(
        description="One of: physionet, huggingface, local, url, kaggle"
    )
    ecg_id_field: str = Field(
        default="record_id",
        description="Column/field name that uniquely identifies each ECG record",
    )
    patient_id_field: Optional[str] = Field(
        default=None,
        description="Column name for patient identifier, if different from record ID",
    )
    num_records: Optional[int] = Field(
        default=None, description="Total number of ECG records if stated"
    )
    official_splits: bool = Field(
        default=False,
        description="True if the dataset comes with predefined train/val/test splits",
    )
    paper_url: Optional[str] = Field(
        default=None, description="URL of the associated dataset paper if found"
    )
    sampling_rate_hz: Optional[int] = Field(
        default=None, description="ECG sampling rate in Hz if mentioned"
    )
    num_leads: Optional[int] = Field(
        default=None, description="Number of ECG leads if mentioned"
    )
    notes: str = Field(
        default="", description="Anything uncertain or worth flagging"
    )

    @field_validator("source_type")
    @classmethod
    def normalise_source_type(cls, v: str) -> str:
        v = v.lower().strip()
        valid = {s.value for s in DatasetSource}
        return v if v in valid else "url"

    @field_validator("dataset_name")
    @classmethod
    def slugify_name(cls, v: str) -> str:
        return v.lower().strip().split()[0][:50]


class PaperOutput(BaseModel):
    """Structured output from the paper analysis agent."""

    found: bool = Field(description="True if a relevant paper was found and parsed")
    split_methodology: Optional[str] = Field(
        default=None,
        description="Description of how the official splits are defined",
    )
    patient_level: bool = Field(
        default=True,
        description="True if splitting is done at patient level to prevent leakage",
    )
    stratify_by: Optional[str] = Field(
        default=None,
        description="Field name used for stratification (e.g. 'diagnosis', 'scp_codes')",
    )
    official_ratios: Optional[dict[str, float]] = Field(
        default=None,
        description="Official train/val/test ratios if explicitly stated, e.g. {'train':0.7,'val':0.1,'test':0.2}",
    )
    exclusion_criteria: list[str] = Field(
        default_factory=list,
        description="Any stated data exclusion criteria",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Exact quoted sentences from the paper supporting each claim",
    )
    paper_source_url: Optional[str] = Field(
        default=None, description="URL of the paper that was analyzed"
    )
    notes: str = Field(
        default="", description="Anything not covered above or uncertain"
    )

    @field_validator("official_ratios")
    @classmethod
    def ratios_must_sum_to_one(cls, v: Optional[dict]) -> Optional[dict]:
        if v is None:
            return v
        total = sum(v.values())
        if not (0.98 <= total <= 1.02):
            # Normalise instead of rejecting
            return {k: val / total for k, val in v.items()}
        return v


class AnalysisOutput(BaseModel):
    """Structured output from the analysis agent."""

    num_records: int = Field(
        description="Total number of ECG records found in the dataset"
    )
    id_field: str = Field(
        description="Exact column/field name that uniquely identifies each ECG record"
    )
    patient_id_field: Optional[str] = Field(
        default=None,
        description="Column name for grouping records by patient (prevents leakage)",
    )
    label_field: Optional[str] = Field(
        default=None,
        description="Column name for diagnostic labels / arrhythmia class",
    )
    label_type: Literal["single", "multi", "none"] = Field(
        default="none",
        description=(
            "single = one label per record, "
            "multi = list/dict of labels per record, "
            "none = no label column found"
        ),
    )
    label_values: list[Any] = Field(
        default_factory=list,
        description="Up to 20 most frequent unique label values",
    )
    recommended_strategy: str = Field(
        description=(
            "One of: patient_stratified, record_stratified, patient_random, record_random, official"
        )
    )
    missing_data_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of records with any missing field (0.0–1.0)",
    )
    available_fields: list[str] = Field(
        default_factory=list,
        description="All column/field names found in metadata files",
    )
    notes: str = Field(
        default="", description="Any caveats, warnings, or observations"
    )

    @field_validator("recommended_strategy")
    @classmethod
    def normalise_strategy(cls, v: str) -> str:
        v = v.lower().strip()
        valid = {
            "patient_stratified", "record_stratified",
            "patient_random", "record_random", "official",
        }
        if v not in valid:
            return "patient_random"
        return v


class NLRequirementOutput(BaseModel):
    """Structured output from the NL requirement parser agent."""

    split_ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15},
        description="Desired split ratios; values must sum to 1.0",
    )
    stratify_by: Optional[str] = Field(
        default=None,
        description="Field name to stratify by, or null",
    )
    exclusion_filters: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of filter dicts: {field, op, value} where op in [eq, ne, gt, lt, notna]",
    )
    patient_level: bool = Field(
        default=True,
        description="True = group records by patient before splitting",
    )
    seed: Optional[int] = Field(
        default=None, description="Explicit random seed, or null to use global default"
    )
    notes: str = Field(
        default="",
        description="Anything that could not be parsed precisely; preserved for human review",
    )
    llm_reasoning: str = Field(
        default="", description="Brief explanation of parsing decisions"
    )

    @field_validator("split_ratios")
    @classmethod
    def ratios_must_sum_to_one(cls, v: dict) -> dict:
        total = sum(v.values())
        if not (0.98 <= total <= 1.02):
            return {k: val / total for k, val in v.items()}
        return v
