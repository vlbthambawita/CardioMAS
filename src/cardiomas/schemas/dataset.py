from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class DatasetSource(str, Enum):
    PHYSIONET = "physionet"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    URL = "url"
    KAGGLE = "kaggle"


class DatasetInfo(BaseModel):
    name: str
    source_type: DatasetSource
    source_url: str | None = None
    local_path: Path | None = None
    description: str | None = None
    paper_url: str | None = None
    official_splits: dict | None = None
    num_records: int | None = None
    metadata_fields: list[str] = Field(default_factory=list)
    ecg_id_field: str = "record_id"
    sampling_rate: int | None = None
    num_leads: int | None = None
    version: str | None = None
    license: str | None = None
