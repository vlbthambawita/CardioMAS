from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SplitManifest(BaseModel):
    dataset_name: str
    split_version: str = "auto-v1"
    author: str = "cardiomas"
    description: str = ""
    cardiomas_version: str = ""
    reproducibility_config: "ReproducibilityConfig"
    splits: dict[str, list[str]] = Field(default_factory=dict)
    split_stats: dict[str, dict] = Field(default_factory=dict)


class ReproducibilityConfig(BaseModel):
    cardiomas_version: str
    seed: int
    dataset_name: str
    dataset_source_url: str | None
    dataset_checksum: str
    split_strategy: str
    split_ratios: dict[str, float]
    stratify_by: list[str] | None = None
    group_by: str | None = None
    custom_params: dict | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    algorithm_version: str = "v1"
