"""
DatasetMap schema — the canonical structured description of a dataset directory.

Built by DatasetMapper (dataset_mapper.py) and used by:
- analysis_agent  → LLM context, field validation
- splitter_agent  → real record IDs (replaces synthetic fallback)
- security_agent  → real patient_record_map for leakage detection
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class FileInventory(BaseModel):
    """Structured description of a single dataset file."""

    path: str
    format: Literal["wfdb", "hdf5", "edf", "csv", "numpy", "unknown"]
    size_bytes: int
    record_ids: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    sample_values: dict[str, Any] = Field(default_factory=dict)


class DatasetMap(BaseModel):
    """
    Complete structured description of a dataset directory.

    - all_record_ids: deduplicated, sorted — authoritative source for splitter
    - patient_record_map: patient_id → [record_ids] — used for leakage detection
    - dataset_checksum: SHA-256 of sorted record IDs — used in ReproducibilityConfig
    """

    root_path: str
    total_files: int = 0
    format_distribution: dict[str, int] = Field(default_factory=dict)

    # Record / patient identity
    all_record_ids: list[str] = Field(default_factory=list)
    patient_record_map: dict[str, list[str]] = Field(default_factory=dict)
    id_field: str = "record_id"
    patient_id_field: Optional[str] = None

    # Label information
    label_field: Optional[str] = None
    label_values: list[Any] = Field(default_factory=list)
    label_type: Literal["single", "multi", "none"] = "none"

    # File inventories
    metadata_files: list[str] = Field(default_factory=list)   # CSV/TSV paths
    signal_files: list[str] = Field(default_factory=list)     # WFDB/HDF5/EDF paths
    file_inventory: list[FileInventory] = Field(default_factory=list)

    # Data quality
    missing_data_fraction: float = 0.0
    available_fields: list[str] = Field(default_factory=list)  # all column names

    # Reproducibility
    dataset_checksum: str = ""  # SHA-256(sorted record IDs)
