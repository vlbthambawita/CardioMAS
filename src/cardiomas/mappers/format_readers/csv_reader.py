"""
CSV/TSV reader for the DatasetMapper.

Responsibilities:
- Detect the ECG record ID column and patient ID column
- Detect the diagnostic label column (single vs. multi-label)
- Build the patient → [record_ids] mapping
- Compute missing-data fraction
- Return all column names
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Heuristic column name patterns (lowercase matching)
_ID_PATTERNS = [
    "ecg_id", "record_id", "recording_id", "study_id",
    "filename", "file_id", "waveform_id", "signal_id",
    "exam_id", "ecg_no", "ecg_num",
]
_PATIENT_PATTERNS = [
    "patient_id", "subject_id", "patientid", "patient",
    "patient_key", "person_id", "pid", "mrn",
]
_LABEL_PATTERNS = [
    "label", "labels", "rhythm", "diagnosis", "diagnoses",
    "scp_codes", "scp_code", "category", "class", "classes",
    "arrhythmia", "annotation", "annotations",
]


def read_csv(
    path: str,
    max_rows: int = 50_000,
) -> dict[str, Any]:
    """
    Read a CSV/TSV file and extract dataset structure.

    Returns a dict with:
        record_ids, patient_record_map, id_field, patient_id_field,
        label_field, label_type, label_values, fields, sample_values,
        missing_data_fraction, total_rows
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available — CSV reader skipped")
        return {"error": "pandas not installed"}

    p = Path(path)
    sep = "\t" if p.suffix == ".tsv" else ","

    try:
        df = pd.read_csv(p, sep=sep, nrows=max_rows, low_memory=False)
    except Exception as exc:
        logger.warning(f"csv_reader: could not read {path}: {exc}")
        return {"error": str(exc)}

    columns = list(df.columns)
    columns_lower = [c.lower() for c in columns]

    # ── Identify ID field ──────────────────────────────────────────────────
    id_field = _match_column(columns, columns_lower, _ID_PATTERNS)
    if id_field is None:
        # Fall back to first column if nothing matches
        id_field = columns[0] if columns else "record_id"
        logger.debug(f"csv_reader: no ID column matched — using first column '{id_field}'")

    # ── Identify patient ID field ──────────────────────────────────────────
    patient_id_field = _match_column(columns, columns_lower, _PATIENT_PATTERNS)

    # ── Identify label field ───────────────────────────────────────────────
    label_field = _match_column(columns, columns_lower, _LABEL_PATTERNS)
    label_type: str = "none"
    label_values: list[Any] = []

    if label_field and label_field in df.columns:
        sample_col = df[label_field].dropna()
        if len(sample_col) > 0:
            first_val = sample_col.iloc[0]
            if isinstance(first_val, (dict, list)) or (
                isinstance(first_val, str) and (
                    first_val.startswith("{") or first_val.startswith("[")
                )
            ):
                label_type = "multi"
            else:
                label_type = "single"
            label_values = sample_col.value_counts().head(20).index.tolist()

    # ── Extract record IDs ─────────────────────────────────────────────────
    if id_field in df.columns:
        raw_ids = df[id_field].dropna().astype(str).str.strip().tolist()
        record_ids = sorted(set(raw_ids))
    else:
        record_ids = [f"record_{i:06d}" for i in range(len(df))]

    # ── Build patient → record mapping ────────────────────────────────────
    patient_record_map: dict[str, list[str]] = {}
    if patient_id_field and patient_id_field in df.columns and id_field in df.columns:
        grp = df[[patient_id_field, id_field]].dropna()
        for pid, rid in zip(
            grp[patient_id_field].astype(str).str.strip(),
            grp[id_field].astype(str).str.strip(),
        ):
            patient_record_map.setdefault(pid, []).append(rid)

    # ── Missing data fraction ──────────────────────────────────────────────
    missing_frac = float(df.isnull().any(axis=1).mean()) if len(df) > 0 else 0.0

    # ── Sample values for context ─────────────────────────────────────────
    sample_values: dict[str, Any] = {}
    for col in columns[:10]:
        try:
            vals = df[col].dropna().head(3).tolist()
            sample_values[col] = vals
        except Exception:
            pass

    return {
        "record_ids": record_ids,
        "patient_record_map": patient_record_map,
        "id_field": id_field,
        "patient_id_field": patient_id_field,
        "label_field": label_field,
        "label_type": label_type,
        "label_values": label_values,
        "fields": columns,
        "sample_values": sample_values,
        "missing_data_fraction": missing_frac,
        "total_rows": len(df),
    }


def _match_column(
    columns: list[str],
    columns_lower: list[str],
    patterns: list[str],
) -> Optional[str]:
    """Return the first column whose lowercase name matches any pattern."""
    for pattern in patterns:
        for orig, low in zip(columns, columns_lower):
            if pattern == low or low.startswith(pattern):
                return orig
    return None
