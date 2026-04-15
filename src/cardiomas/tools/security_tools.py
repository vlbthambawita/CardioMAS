from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# PII patterns
_PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (r"\b\d{3}\.\d{2}\.\d{4}\b", "SSN"),
    (r"\b[A-Z]{1,3}\d{6,10}\b", "possible_MRN"),
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "date_of_birth"),
    (r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", "ISO_date"),
    (r"\b[A-Za-z]+,\s+[A-Za-z]+\b", "possible_name"),
    (r"\b\d{10}\b", "possible_phone"),
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "email"),
]
_MAX_SAFE_SIZE_MB = 10  # Files larger than this in a split output are suspicious


@tool
def scan_for_pii(data: str) -> dict[str, Any]:
    """Scan a text string for PII patterns.
    Returns dict with 'findings' list and 'has_pii' bool."""
    findings = []
    for pattern, label in _PII_PATTERNS:
        matches = re.findall(pattern, data)
        if matches:
            findings.append({"type": label, "examples": matches[:3], "count": len(matches)})
    return {"has_pii": len(findings) > 0, "findings": findings}


@tool
def validate_split_file(path: str) -> dict[str, Any]:
    """Validate that a split file contains only record IDs, no raw data.
    Checks file size, structure, and scans for PII."""
    p = Path(path)
    if not p.exists():
        return {"valid": False, "error": f"File not found: {path}"}

    size_mb = p.stat().st_size / (1024 * 1024)
    issues = []

    if size_mb > _MAX_SAFE_SIZE_MB:
        issues.append(f"File size {size_mb:.1f} MB exceeds {_MAX_SAFE_SIZE_MB} MB — may contain raw data")

    try:
        with open(p) as f:
            content = f.read()
        data = json.loads(content)
    except Exception as e:
        return {"valid": False, "error": f"Cannot parse JSON: {e}"}

    # Check structure
    if "splits" not in data:
        issues.append("Missing 'splits' key")
    else:
        for split_name, ids in data["splits"].items():
            if not isinstance(ids, list):
                issues.append(f"Split '{split_name}' is not a list")
            elif ids and not isinstance(ids[0], str):
                issues.append(f"Split '{split_name}' IDs are not strings")

    # PII scan
    pii_result = scan_for_pii.invoke({"data": content[:50000]})
    if pii_result["has_pii"]:
        issues.extend([f"PII detected: {f['type']}" for f in pii_result["findings"]])

    return {
        "valid": len(issues) == 0,
        "size_mb": round(size_mb, 3),
        "issues": issues,
        "pii_findings": pii_result["findings"],
    }


@tool
def check_patient_leakage(
    splits: dict[str, list[str]],
    patient_mapping: dict[str, str],
) -> dict[str, Any]:
    """Verify that no patient appears in more than one split.
    patient_mapping: {record_id: patient_id}.
    Returns dict with leakage details."""
    from itertools import combinations

    split_patients: dict[str, set[str]] = {}
    for split_name, ids in splits.items():
        split_patients[split_name] = {patient_mapping[i] for i in ids if i in patient_mapping}

    leakage: list[dict[str, Any]] = []
    for a, b in combinations(split_patients.keys(), 2):
        overlap = split_patients[a] & split_patients[b]
        if overlap:
            leakage.append({
                "splits": [a, b],
                "overlapping_patients": list(overlap)[:10],
                "count": len(overlap),
            })

    return {
        "leakage_detected": len(leakage) > 0,
        "leakage_details": leakage,
        "patients_per_split": {k: len(v) for k, v in split_patients.items()},
    }
