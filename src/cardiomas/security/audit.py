from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cardiomas.schemas.audit import SecurityAudit
from cardiomas.tools.security_tools import scan_for_pii, validate_split_file, check_patient_leakage


def run_security_audit(
    splits: dict[str, list[str]],
    patient_mapping: dict[str, str] | None = None,
) -> SecurityAudit:
    """Run full security audit on proposed splits. Returns SecurityAudit."""
    blocking: list[str] = []
    warnings: list[str] = []
    pii_findings: list[str] = []
    leakage_findings: list[str] = []

    # Serialize splits to check for PII in IDs themselves
    split_json = json.dumps(splits)
    pii_result = scan_for_pii.invoke({"data": split_json})
    if pii_result["has_pii"]:
        for f in pii_result["findings"]:
            pii_findings.append(f"PII ({f['type']}) found in record IDs")
        blocking.append("PII detected in record identifiers")

    # File size check
    raw_data_detected = False
    suspicious_files: list[str] = []
    all_ids = [rid for ids in splits.values() for rid in ids]
    # Check total ID list size as a heuristic
    if len(split_json) > 50 * 1024 * 1024:  # > 50 MB when serialized
        suspicious_files.append("splits JSON exceeds 50 MB")
        raw_data_detected = True
        blocking.append("Split data exceeds 50 MB — likely contains raw data")

    # Patient leakage
    patient_leakage = False
    if patient_mapping:
        leak_result = check_patient_leakage.invoke({"splits": splits, "patient_mapping": patient_mapping})
        if leak_result["leakage_detected"]:
            patient_leakage = True
            for detail in leak_result["leakage_details"]:
                leakage_findings.append(
                    f"Patient overlap between {detail['splits']}: {detail['count']} patients"
                )
            blocking.append("Patient-level data leakage detected between splits")

    passed = len(blocking) == 0

    report_lines = ["# Security Audit Report\n"]
    report_lines.append(f"**Result: {'PASSED' if passed else 'FAILED'}**\n")
    if blocking:
        report_lines.append("## Blocking Issues\n")
        for issue in blocking:
            report_lines.append(f"- {issue}")
    if warnings:
        report_lines.append("\n## Warnings\n")
        for w in warnings:
            report_lines.append(f"- {w}")
    if not blocking and not warnings:
        report_lines.append("No issues found. Splits are safe to publish.")

    return SecurityAudit(
        passed=passed,
        pii_detected=len(pii_findings) > 0,
        raw_data_detected=raw_data_detected,
        patient_leakage_detected=patient_leakage,
        suspicious_file_sizes=suspicious_files,
        pii_findings=pii_findings,
        leakage_findings=leakage_findings,
        warnings=warnings,
        blocking_issues=blocking,
        report="\n".join(report_lines),
    )
