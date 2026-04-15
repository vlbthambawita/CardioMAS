"""
Script verification engine for V4 indirect execution.

Provides per-script verifiers that check stdout and generated files for
expected structural markers. All verifiers return ScriptVerification
(never raise).
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class ScriptVerification(BaseModel):
    """Result of verifying a script's outputs."""
    passed: bool
    issues: list[str] = Field(default_factory=list)
    notes: str = ""


# ── Per-script verifiers ───────────────────────────────────────────────────

def verify_explore_output(stdout: str) -> ScriptVerification:
    """Verify 00_explore_structure.py output.

    Checks stdout contains TOTAL_FILES=N and at least one of:
    ROOT=, FORMATS=, or FILE_EXTENSIONS=
    """
    issues: list[str] = []
    notes_parts: list[str] = []

    if not stdout.strip():
        issues.append("stdout is empty — script produced no output")
        return ScriptVerification(passed=False, issues=issues)

    # Check TOTAL_FILES=N
    total_match = re.search(r"TOTAL_FILES\s*=\s*(\d+)", stdout)
    if not total_match:
        issues.append("Missing TOTAL_FILES=N in stdout")
    else:
        notes_parts.append(f"TOTAL_FILES={total_match.group(1)}")

    # Check at least some structural info
    has_structure = any(
        marker in stdout
        for marker in ("ROOT=", "FORMATS=", "FILE_EXTENSIONS=", "EXTENSIONS=", "FORMAT_")
    )
    if not has_structure:
        issues.append(
            "Missing structural info (ROOT=, FORMATS=, or FILE_EXTENSIONS=) in stdout"
        )

    passed = len(issues) == 0
    return ScriptVerification(
        passed=passed,
        issues=issues,
        notes="; ".join(notes_parts) if notes_parts else "",
    )


def verify_metadata_output(stdout: str) -> ScriptVerification:
    """Verify 01_extract_metadata.py output.

    Checks stdout contains COLUMNS= or COLUMN_NAMES=.
    """
    issues: list[str] = []
    notes_parts: list[str] = []

    if not stdout.strip():
        issues.append("stdout is empty — script produced no output")
        return ScriptVerification(passed=False, issues=issues)

    has_columns = any(
        marker in stdout
        for marker in ("COLUMNS=", "COLUMN_NAMES=", "COLUMNS:[", "columns:")
    )
    if not has_columns:
        issues.append("Missing COLUMNS= or COLUMN_NAMES= in stdout")
    else:
        notes_parts.append("column schema found in stdout")

    passed = len(issues) == 0
    return ScriptVerification(
        passed=passed,
        issues=issues,
        notes="; ".join(notes_parts),
    )


def verify_stats_output(
    stdout: str,
    generated_files: dict[str, str],
) -> ScriptVerification:
    """Verify 02_compute_statistics.py output.

    Checks: stats.csv present in generated_files (or stdout confirms write).
    """
    issues: list[str] = []
    notes_parts: list[str] = []

    # Check for stats.csv in generated output files
    has_stats_csv = "stats.csv" in generated_files
    stdout_mentions_stats = "stats.csv" in stdout or "STATS_CSV=" in stdout

    if not has_stats_csv and not stdout_mentions_stats:
        issues.append("stats.csv not found in generated files and not mentioned in stdout")
    else:
        if has_stats_csv:
            notes_parts.append(f"stats.csv found ({len(generated_files['stats.csv'])} bytes)")
        else:
            notes_parts.append("stats.csv mentioned in stdout")

    # Soft check: LABEL_FIELD or VALUE_COUNTS in stdout
    has_label_info = any(
        m in stdout for m in ("LABEL_FIELD=", "VALUE_COUNTS", "label_field", "LABEL_COUNTS")
    )
    if not has_label_info:
        notes_parts.append("note: no LABEL_FIELD info in stdout (may be OK if no labels)")

    passed = len(issues) == 0
    return ScriptVerification(
        passed=passed,
        issues=issues,
        notes="; ".join(notes_parts),
    )


def verify_subset_splits_output(
    stdout: str,
    generated_files: dict[str, str],
    expected_subset_size: int,
) -> ScriptVerification:
    """Verify 03_generate_splits_subset.py output.

    Checks:
    - splits_subset.json in generated_files and parseable
    - Total IDs across all splits <= expected_subset_size
    """
    issues: list[str] = []
    notes_parts: list[str] = []

    # Check splits_subset.json
    if "splits_subset.json" not in generated_files:
        issues.append("splits_subset.json not found in generated files")
        return ScriptVerification(passed=False, issues=issues)

    # Try to parse it
    try:
        data = json.loads(generated_files["splits_subset.json"])
        splits = data.get("splits", data)
        if not isinstance(splits, dict):
            issues.append("splits_subset.json does not contain a 'splits' dict")
        else:
            total_ids = sum(len(v) for v in splits.values())
            notes_parts.append(
                f"splits_subset.json valid: {list(splits.keys())}, total={total_ids}"
            )
            if total_ids == 0:
                issues.append("splits_subset.json has 0 total record IDs")
            elif total_ids > expected_subset_size * 1.1:
                # Allow 10% tolerance
                issues.append(
                    f"splits_subset.json has {total_ids} IDs > expected {expected_subset_size}"
                )
    except json.JSONDecodeError as e:
        issues.append(f"splits_subset.json is not valid JSON: {e}")

    passed = len(issues) == 0
    return ScriptVerification(
        passed=passed,
        issues=issues,
        notes="; ".join(notes_parts),
    )


def verify_full_splits_output(
    stdout: str,
    generated_files: dict[str, str],
) -> ScriptVerification:
    """Verify 04_generate_splits_full.py output.

    Checks: splits.json in generated_files and parseable.
    """
    issues: list[str] = []
    notes_parts: list[str] = []

    if "splits.json" not in generated_files:
        issues.append("splits.json not found in generated files")
        return ScriptVerification(passed=False, issues=issues)

    try:
        data = json.loads(generated_files["splits.json"])
        splits = data.get("splits", data)
        if not isinstance(splits, dict):
            issues.append("splits.json does not contain a 'splits' dict")
        else:
            total_ids = sum(len(v) for v in splits.values())
            notes_parts.append(
                f"splits.json valid: {list(splits.keys())}, total={total_ids}"
            )
            if total_ids == 0:
                issues.append("splits.json has 0 total record IDs")
    except json.JSONDecodeError as e:
        issues.append(f"splits.json is not valid JSON: {e}")

    passed = len(issues) == 0
    return ScriptVerification(
        passed=passed,
        issues=issues,
        notes="; ".join(notes_parts),
    )


def verify_ecg_stats_output(
    script_name: str,
    stdout: str,
    generated_files: dict[str, str],
) -> ScriptVerification:
    """Generic verifier for ECG statistics scripts (10–14).

    Checks expected output files exist for each script:
      10_class_distribution   -> class_dist.csv
      11_per_lead_statistics  -> lead_stats.csv
      12_signal_quality       -> quality_report.csv
      13_clinical_plausibility -> clinical_flags.csv
      14_publication_table    -> table1.md or table1.tex
    """
    issues: list[str] = []
    notes_parts: list[str] = []

    # Map script prefix to expected output files
    expected: dict[str, list[str]] = {
        "10": ["class_dist.csv"],
        "11": ["lead_stats.csv"],
        "12": ["quality_report.csv"],
        "13": ["clinical_flags.csv"],
        "14": ["table1.md", "table1.tex"],
    }

    # Determine prefix from script name
    prefix = None
    for p in expected:
        if script_name.startswith(p):
            prefix = p
            break

    if prefix is None:
        # Unknown script — basic check: exit 0 + some stdout
        if not stdout.strip():
            issues.append(f"Script '{script_name}' produced no stdout output")
        return ScriptVerification(
            passed=len(issues) == 0,
            issues=issues,
            notes=f"generic check for unknown script '{script_name}'",
        )

    required_files = expected[prefix]
    found = []
    missing = []
    for fname in required_files:
        if fname in generated_files:
            found.append(fname)
        else:
            # Also check if stdout mentions the file (script might write to a subdir)
            if fname in stdout:
                found.append(f"{fname} (mentioned in stdout)")
            else:
                missing.append(fname)

    if missing:
        issues.append(f"Expected output file(s) not found: {missing}")
    if found:
        notes_parts.append(f"found: {found}")

    passed = len(issues) == 0
    return ScriptVerification(
        passed=passed,
        issues=issues,
        notes="; ".join(notes_parts),
    )


# ── Dispatcher ─────────────────────────────────────────────────────────────

def verify_script_output(
    script_name: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    generated_files: dict[str, str],
    subset_size: int = 100,
) -> ScriptVerification:
    """Dispatch to the appropriate verifier based on script_name prefix."""
    # Always fail on non-zero exit code
    if exit_code != 0:
        return ScriptVerification(
            passed=False,
            issues=[f"Script exited with code {exit_code}. stderr: {stderr[:300]}"],
        )

    name = script_name.lower()
    if name.startswith("00"):
        return verify_explore_output(stdout)
    elif name.startswith("01"):
        return verify_metadata_output(stdout)
    elif name.startswith("02"):
        return verify_stats_output(stdout, generated_files)
    elif name.startswith("03"):
        return verify_subset_splits_output(stdout, generated_files, subset_size)
    elif name.startswith("04"):
        return verify_full_splits_output(stdout, generated_files)
    elif any(name.startswith(p) for p in ("10", "11", "12", "13", "14")):
        return verify_ecg_stats_output(script_name, stdout, generated_files)
    else:
        # Generic: non-zero exit is already handled above; pass if stdout non-empty
        if not stdout.strip():
            return ScriptVerification(
                passed=False,
                issues=["Script produced no stdout output"],
            )
        return ScriptVerification(passed=True, notes="generic pass (unknown script prefix)")
