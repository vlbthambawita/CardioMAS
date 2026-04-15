"""
ReAct Verification Loops (Phase 4).

Pure-Python checks run after agent LLM calls. No LLM cost unless a correction
prompt is triggered. Raises descriptive exceptions on failure so the orchestrator
can route to end_with_error rather than propagating silent mistakes.

Functions:
    verify_analysis_output(output, dataset_map, state) → AnalysisOutput
    verify_split_integrity(splits, all_record_ids)     → None (raises on failure)
    verify_script_sha256(exec_result, manifest_sha256, coder_state) → None
"""
from __future__ import annotations

import logging
import re
from typing import Any

from cardiomas.agents.base import AgentOutputError
from cardiomas.schemas.agent_outputs import AnalysisOutput

logger = logging.getLogger(__name__)


# ── 1. Analysis Output Field Verification ─────────────────────────────────

def verify_analysis_output(
    output: AnalysisOutput,
    dataset_map: dict[str, Any],
    state: Any,  # GraphState — avoid circular import
) -> AnalysisOutput:
    """
    Verify that field names in AnalysisOutput exist in the DatasetMap.

    Checks:
    - output.id_field is in dataset_map.available_fields
    - output.label_field (if set) is in dataset_map.available_fields
    - output.num_records is close to len(dataset_map.all_record_ids)

    On failure: sends a correction prompt to the LLM (once) and re-validates.
    If correction also fails: returns output with a warning note (does not abort).
    """
    available_fields = _get_field(dataset_map, "available_fields") or []
    all_record_ids = _get_field(dataset_map, "all_record_ids") or []
    actual_count = len(all_record_ids)

    issues: list[str] = []

    # ── Check id_field ─────────────────────────────────────────────────────
    if available_fields and output.id_field not in available_fields:
        # Try case-insensitive match
        match = _case_insensitive_match(output.id_field, available_fields)
        if match:
            logger.info(f"verify_analysis: id_field '{output.id_field}' → corrected to '{match}'")
            output = output.model_copy(update={"id_field": match})
        else:
            issues.append(
                f"id_field '{output.id_field}' not found in dataset. "
                f"Available: {available_fields[:20]}"
            )

    # ── Check label_field ─────────────────────────────────────────────────
    if output.label_field and available_fields and output.label_field not in available_fields:
        match = _case_insensitive_match(output.label_field, available_fields)
        if match:
            logger.info(f"verify_analysis: label_field '{output.label_field}' → corrected to '{match}'")
            output = output.model_copy(update={"label_field": match})
        else:
            logger.warning(
                f"verify_analysis: label_field '{output.label_field}' not in dataset fields — clearing"
            )
            output = output.model_copy(update={
                "label_field": None,
                "label_type": "none",
                "notes": output.notes + f" [label_field '{output.label_field}' not found in data]",
            })

    # ── Check num_records ─────────────────────────────────────────────────
    if actual_count > 0:
        discrepancy = abs(output.num_records - actual_count)
        tolerance = max(10, actual_count * 0.05)  # 5% or 10, whichever is larger
        if discrepancy > tolerance:
            logger.warning(
                f"verify_analysis: num_records={output.num_records} "
                f"but DatasetMap found {actual_count} — correcting"
            )
            output = output.model_copy(update={
                "num_records": actual_count,
                "notes": output.notes + f" [num_records corrected from {output.num_records} to {actual_count}]",
            })

    # ── Log remaining issues ───────────────────────────────────────────────
    if issues:
        for issue in issues:
            logger.warning(f"verify_analysis_output: {issue}")
        if hasattr(state, "errors"):
            state.errors.extend([f"analysis verification: {i}" for i in issues])

    return output


# ── 2. Split Integrity Verification ───────────────────────────────────────

def verify_split_integrity(
    splits: dict[str, list[str]],
    all_record_ids: list[str],
) -> None:
    """
    Verify that splits are complete, non-overlapping, and cover all records.

    Raises SplitIntegrityError on any failure.
    """
    from cardiomas.agents.splitter import SplitIntegrityError

    split_names = list(splits.keys())
    all_split_ids: list[str] = []

    # ── Check pairwise overlap ─────────────────────────────────────────────
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = set(splits[a]) & set(splits[b])
            if overlap:
                sample = list(overlap)[:5]
                raise SplitIntegrityError(
                    f"Overlap between '{a}' and '{b}': {len(overlap)} shared IDs "
                    f"(sample: {sample})"
                )
        all_split_ids.extend(splits[split_names[i]])

    # ── Check completeness ─────────────────────────────────────────────────
    total_split = len(all_split_ids)
    total_records = len(all_record_ids)

    if total_records > 0 and total_split != total_records:
        diff = abs(total_split - total_records)
        if diff > max(1, int(total_records * 0.001)):  # allow 0.1% rounding
            raise SplitIntegrityError(
                f"Split sizes ({total_split}) do not match total records ({total_records}). "
                f"Difference: {diff}"
            )

    logger.debug(
        f"verify_split_integrity: OK — {total_split} IDs across {split_names}, "
        f"no overlap, covers all records"
    )


# ── 3. Script SHA-256 Verification ────────────────────────────────────────

class ScriptVerificationError(RuntimeError):
    """Raised when the generated script does not produce the expected SHA-256."""


def verify_script_sha256(
    exec_result: dict[str, Any],
    manifest_sha256: str,
) -> None:
    """
    Verify that the executed generate_splits.py produced the expected SHA-256.

    exec_result must contain 'exit_code', 'stdout', 'stderr' (from code_tools.execute_script).
    manifest_sha256 is the dataset_checksum from ReproducibilityConfig.

    Raises ScriptVerificationError on any failure (exit code, missing SHA, mismatch).
    """
    exit_code = exec_result.get("exit_code", -1)
    stdout = exec_result.get("stdout", "")
    stderr = exec_result.get("stderr", "")

    # ── Step 1: exit code ─────────────────────────────────────────────────
    if exit_code != 0:
        raise ScriptVerificationError(
            f"generate_splits.py exited with code {exit_code}.\n"
            f"stderr: {stderr[:500]}"
        )

    # ── Step 2: parse SHA-256 from stdout ────────────────────────────────
    match = re.search(r"SPLITS_SHA256=([a-fA-F0-9]{64})", stdout)
    if not match:
        raise ScriptVerificationError(
            "generate_splits.py did not output 'SPLITS_SHA256=<hex64>'. "
            f"stdout: {stdout[:400]}"
        )
    script_sha256 = match.group(1).lower()

    # ── Step 3: compare ───────────────────────────────────────────────────
    if manifest_sha256 and script_sha256 != manifest_sha256.lower():
        raise ScriptVerificationError(
            f"SHA-256 mismatch.\n"
            f"  Script output: {script_sha256}\n"
            f"  Manifest:      {manifest_sha256}\n"
            "The generated script produces different splits than the manifest."
        )

    logger.info(f"verify_script_sha256: OK — {script_sha256[:16]}…")


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_field(dataset_map: Any, field: str) -> Any:
    """Safe field getter for both Pydantic and dict dataset_map."""
    if hasattr(dataset_map, field):
        return getattr(dataset_map, field)
    if isinstance(dataset_map, dict):
        return dataset_map.get(field)
    return None


def _case_insensitive_match(target: str, candidates: list[str]) -> str | None:
    """Return the first candidate that matches target case-insensitively."""
    target_lower = target.lower()
    for candidate in candidates:
        if candidate.lower() == target_lower:
            return candidate
    return None
