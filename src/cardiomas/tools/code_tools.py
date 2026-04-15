"""
Code tools for the Coder Agent — write, execute, and verify Python scripts.

All tools return dicts (never raise). Errors land in the "error" key.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


@tool
def write_script(script_path: str, content: str) -> dict[str, Any]:
    """Write a Python script to disk. Creates parent directories as needed."""
    try:
        path = Path(script_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        sha256 = hashlib.sha256(content.encode()).hexdigest()
        return {"status": "ok", "path": str(path), "sha256": sha256, "size": len(content)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def execute_script(script_path: str, timeout: int = 120, working_dir: str = "") -> dict[str, Any]:
    """Execute a Python script in a subprocess. Returns stdout, stderr, exit_code."""
    try:
        path = Path(script_path)
        if not path.exists():
            return {
                "status": "error", "error": f"Script not found: {script_path}",
                "exit_code": -1, "stdout": "", "stderr": "",
            }
        cwd = working_dir or str(path.parent)
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "status": "ok" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:2000],
            "script": str(path),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout", "exit_code": -1,
            "error": f"Timed out after {timeout}s", "stdout": "", "stderr": "",
        }
    except Exception as e:
        return {"status": "error", "exit_code": -1, "error": str(e), "stdout": "", "stderr": ""}


@tool
def verify_script_output(output_splits_file: str, reference_splits_file: str) -> dict[str, Any]:
    """Compare the SHA-256 of script-generated splits.json against the reference splits.json."""
    try:
        with open(output_splits_file) as f:
            output = json.load(f)
        with open(reference_splits_file) as f:
            reference = json.load(f)

        out_splits = output.get("splits", output)
        ref_splits = reference.get("splits", reference)

        out_hash = _splits_hash(out_splits)
        ref_hash = _splits_hash(ref_splits)

        mismatches = []
        for split_name in ref_splits:
            if split_name not in out_splits:
                mismatches.append(f"Missing split: {split_name}")
            elif sorted(ref_splits[split_name]) != sorted(out_splits[split_name]):
                mismatches.append(
                    f"{split_name}: {len(ref_splits[split_name])} ref vs "
                    f"{len(out_splits[split_name])} output"
                )

        return {
            "status": "ok",
            "match": out_hash == ref_hash,
            "output_hash": out_hash,
            "reference_hash": ref_hash,
            "mismatches": mismatches,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "match": False}


def _splits_hash(splits: dict) -> str:
    """Deterministic SHA-256 of splits (sorted keys, sorted IDs within each split)."""
    canonical = {k: sorted(v) for k, v in sorted(splits.items())}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
