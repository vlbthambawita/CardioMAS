"""
V4 output directory tools — manage script output dirs, read generated files,
write execution logs.

All tools return dicts (never raise). Errors land in the "error" key.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# Extensions that are raw ECG signal files — we refuse to read these
_SIGNAL_EXTENSIONS = {".dat", ".h5", ".hdf5", ".edf", ".npy", ".rec", ".bin"}


@tool
def setup_v4_output_dir(dataset_name: str, base_dir: str = "output") -> dict[str, Any]:
    """Create the V4 structured output directory tree.

    Creates:
        {base_dir}/{dataset_name}/v4/
            scripts/subset/     <- subset validation scripts
            scripts/full/       <- full-run scripts
            scripts/ecg_stats/  <- ECG statistics scripts
            outputs/subset/     <- subset script outputs
            outputs/full/       <- full-run script outputs
            outputs/ecg_stats/  <- ECG stat outputs
            logs/               <- execution logs

    Returns: dict with all directory paths keyed by role.
    """
    try:
        root = Path(base_dir).resolve() / dataset_name / "v4"
        dirs = {
            "root": root,
            "scripts_subset": root / "scripts" / "subset",
            "scripts_full": root / "scripts" / "full",
            "scripts_ecg_stats": root / "scripts" / "ecg_stats",
            "outputs_subset": root / "outputs" / "subset",
            "outputs_full": root / "outputs" / "full",
            "outputs_ecg_stats": root / "outputs" / "ecg_stats",
            "logs": root / "logs",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        return {
            "status": "ok",
            "root": str(root),
            "dirs": {k: str(v) for k, v in dirs.items()},
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def read_generated_file(file_path: str, max_bytes: int = 500_000) -> dict[str, Any]:
    """Read a script-generated output file (CSV, JSON, or text).

    ONLY reads files that are script-generated outputs (CSV, JSON, MD, TEX, TXT).
    Refuses to read raw ECG signal files (.dat, .h5, .hdf5, .edf, .npy).
    Caps content at max_bytes to avoid flooding LLM context.
    """
    try:
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Refuse signal files
        if suffix in _SIGNAL_EXTENSIONS:
            return {
                "status": "error",
                "error": (
                    f"Refusing to read raw ECG signal file '{path.name}' "
                    f"(extension '{suffix}'). V4 core constraint: agents must not "
                    "load raw signal data."
                ),
            }

        if not path.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}

        size = path.stat().st_size
        content = path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
        truncated = size > max_bytes

        return {
            "status": "ok",
            "path": str(path),
            "name": path.name,
            "size_bytes": size,
            "content": content,
            "truncated": truncated,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def list_generated_files(output_dir: str) -> dict[str, Any]:
    """List all files generated in a V4 output directory.

    Returns file names, sizes, and suffixes. Excludes raw ECG signal files.
    """
    try:
        path = Path(output_dir)
        if not path.exists():
            return {"status": "ok", "files": [], "count": 0}

        files = []
        for f in sorted(path.iterdir()):
            if f.is_file() and f.suffix.lower() not in _SIGNAL_EXTENSIONS:
                files.append({
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "suffix": f.suffix.lower(),
                })

        return {
            "status": "ok",
            "output_dir": str(path),
            "files": files,
            "count": len(files),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def write_execution_log(
    log_dir: str,
    script_name: str,
    execution_result: dict,
) -> dict[str, Any]:
    """Persist an execution result to a structured JSON log file.

    Saves {log_dir}/{script_name}.log.json with full execution metadata.
    """
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Sanitise script_name for use as filename
        safe_name = script_name.replace("/", "_").replace("\\", "_")
        log_file = log_path / f"{safe_name}.log.json"

        # Drop large generated_files content from the log to keep it compact
        log_data = dict(execution_result)
        if "generated_files" in log_data and isinstance(log_data["generated_files"], dict):
            log_data["generated_files"] = {
                k: f"<{len(v)} bytes>" for k, v in log_data["generated_files"].items()
            }

        log_file.write_text(json.dumps(log_data, indent=2, default=str))
        return {"status": "ok", "log_file": str(log_file)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Non-tool helper (used by executor directly) ────────────────────────────

def collect_generated_files(output_dir: str, size_cap_bytes: int = 100_000) -> dict[str, str]:
    """Read text content of files generated by a script.

    Reads CSV (≤100KB) and JSON (≤500KB) files from output_dir.
    Skips raw ECG signal files.
    """
    result: dict[str, str] = {}
    output_path = Path(output_dir)
    if not output_path.exists():
        return result

    for f in output_path.glob("*.csv"):
        if f.suffix.lower() not in _SIGNAL_EXTENSIONS and f.stat().st_size <= size_cap_bytes:
            try:
                result[f.name] = f.read_text(errors="replace")
            except Exception:
                pass

    for f in output_path.glob("*.json"):
        if f.suffix.lower() not in _SIGNAL_EXTENSIONS and f.stat().st_size <= 500_000:
            try:
                result[f.name] = f.read_text(errors="replace")
            except Exception:
                pass

    for f in output_path.glob("*.md"):
        if f.stat().st_size <= size_cap_bytes:
            try:
                result[f.name] = f.read_text(errors="replace")
            except Exception:
                pass

    return result
