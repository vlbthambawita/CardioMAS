"""
Shell tools — safe read-only subprocess commands for agents to discover dataset structure.

Allowed commands (read-only, no destructive operations):
    find, ls, ls -la, head, wc, stat, file, du, tree, cat (text files ≤ 10 KB)

All commands run with a 30-second timeout. Output is capped at 4000 chars.
"""
from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Whitelist of executable names that are safe (read-only, non-destructive)
_ALLOWED_EXECUTABLES = {
    "find", "ls", "head", "tail", "wc", "stat", "file", "du",
    "tree", "cat", "less", "more", "grep", "awk", "sort", "uniq",
    "cut", "echo", "pwd", "which", "basename", "dirname",
}

_MAX_OUTPUT_CHARS = 4000
_TIMEOUT_SECONDS = 30


@tool
def run_bash_command(command: str, working_dir: str = "") -> dict[str, Any]:
    """Run a safe, read-only shell command to explore the dataset directory.

    Useful for discovering files, checking directory structure, counting records, etc.
    Examples:
        find /data/ptb-xl -name "*.csv" -maxdepth 5
        ls -la /data/ptb-xl/physionet/files/ptb-xl/1_0_1/
        head -3 /data/ptb-xl/physionet/files/ptb-xl/1_0_1/ptbxl_database.csv
        find /data -name "*.hea" | wc -l

    Returns dict with stdout, stderr, exit_code, and truncated flag.
    """
    command = command.strip()
    if not command:
        return {"stdout": "", "stderr": "Empty command", "exit_code": 1}

    # Parse and validate the executable
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return {"stdout": "", "stderr": f"Command parse error: {exc}", "exit_code": 1}

    if not parts:
        return {"stdout": "", "stderr": "Empty command", "exit_code": 1}

    executable = Path(parts[0]).name  # strip any path prefix
    if executable not in _ALLOWED_EXECUTABLES:
        return {
            "stdout": "",
            "stderr": (
                f"Command '{executable}' is not in the allowed list. "
                f"Allowed: {sorted(_ALLOWED_EXECUTABLES)}"
            ),
            "exit_code": 1,
        }

    cwd = working_dir if working_dir and Path(working_dir).is_dir() else None

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            cwd=cwd,
        )
        stdout = result.stdout
        truncated = len(stdout) > _MAX_OUTPUT_CHARS
        return {
            "stdout": stdout[:_MAX_OUTPUT_CHARS],
            "stderr": result.stderr[:500],
            "exit_code": result.returncode,
            "truncated": truncated,
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {_TIMEOUT_SECONDS}s",
            "exit_code": -1,
            "truncated": False,
            "command": command,
        }
    except Exception as exc:
        logger.warning(f"run_bash_command failed: {exc}")
        return {
            "stdout": "",
            "stderr": str(exc),
            "exit_code": -1,
            "truncated": False,
            "command": command,
        }


@tool
def find_dataset_files(
    root_path: str,
    extensions: list[str] | None = None,
    max_depth: int = 8,
    max_results: int = 200,
) -> dict[str, Any]:
    """Recursively find dataset files under root_path matching given extensions.

    Defaults to ECG-relevant extensions: .csv, .tsv, .hea, .h5, .hdf5, .edf, .npy, .npz.
    Returns a list of relative paths sorted by depth (shallowest first).

    Args:
        root_path: Directory to search
        extensions: File extensions to include (e.g. ['.csv', '.hea'])
        max_depth: Maximum directory depth to search (default 8)
        max_results: Cap on number of results (default 200)
    """
    if extensions is None:
        extensions = [".csv", ".tsv", ".hea", ".h5", ".hdf5", ".edf", ".npy", ".npz", ".mat"]

    root = Path(root_path)
    if not root.exists():
        return {"error": f"Path not found: {root_path}", "files": []}

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    found: list[dict[str, Any]] = []

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        depth = len(p.relative_to(root).parts)
        if depth > max_depth:
            continue
        if p.suffix.lower() in exts:
            found.append({
                "path": str(p),
                "relative": str(p.relative_to(root)),
                "depth": depth,
                "size_bytes": p.stat().st_size,
                "suffix": p.suffix.lower(),
            })
        if len(found) >= max_results:
            break

    # Sort shallowest first
    found.sort(key=lambda x: (x["depth"], x["relative"]))

    return {
        "root": str(root),
        "files": found[:max_results],
        "total_found": len(found),
        "extensions_searched": sorted(exts),
    }
