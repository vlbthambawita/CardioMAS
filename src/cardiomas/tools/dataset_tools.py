from __future__ import annotations

import csv
from pathlib import Path

from cardiomas.schemas.tools import ToolResult


def inspect_dataset(dataset_path: str) -> ToolResult:
    root = Path(dataset_path)
    if not root.exists():
        return ToolResult(tool_name="inspect_dataset", ok=False, summary="", error=f"Dataset path not found: {dataset_path}")
    if not root.is_dir():
        return ToolResult(tool_name="inspect_dataset", ok=False, summary="", error=f"Dataset path is not a directory: {dataset_path}")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    extension_counts: dict[str, int] = {}
    csv_headers: dict[str, list[str]] = {}

    for path in files:
        suffix = path.suffix.lower() or "<no_extension>"
        extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
        if suffix == ".csv" and len(csv_headers) < 5:
            with path.open("r", encoding="utf-8", newline="") as handle:
                headers = next(csv.reader(handle), [])
            csv_headers[str(path.relative_to(root))] = headers

    summary = (
        f"Dataset inspection found {len(files)} file(s) across {len(extension_counts)} extension group(s)."
    )
    return ToolResult(
        tool_name="inspect_dataset",
        ok=True,
        summary=summary,
        data={
            "dataset_path": str(root),
            "total_files": len(files),
            "extension_counts": extension_counts,
            "csv_headers": csv_headers,
        },
    )
