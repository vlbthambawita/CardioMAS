from __future__ import annotations

import csv
from pathlib import Path

from cardiomas.schemas.tools import ToolResult

_FOLDER_TREE_MAX_FILES = 200  # cap to avoid flooding the context window


def list_folder_structure(dataset_path: str, max_depth: int = 4) -> ToolResult:
    """Return a human-readable tree of the directory layout at *dataset_path*.

    Walks the directory up to *max_depth* levels deep and lists every file and
    sub-directory. For each CSV/TSV file found it also reads the header row so
    the agent knows which columns are available without having to open the file
    separately. The output is intentionally compact: directories are shown with
    a trailing ``/`` and files are shown with their size in bytes. If the tree
    exceeds ``_FOLDER_TREE_MAX_FILES`` entries it is truncated with a note so
    the LLM is not overwhelmed.

    Intended use: call this tool first when you need to understand what data
    files are present and how they are organised before deciding which files to
    read or which analysis script to generate.

    Args:
        dataset_path: Absolute or relative path to the root directory to inspect.
        max_depth: How many directory levels to descend (default 4).

    Returns:
        ToolResult whose ``summary`` contains the printable tree and whose
        ``data`` dict exposes ``tree_lines``, ``total_files``,
        ``total_dirs``, ``csv_headers``, and ``truncated`` for programmatic use.
    """
    root = Path(dataset_path)
    if not root.exists():
        return ToolResult(
            tool_name="list_folder_structure",
            ok=False,
            summary="",
            error=f"Path not found: {dataset_path}",
        )
    if not root.is_dir():
        return ToolResult(
            tool_name="list_folder_structure",
            ok=False,
            summary="",
            error=f"Path is not a directory: {dataset_path}",
        )

    tree_lines: list[str] = [f"{root.name}/"]
    csv_headers: dict[str, list[str]] = {}
    total_files = 0
    total_dirs = 0
    truncated = False

    def _walk(directory: Path, prefix: str, depth: int) -> None:
        nonlocal total_files, total_dirs, truncated
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            extension = "│   " if i < len(entries) - 1 else "    "
            if entry.is_dir():
                total_dirs += 1
                tree_lines.append(f"{prefix}{connector}{entry.name}/")
                _walk(entry, prefix + extension, depth + 1)
            else:
                total_files += 1
                if total_files > _FOLDER_TREE_MAX_FILES:
                    truncated = True
                    return
                size_kb = entry.stat().st_size / 1024
                size_str = f"{size_kb:.1f} KB" if size_kb >= 1 else f"{entry.stat().st_size} B"
                tree_lines.append(f"{prefix}{connector}{entry.name}  [{size_str}]")
                # Read CSV/TSV headers for quick column preview
                if entry.suffix.lower() in {".csv", ".tsv"} and len(csv_headers) < 10:
                    try:
                        delimiter = "\t" if entry.suffix.lower() == ".tsv" else ","
                        with entry.open("r", encoding="utf-8", newline="") as fh:
                            headers = next(csv.reader(fh, delimiter=delimiter), [])
                        if headers:
                            rel = str(entry.relative_to(root))
                            csv_headers[rel] = headers
                    except Exception:
                        pass

    _walk(root, "", 1)

    if truncated:
        tree_lines.append(f"... (truncated after {_FOLDER_TREE_MAX_FILES} files)")

    header_lines: list[str] = []
    if csv_headers:
        header_lines.append("\nCSV/TSV column headers:")
        for fname, cols in csv_headers.items():
            header_lines.append(f"  {fname}: {', '.join(cols)}")

    summary = "\n".join(tree_lines + header_lines)
    summary += f"\n\nTotal: {total_files} file(s), {total_dirs} sub-directory/ies"
    if truncated:
        summary += f" (tree truncated at {_FOLDER_TREE_MAX_FILES} files)"

    return ToolResult(
        tool_name="list_folder_structure",
        ok=True,
        summary=summary,
        data={
            "dataset_path": str(root),
            "tree_lines": tree_lines,
            "total_files": total_files,
            "total_dirs": total_dirs,
            "csv_headers": csv_headers,
            "truncated": truncated,
        },
    )


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
