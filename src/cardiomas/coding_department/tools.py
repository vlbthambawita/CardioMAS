from __future__ import annotations

import csv
import json
from pathlib import Path

from cardiomas.coding_department.contracts import DatasetInventorySummary


def summarize_dataset_directory(dataset_name: str, dataset_dir: str) -> DatasetInventorySummary:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    directories = sorted(path for path in root.rglob("*") if path.is_dir())

    extension_counts: dict[str, int] = {}
    total_size_bytes = 0
    csv_schemas: dict[str, list[str]] = {}

    for file_path in files:
        suffix = file_path.suffix.lower() or "<no_extension>"
        extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
        total_size_bytes += file_path.stat().st_size
        if suffix in {".csv", ".tsv"} and len(csv_schemas) < 5:
            delimiter = "\t" if suffix == ".tsv" else ","
            with file_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                headers = next(reader, [])
            csv_schemas[str(file_path.relative_to(root))] = headers

    sample_files = [str(path.relative_to(root)) for path in files[:10]]

    return DatasetInventorySummary(
        dataset_name=dataset_name,
        dataset_dir=str(root),
        total_files=len(files),
        total_directories=len(directories),
        total_size_bytes=total_size_bytes,
        extension_counts=dict(sorted(extension_counts.items())),
        sample_files=sample_files,
        csv_schemas=csv_schemas,
    )


def write_dataset_summary(summary: DatasetInventorySummary, output_root: str) -> dict[str, str]:
    base = Path(output_root) / "tools" / summary.dataset_name
    base.mkdir(parents=True, exist_ok=True)

    json_path = base / "dataset_inventory.json"
    csv_path = base / "file_extensions.csv"
    md_path = base / "dataset_inventory.md"

    json_path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["extension", "count"])
        for extension, count in summary.extension_counts.items():
            writer.writerow([extension, count])

    lines = [
        f"# Dataset Inventory: {summary.dataset_name}",
        "",
        f"- Dataset dir: `{summary.dataset_dir}`",
        f"- Total files: {summary.total_files}",
        f"- Total directories: {summary.total_directories}",
        f"- Total size (bytes): {summary.total_size_bytes}",
        "",
        "## Extensions",
        "",
    ]
    for extension, count in summary.extension_counts.items():
        lines.append(f"- `{extension}`: {count}")

    if summary.csv_schemas:
        lines.extend(["", "## CSV Schemas", ""])
        for relative_path, headers in summary.csv_schemas.items():
            lines.append(f"- `{relative_path}`: {', '.join(headers) if headers else '(no header found)'}")

    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    return {
        "dataset_inventory.json": str(json_path),
        "file_extensions.csv": str(csv_path),
        "dataset_inventory.md": str(md_path),
    }
