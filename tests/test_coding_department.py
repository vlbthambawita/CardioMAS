from __future__ import annotations

import json
from pathlib import Path

from cardiomas.coding_department.tools import summarize_dataset_directory, write_dataset_summary


def test_summarize_dataset_directory_collects_extensions_and_headers(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "signals").mkdir()
    (dataset_dir / "metadata.csv").write_text("record_id,patient_id,label\nr1,p1,NORM\n", encoding="utf-8")
    (dataset_dir / "signals" / "rec_0001.txt").write_text("waveform", encoding="utf-8")

    summary = summarize_dataset_directory("tiny-demo", str(dataset_dir))

    assert summary.total_files == 2
    assert summary.extension_counts[".csv"] == 1
    assert summary.extension_counts[".txt"] == 1
    assert summary.csv_schemas["metadata.csv"] == ["record_id", "patient_id", "label"]


def test_write_dataset_summary_creates_json_csv_and_markdown(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.csv").write_text("record_id,label\nr1,NORM\n", encoding="utf-8")

    summary = summarize_dataset_directory("tiny-demo", str(dataset_dir))
    output_paths = write_dataset_summary(summary, str(tmp_path))

    assert Path(output_paths["dataset_inventory.json"]).exists()
    assert Path(output_paths["file_extensions.csv"]).exists()
    assert Path(output_paths["dataset_inventory.md"]).exists()
    stored = json.loads(Path(output_paths["dataset_inventory.json"]).read_text(encoding="utf-8"))
    assert stored["dataset_name"] == "tiny-demo"
