from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cardiomas.agents.base import run_agent
from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.tools.data_tools import list_dataset_files, read_csv_metadata, compute_statistics

logger = logging.getLogger(__name__)


def analysis_agent(state: GraphState) -> GraphState:
    """Scan dataset files, extract metadata, compute statistics."""
    from cardiomas.llm_factory import get_llm

    opts = state.user_options
    info = state.dataset_info
    state.execution_log.append(LogEntry(agent="analysis", action="start"))

    # Determine local path
    local_path = opts.local_path or (str(info.local_path) if info and info.local_path else "")
    if not local_path:
        state.analysis_report = {
            "status": "skipped",
            "reason": "No local path — dataset not downloaded yet. Use --local-path to provide it.",
        }
        return state

    # List files
    files_result = list_dataset_files.invoke({"path": local_path, "max_depth": 3})
    files = files_result.get("files", [])

    # Find CSV metadata files
    csv_files = [f for f in files if f["suffix"] in (".csv", ".tsv")]
    metadata_sample: dict[str, Any] = {}
    for csv_file in csv_files[:3]:
        full_path = str(Path(local_path) / csv_file["path"])
        result = read_csv_metadata.invoke({"path": full_path, "nrows": 5})
        if "error" not in result:
            metadata_sample[csv_file["path"]] = result
            break

    # Compute statistics on key columns if metadata found
    stats: dict[str, Any] = {}
    if metadata_sample:
        first_csv_path, first_csv_info = next(iter(metadata_sample.items()))
        full_csv = str(Path(local_path) / first_csv_path)
        cols = first_csv_info.get("columns", [])[:5]
        if cols:
            stats = compute_statistics.invoke({"data_path": full_csv, "columns": cols})

    # Ask LLM to interpret the findings
    llm = get_llm(prefer_cloud=opts.use_cloud_llm)
    context = (
        f"Dataset: {info.name if info else 'unknown'}\n"
        f"Files found: {len(files)} total\n"
        f"File types: {list({f['suffix'] for f in files})}\n"
        f"Metadata sample: {str(metadata_sample)[:2000]}\n"
        f"Statistics: {str(stats)[:1000]}"
    )
    prompt = (
        "Analyze this ECG dataset structure and produce a report covering:\n"
        "1. Total number of ECG records (estimate if not exact)\n"
        "2. Unique patient count (if determinable)\n"
        "3. Diagnostic label distribution (top labels)\n"
        "4. Patient ID field name (field to use for grouping to prevent leakage)\n"
        "5. ECG record identifier field\n"
        "6. Missing data issues\n"
        "7. Recommended split strategy (patient-level vs record-level, stratify by what field)\n\n"
        "Base your answer only on what is in the data above. Cite field names exactly."
    )
    report_text = run_agent(llm, "data_analysis", prompt, context)

    state.analysis_report = {
        "status": "complete",
        "local_path": local_path,
        "num_files": len(files),
        "metadata_files": list(metadata_sample.keys()),
        "statistics": stats,
        "report": report_text,
        "id_field": info.ecg_id_field if info else "record_id",
    }
    state.execution_log.append(LogEntry(agent="analysis", action="complete", detail=f"{len(files)} files"))
    return state
