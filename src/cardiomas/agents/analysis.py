from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cardiomas.agents.base import AgentOutputError, run_structured_agent
from cardiomas.schemas.agent_outputs import AnalysisOutput
from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def analysis_agent(state: GraphState) -> GraphState:
    """Analyze dataset structure and produce a structured AnalysisOutput.

    V4 mode: reads executor script outputs only (no direct file access).
    V3 fallback: uses DatasetMapper if v4_execution_summary is empty.
    """
    from cardiomas.llm_factory import get_llm_for_agent

    opts = state.user_options
    info = state.dataset_info
    state.execution_log.append(LogEntry(agent="analysis", action="start"))
    vprint("analysis", "analyzing dataset (V4 indirect mode)…")

    # ── V4 mode: use executor outputs ──────────────────────────────────────
    if state.v4_execution_summary or state.v4_execution_results:
        return _analysis_from_executor(state)

    # ── V3 fallback: direct file access (backwards compatibility) ──────────
    vprint("analysis", "[dim]no V4 executor output — falling back to V3 direct scan[/dim]")
    return _analysis_v3_fallback(state)


def _analysis_from_executor(state: GraphState) -> GraphState:
    """V4 path: build analysis from executor script outputs only."""
    from cardiomas.llm_factory import get_llm_for_agent

    opts = state.user_options
    info = state.dataset_info

    context = _build_analysis_context(state)
    if not context.strip():
        vprint("analysis", "[yellow]executor output is empty — skipping analysis[/yellow]")
        state.analysis_report = {
            "status": "skipped",
            "reason": "No executor output available for analysis.",
        }
        return state

    prompt = (
        "Analyze this ECG dataset structure and produce a structured analysis.\n"
        "Base your answer ONLY on the data provided in the context.\n"
        "Use exact field/column names as they appear in the data.\n"
        "For 'recommended_strategy' choose from: "
        "patient_stratified, record_stratified, patient_random, record_random, official.\n"
        "For 'label_type': 'single' if one label per record, 'multi' if list/dict, 'none' if absent.\n"
        "For 'available_fields': list ALL column names found in metadata files."
    )

    llm = get_llm_for_agent(
        "analysis",
        prefer_cloud=opts.use_cloud_llm,
        agent_llm_map=opts.agent_llm_map,
    )

    try:
        output: AnalysisOutput = run_structured_agent(
            llm, "data_analysis", prompt, AnalysisOutput, extra_context=context
        )
    except AgentOutputError as exc:
        logger.error(f"analysis_agent (V4): structured output failed — {exc}")
        state.errors.append(f"analysis: {exc}")
        output = AnalysisOutput(
            num_records=0,
            id_field=info.ecg_id_field if info else "record_id",
            recommended_strategy="patient_random",
            notes=f"Structured analysis failed: {exc}",
        )

    # ── Store in state ─────────────────────────────────────────────────────
    dataset_name = info.name if info else "unknown"
    state.analysis_report = {
        "status": "complete",
        "source": "v4_executor",
        "report": _build_report_text(output),
        "id_field": output.id_field,
        "patient_id_field": output.patient_id_field,
        "label_field": output.label_field,
        "label_type": output.label_type,
        "recommended_strategy": output.recommended_strategy,
        "available_fields": output.available_fields,
        "num_records": output.num_records,
        "dataset_map": None,  # V4 has no DatasetMap
    }

    # ── Save analysis report ───────────────────────────────────────────────
    out_dir = Path(opts.output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "analysis_report.md"
    report_file.write_text(
        f"# Analysis Report — {dataset_name}\n\n{_build_report_text(output)}\n"
    )
    vprint("analysis", f"saved report → {report_file}")

    state.execution_log.append(LogEntry(
        agent="analysis",
        action="complete",
        detail=f"{output.num_records} records (V4)",
    ))
    vprint(
        "analysis",
        f"complete — {output.num_records} records"
        f" id={output.id_field}"
        f" label={output.label_field} ({output.label_type})"
        f" strategy={output.recommended_strategy}",
    )
    return state


def _analysis_v3_fallback(state: GraphState) -> GraphState:
    """V3 backward-compatible path: direct file access via DatasetMapper."""
    from cardiomas.llm_factory import get_llm_for_agent
    from cardiomas.tools.data_tools import compute_statistics, list_dataset_files, read_csv_metadata
    from cardiomas.tools.shell_tools import find_dataset_files

    opts = state.user_options
    info = state.dataset_info

    local_path = (
        opts.local_path
        or (str(info.local_path) if info and info.local_path else "")
        or (opts.dataset_source if not opts.dataset_source.startswith("http") else "")
    )
    if not local_path or not Path(local_path).exists():
        vprint("analysis", "no local path — skipping file scan")
        state.analysis_report = {
            "status": "skipped",
            "reason": "No local path — dataset not downloaded yet.",
        }
        return state

    effective_path = _find_data_root(local_path)
    if effective_path != local_path:
        vprint("analysis", f"data root resolved: {local_path} → {effective_path}")

    dataset_map = _build_dataset_map(effective_path, state)

    files_result = list_dataset_files.invoke({"path": effective_path, "max_depth": 5})
    files = files_result.get("files", [])
    vprint("analysis", f"found {len(files)} files")

    csv_files = [f for f in files if f["suffix"] in (".csv", ".tsv")]
    metadata_sample: dict[str, Any] = {}
    stats: dict[str, Any] = {}

    for csv_file in csv_files[:3]:
        full_path = str(Path(effective_path) / csv_file["path"])
        result = read_csv_metadata.invoke({"path": full_path, "nrows": 5})
        if "error" not in result:
            metadata_sample[csv_file["path"]] = result
            if not stats:
                cols = result.get("columns", [])[:5]
                if cols:
                    stats = compute_statistics.invoke({"data_path": full_path, "columns": cols})
            break

    context_parts = [
        f"Dataset: {info.name if info else 'unknown'}",
        f"Local path: {effective_path}",
        f"Total files: {len(files)}",
        f"File types: {list({f['suffix'] for f in files})}",
    ]
    if dataset_map:
        context_parts += [
            f"\n## DatasetMap (from structured scan)",
            f"Total record IDs found: {len(dataset_map.get('all_record_ids', []))}",
            f"Format distribution: {dataset_map.get('format_distribution', {})}",
            f"ID field (confirmed): {dataset_map.get('id_field', 'unknown')}",
            f"Patient ID field: {dataset_map.get('patient_id_field')}",
            f"Label field: {dataset_map.get('label_field')}",
            f"Label type: {dataset_map.get('label_type')}",
            f"Label values (sample): {dataset_map.get('label_values', [])[:10]}",
            f"Missing data fraction: {dataset_map.get('missing_data_fraction', 0):.3f}",
            f"Available fields: {dataset_map.get('available_fields', [])}",
        ]
    else:
        context_parts += [
            f"\n## CSV Metadata Sample",
            f"{str(metadata_sample)[:2000]}",
            f"\n## Column Statistics",
            f"{str(stats)[:1000]}",
        ]

    context = "\n".join(str(p) for p in context_parts)
    prompt = (
        "Analyze this ECG dataset structure and produce a structured analysis.\n"
        "Base your answer ONLY on the data provided in the context.\n"
        "Use exact field/column names as they appear in the data.\n"
        "For 'recommended_strategy' choose from: "
        "patient_stratified, record_stratified, patient_random, record_random, official.\n"
        "For 'label_type': 'single' if one label per record, 'multi' if list/dict, 'none' if absent.\n"
        "For 'available_fields': list ALL column names found in metadata files."
    )

    llm = get_llm_for_agent(
        "analysis",
        prefer_cloud=opts.use_cloud_llm,
        agent_llm_map=opts.agent_llm_map,
    )

    try:
        output: AnalysisOutput = run_structured_agent(
            llm, "data_analysis", prompt, AnalysisOutput, extra_context=context
        )
    except AgentOutputError as exc:
        logger.error(f"analysis_agent (V3): structured output failed — {exc}")
        state.errors.append(f"analysis: {exc}")
        output = AnalysisOutput(
            num_records=len(dataset_map.get("all_record_ids", [])) if dataset_map else len(files),
            id_field=info.ecg_id_field if info else "record_id",
            recommended_strategy="patient_random",
            notes=f"Structured analysis failed: {exc}",
        )

    if dataset_map:
        output = _verify_analysis_output(output, dataset_map, state)

    dataset_name = info.name if info else "unknown"
    state.analysis_report = {
        "status": "complete",
        "source": "v3_direct",
        "local_path": effective_path,
        "num_files": len(files),
        "metadata_files": list(metadata_sample.keys()),
        "statistics": stats,
        "report": _build_report_text(output),
        "id_field": output.id_field,
        "patient_id_field": output.patient_id_field,
        "label_field": output.label_field,
        "label_type": output.label_type,
        "recommended_strategy": output.recommended_strategy,
        "available_fields": output.available_fields,
        "dataset_map": dataset_map,
    }

    out_dir = Path(opts.output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "analysis_report.md"
    report_file.write_text(f"# Analysis Report — {dataset_name}\n\n{_build_report_text(output)}\n")
    vprint("analysis", f"saved report → {report_file}")

    state.execution_log.append(
        LogEntry(agent="analysis", action="complete", detail=f"{len(files)} files (V3)")
    )
    vprint(
        "analysis",
        f"complete (V3) — {output.num_records} records"
        f" id={output.id_field}"
        f" label={output.label_field} ({output.label_type})"
        f" strategy={output.recommended_strategy}",
    )
    return state


def _build_analysis_context(state: GraphState) -> str:
    """Build LLM context purely from executor outputs (V4 path).

    Never reads dataset files directly.
    """
    parts: list[str] = []

    # 1. Execution summary (text aggregated by executor)
    if state.v4_execution_summary:
        parts.append(f"## Script Execution Summary\n{state.v4_execution_summary}")

    # 2. Key stdout excerpts per script
    for result in state.v4_execution_results:
        parts.append(
            f"### Script: {result.script_name}\n"
            f"Exit code: {result.exit_code}\n"
            f"Verification: {'PASS' if result.verification_passed else 'FAIL'}\n"
            f"Output:\n{result.stdout[:2000]}\n"
        )

    # 3. Generated CSV content (stats.csv, class_dist.csv) — script outputs only
    for fname, content in state.v4_generated_files.items():
        if fname.endswith(".csv"):
            parts.append(f"### Generated file: {fname}\n{content[:3000]}\n")

    return "\n\n".join(parts)


def _find_data_root(local_path: str) -> str:
    """
    Given a top-level dataset directory, locate the deepest directory that
    contains ECG data files (CSV, WFDB, HDF5, EDF).

    For datasets like PTB-XL with nested physionet/files/ptb-xl/1_0_1/ layouts,
    this returns the inner directory so the DatasetMapper scans the right place.
    Uses the find_dataset_files bash tool for recursive discovery.
    """
    result = find_dataset_files.invoke({
        "root_path": local_path,
        "extensions": [".csv", ".tsv", ".hea", ".h5", ".hdf5", ".edf"],
        "max_depth": 8,
        "max_results": 20,
    })
    files = result.get("files", [])
    if not files:
        return local_path

    # Find the shallowest CSV/TSV (metadata file) and use its parent directory
    for f in files:
        if f["suffix"] in (".csv", ".tsv"):
            parent = str(Path(f["path"]).parent)
            candidate = str(Path(local_path) / parent)
            if Path(candidate).is_dir():
                return candidate

    # No CSV found — use the directory containing the shallowest data file
    parent = str(Path(files[0]["path"]).parent)
    candidate = str(Path(local_path) / parent)
    if Path(candidate).is_dir():
        return candidate

    return local_path


def _build_dataset_map(local_path: str, state: GraphState) -> dict | None:
    """Try to build a DatasetMap using the Phase 2 mapper; return None if not available."""
    try:
        from cardiomas.mappers.dataset_mapper import DatasetMapper
        vprint("analysis", "running DatasetMapper (Phase 2)…")
        mapper = DatasetMapper(local_path)
        dataset_map_obj = mapper.build()
        dm = dataset_map_obj.model_dump()
        # Attach to state if GraphState has dataset_map field (Phase 2+)
        if hasattr(state, "dataset_map"):
            state.dataset_map = dataset_map_obj
        vprint(
            "analysis",
            f"DatasetMap: {len(dm.get('all_record_ids', []))} records"
            f" formats={dm.get('format_distribution')}"
            f" id_field={dm.get('id_field')}",
        )
        return dm
    except ImportError:
        logger.debug("DatasetMapper not yet available (Phase 2 not installed)")
        return None
    except Exception as exc:
        logger.warning(f"DatasetMapper failed: {exc} — falling back to CSV scan")
        return None


def _verify_analysis_output(
    output: AnalysisOutput,
    dataset_map: dict,
    state: GraphState,
) -> AnalysisOutput:
    """Phase 4 hook: verify field names and record count against DatasetMap."""
    try:
        from cardiomas.agents.verification import verify_analysis_output
        return verify_analysis_output(output, dataset_map, state)
    except ImportError:
        return output
    except Exception as exc:
        logger.warning(f"verify_analysis_output failed: {exc}")
        return output


def _build_report_text(output: AnalysisOutput) -> str:
    lines = [
        f"- **Records:** {output.num_records}",
        f"- **ID field:** `{output.id_field}`",
        f"- **Patient ID field:** `{output.patient_id_field}`",
        f"- **Label field:** `{output.label_field}` ({output.label_type})",
        f"- **Recommended strategy:** {output.recommended_strategy}",
        f"- **Missing data:** {output.missing_data_fraction:.1%}",
    ]
    if output.label_values:
        lines.append(f"- **Label values (sample):** {output.label_values[:10]}")
    if output.notes:
        lines.append(f"\n**Notes:** {output.notes}")
    return "\n".join(lines)
