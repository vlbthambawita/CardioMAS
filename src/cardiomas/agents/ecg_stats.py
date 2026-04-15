"""
ECG Statistics Agent (V4)

ECG-domain expert. Generates scripts for clinical-grade statistical analysis.
Produces CSV summaries, publication-ready tables, and plots.
Operates exclusively on script generation — all execution is delegated to executor.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from cardiomas.schemas.state import GraphState, LogEntry, ScriptRecord

logger = logging.getLogger(__name__)


def ecg_stats_agent(state: GraphState) -> GraphState:
    """Generate ECG statistical analysis scripts.

    Uses LLM to generate 5 scripts (10-14) that compute clinical statistics
    on the full dataset. Scripts are added to v4_ecg_stats_scripts.
    """
    from cardiomas.agents.base import run_structured_agent, AgentOutputError
    from cardiomas.llm_factory import get_llm_for_agent
    from cardiomas.verbose import vprint
    from pydantic import BaseModel, Field

    opts = state.user_options
    info = state.dataset_info
    state.execution_log.append(LogEntry(agent="ecg_stats", action="start"))
    vprint("ecg_stats", "generating ECG statistical analysis scripts…")

    # ── Determine paths ────────────────────────────────────────────────────
    dataset_name = (info.name if info else None) or "dataset"
    dataset_path = (
        opts.local_path
        or (str(info.local_path) if info and info.local_path else "")
        or (opts.dataset_source if not opts.dataset_source.startswith("http") else "")
    )
    if not dataset_path:
        dataset_path = "/path/to/dataset"

    # ECG stats dir sits under the v4 output tree
    v4_root = Path(state.v4_output_dir) if state.v4_output_dir else (
        Path(opts.output_dir) / dataset_name / "v4"
    )
    ecg_stats_scripts_dir = v4_root / "scripts" / "ecg_stats"
    ecg_stats_output_dir = v4_root / "outputs" / "ecg_stats"
    ecg_stats_scripts_dir.mkdir(parents=True, exist_ok=True)
    ecg_stats_output_dir.mkdir(parents=True, exist_ok=True)
    state.v4_ecg_stats_dir = str(ecg_stats_output_dir)

    # ── Gather context from previous phases ────────────────────────────────
    analysis = state.analysis_report or {}
    # Collect metadata CSV paths from execution summary
    metadata_csv_paths: list[str] = []
    for result in state.v4_execution_results:
        if result.script_name.startswith("01") and result.stdout:
            # Try to extract CSV path from stdout
            import re
            m = re.search(r"CSV_PATH\s*=\s*(.+)", result.stdout)
            if m:
                metadata_csv_paths.append(m.group(1).strip())

    context: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "execution_summary": state.v4_execution_summary[:3000] if state.v4_execution_summary else "",
        "analysis_report": {
            "id_field": analysis.get("id_field", "record_id"),
            "label_field": analysis.get("label_field"),
            "label_type": analysis.get("label_type", "unknown"),
            "num_records": analysis.get("num_records") or (info.num_records if info else None),
            "available_fields": analysis.get("available_fields", [])[:20],
        },
        "output_dir": str(ecg_stats_output_dir),
        "metadata_csv_paths": metadata_csv_paths,
        "subset_stats_csv": state.v4_generated_files.get("stats.csv", "")[:1000],
    }

    class ScriptBundle(BaseModel):
        scripts: dict[str, str] = Field(
            description="Dict of script_name -> Python script content"
        )
        notes: str = Field(default="", description="Any caveats")

    prompt = (
        "Generate the 5 ECG statistical analysis scripts described in your system prompt.\n"
        "Context:\n" + json.dumps(context, indent=2, default=str)
    )

    llm = get_llm_for_agent(
        "ecg_stats",
        prefer_cloud=opts.use_cloud_llm,
        agent_llm_map=opts.agent_llm_map,
    )

    try:
        bundle: ScriptBundle = run_structured_agent(
            llm, "ecg_stats", prompt, ScriptBundle, max_retries=1
        )
    except AgentOutputError as exc:
        logger.error(f"ecg_stats: LLM failed to generate scripts: {exc}")
        state.errors.append(f"ecg_stats: script generation failed: {exc}")
        return state

    # ── Write scripts to disk ──────────────────────────────────────────────
    scripts_written: dict[str, ScriptRecord] = {}
    output_dir_str = str(ecg_stats_output_dir)

    for script_name, script_content in bundle.scripts.items():
        # Inject correct constants
        script_content = _inject_constants(script_content, output_dir_str, dataset_path)

        script_path = ecg_stats_scripts_dir / script_name
        try:
            script_path.write_text(script_content)
        except Exception as e:
            state.errors.append(f"ecg_stats: could not write {script_name}: {e}")
            continue

        sha256 = hashlib.sha256(script_content.encode()).hexdigest()
        record = ScriptRecord(
            name=script_name,
            path=str(script_path),
            purpose=_script_purpose(script_name),
            output_dir=output_dir_str,
            timeout=600,
            phase="ecg_stats",
            sha256=sha256,
        )
        scripts_written[script_name] = record
        vprint("ecg_stats", f"  wrote {script_name} ({len(script_content)} chars)")

    if not scripts_written:
        state.errors.append("ecg_stats: no scripts were written successfully")
        return state

    # ── Update state ───────────────────────────────────────────────────────
    state.v4_ecg_stats_scripts.update(scripts_written)
    state.v4_generated_scripts.update(scripts_written)
    state.v4_pipeline_phase = "ecg_stats_run"

    state.execution_log.append(LogEntry(
        agent="ecg_stats",
        action="complete",
        detail=f"generated {len(scripts_written)} ECG stat scripts: {list(scripts_written.keys())}",
    ))
    vprint("ecg_stats", f"complete — {len(scripts_written)} scripts generated")
    return state


# ── Helpers ────────────────────────────────────────────────────────────────

def _inject_constants(script_content: str, output_dir: str, dataset_path: str) -> str:
    """Ensure OUTPUT_DIR and DATASET_PATH constants are correctly set."""
    lines = script_content.splitlines()
    new_lines = []
    injected_output = False
    injected_dataset = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("OUTPUT_DIR") and "=" in stripped and not injected_output:
            new_lines.append(f'OUTPUT_DIR = r"{output_dir}"')
            injected_output = True
        elif stripped.startswith("DATASET_PATH") and "=" in stripped and not injected_dataset:
            new_lines.append(f'DATASET_PATH = r"{dataset_path}"')
            injected_dataset = True
        else:
            new_lines.append(line)

    result = "\n".join(new_lines)
    if not injected_output:
        result = f'OUTPUT_DIR = r"{output_dir}"\n' + result
    if not injected_dataset:
        result = f'DATASET_PATH = r"{dataset_path}"\n' + result
    return result


def _script_purpose(name: str) -> str:
    purposes = {
        "10": "Label/class frequency analysis with multi-label SCP code handling",
        "11": "Per-lead statistics from .hea header files (text only, no signals)",
        "12": "Signal quality check: completeness, duplicates, records per patient",
        "13": "Clinical plausibility: age outliers, invalid SCP codes, HR ranges",
        "14": "Publication-ready Table 1 in Markdown and LaTeX format",
    }
    prefix = name[:2]
    return purposes.get(prefix, f"ECG stats script {name}")
