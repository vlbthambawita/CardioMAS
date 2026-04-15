"""
Data Engineering Agent (V4)

Generates Python scripts to explore ECG dataset structure, extract metadata,
compute statistics, and produce splits. NEVER reads dataset files directly —
all dataset understanding comes through executing scripts and reading their output.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from cardiomas.schemas.state import GraphState, LogEntry, ScriptRecord

logger = logging.getLogger(__name__)


def data_engineer_agent(state: GraphState) -> GraphState:
    """Generate exploration and split scripts for the dataset.

    Does NOT read dataset files directly. Produces scripts that will be
    executed by executor_agent.
    """
    from cardiomas.agents.base import run_structured_agent, AgentOutputError
    from cardiomas.llm_factory import get_llm_for_agent
    from cardiomas.tools.v4_output_tools import setup_v4_output_dir
    from cardiomas.verbose import vprint
    from pydantic import BaseModel, Field

    opts = state.user_options
    info = state.dataset_info
    state.execution_log.append(LogEntry(agent="data_engineer", action="start"))
    vprint("data_engineer", "generating dataset exploration scripts (V4)…")

    # ── Determine dataset path ─────────────────────────────────────────────
    dataset_path = (
        opts.local_path
        or (str(info.local_path) if info and info.local_path else "")
        or (opts.dataset_source if not opts.dataset_source.startswith("http") else "")
    )
    if not dataset_path:
        dataset_path = "/path/to/dataset"
        vprint("data_engineer", "[yellow]no local_path set — using placeholder path[/yellow]")

    dataset_name = (info.name if info else None) or Path(dataset_path).name or "dataset"

    # ── Set up V4 output directory ─────────────────────────────────────────
    v4_dirs = setup_v4_output_dir.invoke({
        "dataset_name": dataset_name,
        "base_dir": opts.output_dir,
    })
    if v4_dirs.get("status") != "ok":
        state.errors.append(f"data_engineer: failed to create V4 output dirs: {v4_dirs.get('error')}")
        return state

    state.v4_output_dir = v4_dirs["root"]
    dirs = v4_dirs["dirs"]
    subset_size = opts.v4_subset_size if hasattr(opts, "v4_subset_size") else state.v4_subset_size

    # ── Build context ──────────────────────────────────────────────────────
    paper = state.paper_findings or {}
    split_ratios = opts.custom_split or {"train": 0.70, "val": 0.15, "test": 0.15}

    context: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "id_field_hint": info.ecg_id_field if info else "record_id",
        "num_records_estimate": info.num_records if info else None,
        "subset_size": subset_size,
        "seed": opts.seed,
        "output_dir": dirs.get("outputs_subset", str(Path(state.v4_output_dir) / "outputs" / "subset")),
        "split_ratios": split_ratios,
        "stratify_by": opts.stratify_by,
        "paper_methodology": paper.get("split_methodology", "") or paper.get("analysis", ""),
        "refinement_context": None,
    }

    # ── Refinement mode: only regenerate the failed script ────────────────
    is_refinement = state.v4_refinement_context is not None
    if is_refinement:
        rc = state.v4_refinement_context
        context["refinement_context"] = {
            "failed_script": rc.failed_script,
            "error_message": rc.error_message,
            "stdout_excerpt": rc.stdout_excerpt,
            "suggested_fix": rc.suggested_fix,
            "attempt": rc.attempt,
        }
        vprint(
            "data_engineer",
            f"[yellow]refinement mode — regenerating '{rc.failed_script}' "
            f"(attempt {rc.attempt})[/yellow]",
        )
        # Increment attempt counter
        state.v4_refinement_rounds[rc.failed_script] = (
            state.v4_refinement_rounds.get(rc.failed_script, 0) + 1
        )

    # ── Output schema for structured agent ────────────────────────────────
    class ScriptBundle(BaseModel):
        scripts: dict[str, str] = Field(
            description="Dict of script_name -> Python script content"
        )
        notes: str = Field(default="", description="Any caveats about the generated scripts")

    prompt = (
        "Generate the dataset exploration and split scripts described in your system prompt.\n"
        "Context:\n" + json.dumps(context, indent=2, default=str)
    )
    if is_refinement and state.v4_refinement_context:
        rc = state.v4_refinement_context
        prompt = (
            f"Regenerate ONLY the failed script '{rc.failed_script}' to fix this error:\n"
            f"Error: {rc.error_message}\n"
            f"Stdout excerpt: {rc.stdout_excerpt}\n\n"
            "Context:\n" + json.dumps(context, indent=2, default=str)
        )

    llm = get_llm_for_agent(
        "data_engineer",
        prefer_cloud=opts.use_cloud_llm,
        agent_llm_map=opts.agent_llm_map,
    )

    try:
        bundle: ScriptBundle = run_structured_agent(
            llm, "data_engineer", prompt, ScriptBundle, max_retries=1
        )
    except AgentOutputError as exc:
        logger.error(f"data_engineer: LLM failed to generate scripts: {exc}")
        state.errors.append(f"data_engineer: script generation failed: {exc}")
        return state

    # ── Write scripts to disk ──────────────────────────────────────────────
    scripts_subset_dir = Path(dirs.get("scripts_subset", str(Path(state.v4_output_dir) / "scripts" / "subset")))
    scripts_full_dir = Path(dirs.get("scripts_full", str(Path(state.v4_output_dir) / "scripts" / "full")))
    outputs_subset_dir = dirs.get("outputs_subset", str(Path(state.v4_output_dir) / "outputs" / "subset"))
    outputs_full_dir = dirs.get("outputs_full", str(Path(state.v4_output_dir) / "outputs" / "full"))

    scripts_written: dict[str, ScriptRecord] = {}

    for script_name, script_content in bundle.scripts.items():
        # Determine script directory and output directory
        if script_name.startswith("04"):
            script_dir = scripts_full_dir
            output_dir = outputs_full_dir
            phase = "full"
        else:
            script_dir = scripts_subset_dir
            output_dir = outputs_subset_dir
            phase = "subset"

        # Inject correct OUTPUT_DIR constant into the script
        script_content = _inject_output_dir(script_content, output_dir, dataset_path)

        script_path = script_dir / script_name
        try:
            script_path.write_text(script_content)
        except Exception as e:
            state.errors.append(f"data_engineer: could not write {script_name}: {e}")
            continue

        sha256 = hashlib.sha256(script_content.encode()).hexdigest()
        record = ScriptRecord(
            name=script_name,
            path=str(script_path),
            purpose=_script_purpose(script_name),
            output_dir=output_dir,
            timeout=300,
            phase=phase,
            sha256=sha256,
        )
        scripts_written[script_name] = record
        vprint("data_engineer", f"  wrote {script_name} ({len(script_content)} chars) → {script_path}")

    if not scripts_written:
        state.errors.append("data_engineer: no scripts were written successfully")
        return state

    # In refinement mode: update only the regenerated script
    if is_refinement and state.v4_refinement_context:
        failed = state.v4_refinement_context.failed_script
        if failed in scripts_written:
            state.v4_generated_scripts[failed] = scripts_written[failed]
            # Clear refinement context now that we've regenerated
            state.v4_refinement_context = None
        else:
            state.v4_generated_scripts.update(scripts_written)
    else:
        state.v4_generated_scripts.update(scripts_written)

    # ── Set pipeline phase ─────────────────────────────────────────────────
    if not is_refinement:
        state.v4_pipeline_phase = "subset_validation"

    state.execution_log.append(LogEntry(
        agent="data_engineer",
        action="complete",
        detail=f"generated {len(scripts_written)} scripts: {list(scripts_written.keys())}",
    ))
    vprint(
        "data_engineer",
        f"complete — {len(scripts_written)} scripts generated "
        f"(phase={state.v4_pipeline_phase})",
    )
    return state


# ── Helpers ────────────────────────────────────────────────────────────────

def _inject_output_dir(script_content: str, output_dir: str, dataset_path: str) -> str:
    """Ensure the script's OUTPUT_DIR constant is set to our managed directory.

    If the LLM used a different path, replace it. Also ensure DATASET_PATH
    points to the actual dataset.
    """
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

    # If constants weren't in the script, prepend them
    if not injected_output:
        result = f'OUTPUT_DIR = r"{output_dir}"\n' + result
    if not injected_dataset:
        result = f'DATASET_PATH = r"{dataset_path}"\n' + result

    return result


def _script_purpose(name: str) -> str:
    """Return a brief purpose description for a script by its name."""
    purposes = {
        "00": "Walk directory tree, count files by extension, detect format",
        "01": "Open metadata CSV, print column schema, write patient_map.json",
        "02": "Compute value_counts on label columns, write stats.csv",
        "03": "Generate deterministic splits on first SUBSET_SIZE records",
        "04": "Generate deterministic splits on full dataset",
    }
    prefix = name[:2]
    return purposes.get(prefix, f"Script {name}")
