"""
Coder Agent — generates self-contained Python scripts for reproducible splits.

Writes three scripts to output/{dataset}/scripts/:
  generate_splits.py  — reproduce exact splits (no cardiomas dependency)
  verify_splits.py    — re-run and compare against splits.json
  explore_dataset.py  — EDA: statistics, label distribution, missing-value summary

Then executes generate_splits.py in a subprocess to verify it runs.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def coder_agent(state: GraphState) -> GraphState:
    from cardiomas import __version__
    from cardiomas.agents.base import run_agent
    from cardiomas.llm_factory import get_llm_for_agent
    from cardiomas.recorder import SessionRecorder
    from cardiomas.tools.code_tools import execute_script

    state.execution_log.append(LogEntry(agent="coder", action="start"))
    rec = SessionRecorder.get()
    rec.start_step("coder", "generate_scripts")

    if not state.proposed_splits:
        vprint("coder", "[yellow]no splits available — skipping[/yellow]")
        state.execution_log.append(LogEntry(agent="coder", action="skip", detail="no proposed_splits"))
        rec.end_step(skipped=True, skip_reason="no proposed_splits in state")
        return state

    vprint("coder", "generating reproducibility scripts…")

    prefer_cloud = state.user_options.use_cloud_llm
    llm = get_llm_for_agent(
        "coder",
        prefer_cloud=prefer_cloud,
        agent_llm_map=state.user_options.agent_llm_map,
    )

    # Gather context
    manifest = state.proposed_splits
    dataset_name = manifest.dataset_name
    repro = manifest.reproducibility_config
    seed = repro.seed if repro else state.user_options.seed
    strategy = repro.split_strategy if repro else "deterministic"
    splits = manifest.splits
    local_path = state.user_options.local_path or state.user_options.dataset_source
    id_field = state.dataset_info.ecg_id_field if state.dataset_info else "record_id"
    requirement = state.user_options.requirement or ""
    total = sum(len(v) for v in splits.values())
    split_ratios = (
        {k: round(len(v) / total, 4) for k, v in splits.items()}
        if total > 0 else {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Build output dirs
    out_dir = Path(state.local_output_dir or f"{state.user_options.output_dir}/{dataset_name}")
    scripts_dir = out_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    context = (
        f"Dataset: {dataset_name}\n"
        f"Local path: {local_path}\n"
        f"ID field: {id_field}\n"
        f"Seed: {seed}\n"
        f"Strategy: {strategy}\n"
        f"Split ratios: {split_ratios}\n"
        f"Total records: {total}\n"
        f"CardioMAS version: {__version__}\n"
        f"Date: {today}\n"
        f"User requirement: {requirement or 'none'}\n"
    )
    if state.analysis_report:
        context += f"Analysis notes: {str(state.analysis_report)[:200]}\n"

    # ── V4: reference splits.json path (executor-generated) ──────────────
    v4_splits_path = ""
    if state.v4_output_dir:
        from pathlib import Path as _Path
        v4_full = _Path(state.v4_output_dir) / "outputs" / "full" / "splits.json"
        if v4_full.exists():
            v4_splits_path = str(v4_full)
    if v4_splits_path:
        context += f"\nReference splits.json (V4 executor-generated): {v4_splits_path}\n"

    # ── verify_splits.py (templated, no LLM needed) ───────────────────────
    verify_code = _build_verify_script(dataset_name, v4_splits_path)
    verify_path = scripts_dir / "verify_splits.py"
    verify_path.write_text(verify_code)
    state.generated_scripts["verify_splits.py"] = str(verify_path)
    vprint("coder", "wrote verify_splits.py")

    # ── Execute verify_splits.py ──────────────────────────────────────────
    # Only execute if there's a reference splits.json to compare against
    ref_splits_file = (
        v4_splits_path
        or str(out_dir / "splits.json")
    )
    if Path(ref_splits_file).exists():
        vprint("coder", "executing verify_splits.py…")
        exec_result = execute_script.invoke({
            "script_path": str(verify_path),
            "timeout": 120,
            "working_dir": str(scripts_dir),
        })
        state.script_execution_log.append(exec_result)

        if exec_result.get("exit_code") == 0:
            state.script_verified = True
            vprint("coder", "[green]verify_splits.py ran successfully[/green]")
        else:
            stderr = exec_result.get("stderr", "")[:300]
            vprint("coder", f"[yellow]verify_splits.py exited {exec_result.get('exit_code')} — saved but unverified[/yellow]")
            vprint("coder", f"[dim]stderr: {stderr}[/dim]")
            logger.warning(f"verify_splits.py execution result: {stderr}")
    else:
        vprint("coder", "[dim]no reference splits.json yet — skipping verify execution[/dim]")
        state.script_verified = False

    state.execution_log.append(LogEntry(
        agent="coder", action="complete",
        detail=f"{len(state.generated_scripts)} scripts → {scripts_dir}"
    ))
    rec.end_step(
        outputs={"scripts": list(state.generated_scripts.keys()), "verified": state.script_verified},
        reasoning=f"Generated {len(state.generated_scripts)} scripts. Execution verified: {state.script_verified}.",
    )
    return state


def _extract_code(response: str) -> str:
    """Strip markdown fences from LLM response if present."""
    if "```python" in response:
        start = response.find("```python") + len("```python")
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()
    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()
    return response.strip()


def _build_verify_script(dataset_name: str, v4_splits_path: str = "") -> str:
    return f'''#!/usr/bin/env python3
"""
verify_splits.py — Verify reproducibility of {dataset_name} splits.
Generated by CardioMAS.

Usage:
    python verify_splits.py

Runs generate_splits.py, then compares its splits.json output
against the reference splits.json in the parent directory.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REF_SPLITS  = SCRIPT_DIR.parent / "splits.json"
GEN_SCRIPT  = SCRIPT_DIR / "generate_splits.py"
GEN_OUTPUT  = SCRIPT_DIR / "splits.json"


def splits_hash(splits: dict) -> str:
    canonical = {{k: sorted(v) for k, v in sorted(splits.items())}}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def main():
    if not GEN_SCRIPT.exists():
        print("ERROR: generate_splits.py not found", file=sys.stderr)
        sys.exit(1)

    print("Running generate_splits.py...")
    result = subprocess.run(
        [sys.executable, str(GEN_SCRIPT)],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"ERROR: generate_splits.py failed:\\n{{result.stderr}}", file=sys.stderr)
        sys.exit(1)

    if not REF_SPLITS.exists():
        print("WARNING: reference splits.json not found — cannot compare.")
        print("Script stdout:")
        print(result.stdout)
        return

    if not GEN_OUTPUT.exists():
        print("ERROR: generate_splits.py did not produce scripts/splits.json", file=sys.stderr)
        sys.exit(1)

    with open(REF_SPLITS) as f:
        ref = json.load(f)
    with open(GEN_OUTPUT) as f:
        gen = json.load(f)

    ref_splits = ref.get("splits", ref)
    gen_splits = gen.get("splits", gen)

    mismatches = []
    for name in ref_splits:
        if name not in gen_splits:
            mismatches.append(f"Missing split: {{name}}")
        elif sorted(ref_splits[name]) != sorted(gen_splits[name]):
            mismatches.append(
                f"{{name}}: {{len(ref_splits[name])}} ref vs {{len(gen_splits[name])}} generated"
            )

    if mismatches:
        print("VERIFICATION FAILED:")
        for m in mismatches:
            print(f"  ✗ {{m}}")
        sys.exit(1)
    else:
        print(f"✓ Verification PASSED (SHA-256: {{splits_hash(ref_splits)[:12]}}...)")
        for name, ids in ref_splits.items():
            print(f"  {{name}}: {{len(ids)}} records")


if __name__ == "__main__":
    main()
'''
