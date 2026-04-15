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

    # ── generate_splits.py ────────────────────────────────────────────────
    gen_msg = f"""Write a self-contained Python script that reproduces the exact splits for the `{dataset_name}` ECG dataset.

Script file: generate_splits.py

Requirements:
1. ONLY stdlib + pandas + numpy. NO cardiomas or other external imports beyond these.
2. ALL parameters as ALL_CAPS constants at the top:
   DATA_PATH = "{local_path}"
   ID_FIELD = "{id_field}"
   SEED = {seed}
   STRATEGY = "{strategy}"
   SPLIT_RATIOS = {split_ratios}
3. Header comment with: dataset={dataset_name}, cardiomas={__version__}, date={today}, seed={seed}, strategy={strategy}, requirement="{requirement or 'none'}"
4. Load record IDs by scanning DATA_PATH for CSV/TSV files and extracting the ID_FIELD column. If not found, list all files and use filenames as IDs.
5. Sort IDs. Compute SHA-256(sorted_ids_string + str(SEED) + STRATEGY) as hex. Convert first 8 hex chars to int for numpy seed.
6. np.random.default_rng(numpy_seed).shuffle(sorted_ids). Then slice by SPLIT_RATIOS.
7. Save output as splits.json in the SAME DIRECTORY as the script.
8. Print a SHA-256 of the output dict (sorted keys, sorted IDs) to stdout as: SPLITS_SHA256=<hash>
9. Print a summary table showing split name, count, and ratio.

Return ONLY valid Python code. No markdown fences, no explanation."""

    gen_response = run_agent(llm, "coder", gen_msg, extra_context=context)
    gen_code = _extract_code(gen_response)

    gen_path = scripts_dir / "generate_splits.py"
    gen_path.write_text(gen_code)
    state.generated_scripts["generate_splits.py"] = str(gen_path)
    vprint("coder", f"wrote generate_splits.py ({len(gen_code)} chars)")

    # ── verify_splits.py (templated, no LLM needed) ───────────────────────
    verify_code = _build_verify_script(dataset_name)
    verify_path = scripts_dir / "verify_splits.py"
    verify_path.write_text(verify_code)
    state.generated_scripts["verify_splits.py"] = str(verify_path)
    vprint("coder", f"wrote verify_splits.py")

    # ── explore_dataset.py ────────────────────────────────────────────────
    eda_msg = f"""Write a self-contained Python EDA script for the `{dataset_name}` ECG dataset.

Script file: explore_dataset.py

Requirements:
1. ONLY stdlib + pandas (+ matplotlib optional). NO cardiomas imports.
2. DATA_PATH = "{local_path}" as constant at top.
3. Print: total record count, unique patient count (if patient_id/ecg_id column exists), missing value summary per column, label distribution (top 10 values for any column named 'label', 'rhythm', 'scp_codes', 'diagnosis', 'class').
4. Try to save a bar chart as label_distribution.png using matplotlib if available; silently skip if not.
5. All output to stdout. Catch all file errors gracefully.

Return ONLY valid Python code. No markdown fences, no explanation."""

    eda_response = run_agent(llm, "coder", eda_msg, extra_context=context)
    eda_code = _extract_code(eda_response)
    eda_path = scripts_dir / "explore_dataset.py"
    eda_path.write_text(eda_code)
    state.generated_scripts["explore_dataset.py"] = str(eda_path)
    vprint("coder", f"wrote explore_dataset.py")

    # ── Execute generate_splits.py ────────────────────────────────────────
    vprint("coder", "executing generate_splits.py to verify…")
    exec_result = execute_script.invoke({
        "script_path": str(gen_path),
        "timeout": 120,
        "working_dir": str(scripts_dir),
    })
    state.script_execution_log.append(exec_result)

    # ── Phase 4: strict SHA-256 verification ─────────────────────────────
    ref_checksum = repro.dataset_checksum if repro else ""
    try:
        from cardiomas.agents.verification import ScriptVerificationError, verify_script_sha256
        verify_script_sha256(exec_result, ref_checksum)
        state.script_verified = True
        vprint("coder", "[green]generate_splits.py verified — SHA-256 matches manifest[/green]")
    except ScriptVerificationError as exc:
        vprint("coder", f"[yellow]verification failed (attempt 1): {exc}[/yellow]")
        logger.warning(f"coder: script verification failed — requesting LLM correction")
        # Send correction prompt and regenerate
        correction_ctx = (
            f"The generated generate_splits.py failed verification:\n{exc}\n\n"
            f"Original context:\n{context}"
        )
        corrected_response = run_agent(llm, "coder", gen_msg, extra_context=correction_ctx)
        corrected_code = _extract_code(corrected_response)
        gen_path.write_text(corrected_code)
        state.generated_scripts["generate_splits.py"] = str(gen_path)

        # Re-execute corrected script
        exec_result2 = execute_script.invoke({
            "script_path": str(gen_path),
            "timeout": 120,
            "working_dir": str(scripts_dir),
        })
        state.script_execution_log.append(exec_result2)

        try:
            verify_script_sha256(exec_result2, ref_checksum)
            state.script_verified = True
            vprint("coder", "[green]corrected script verified — SHA-256 matches[/green]")
        except ScriptVerificationError as exc2:
            state.script_verified = False
            state.errors.append(f"coder: script SHA-256 mismatch after retry: {exc2}")
            vprint("coder", f"[red]script verification failed after retry: {exc2}[/red]")
            logger.error(f"coder: script verification failed after retry: {exc2}")
    except ImportError:
        # Phase 4 not installed — fall back to basic check
        if exec_result.get("exit_code") == 0:
            state.script_verified = True
            vprint("coder", "[green]generate_splits.py ran successfully[/green]")
        else:
            stderr = exec_result.get("stderr", "")[:300]
            vprint("coder", f"[yellow]script exited {exec_result.get('exit_code')} — saved but unverified[/yellow]")
            vprint("coder", f"[dim]stderr: {stderr}[/dim]")
            logger.warning(f"generate_splits.py execution failed: {stderr}")

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


def _build_verify_script(dataset_name: str) -> str:
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
