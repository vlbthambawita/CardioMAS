"""
Executor Agent (V4)

Runs generated scripts step-by-step. Captures all stdout, stderr, exit codes.
Reads generated files as text. Stores everything in state. Verifies correctness.

NO LLM CALLS — this agent is purely deterministic execution + output capture.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from cardiomas.schemas.state import (
    ExecutionResult,
    GraphState,
    LogEntry,
    RefinementContext,
)
from cardiomas.tools.script_verification import verify_script_output
from cardiomas.tools.v4_output_tools import collect_generated_files
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def executor_agent(state: GraphState) -> GraphState:
    """Run generated scripts in sorted order. Capture all output.

    Phase-aware:
    - subset_validation: runs scripts 00-03
    - full_run: runs script 04_generate_splits_full.py
    - ecg_stats_run: runs scripts 10-14
    """
    state.execution_log.append(LogEntry(agent="executor", action="start"))
    phase = state.v4_pipeline_phase
    vprint("executor", f"running scripts (phase={phase})…")

    # ── Select scripts for this phase ──────────────────────────────────────
    scripts_to_run = _select_scripts(state, phase)
    if not scripts_to_run:
        state.errors.append(f"executor: no scripts found for phase '{phase}'")
        return state

    vprint("executor", f"  scripts: {list(scripts_to_run.keys())}")

    # ── Execute scripts in sorted order ────────────────────────────────────
    all_passed = True
    for script_name in sorted(scripts_to_run.keys()):
        record = scripts_to_run[script_name]
        vprint("executor", f"  running {script_name}…")

        exec_result = _run_script(script_name, record, state)
        state.v4_execution_results.append(exec_result)

        # Read generated files from the script's output dir
        generated = collect_generated_files(record.output_dir)
        exec_result.generated_files = generated
        # Merge all generated files into the state-level dict
        state.v4_generated_files.update(generated)

        # Write execution log
        _write_log(state, script_name, exec_result)

        vprint(
            "executor",
            f"    exit={exec_result.exit_code} "
            f"duration={exec_result.duration_seconds:.1f}s "
            f"files={list(generated.keys())}",
        )

        if not exec_result.verification_passed or exec_result.exit_code != 0:
            vprint(
                "executor",
                f"    [red]FAILED: {exec_result.verification_notes}[/red]",
            )
            all_passed = False

            # Set refinement context so data_engineer can fix the script
            attempt = state.v4_refinement_rounds.get(script_name, 0) + 1
            state.v4_refinement_context = RefinementContext(
                failed_script=script_name,
                error_message=exec_result.stderr[:1000] or exec_result.verification_notes,
                stdout_excerpt=exec_result.stdout[:2000],
                suggested_fix="",
                attempt=attempt,
            )
            state.errors.append(
                f"executor: script '{script_name}' failed: "
                f"{exec_result.stderr[:200] or exec_result.verification_notes}"
            )
            break  # stop execution on first failure

        vprint("executor", f"    [green]OK: {exec_result.verification_notes}[/green]")

    # ── Post-execution state updates ───────────────────────────────────────
    if all_passed:
        # Clear any prior refinement context on success
        state.v4_refinement_context = None

        if phase == "subset_validation":
            state.v4_subset_validated = True
            state.v4_execution_summary = _build_execution_summary(state)
            vprint("executor", "subset validation PASSED — all scripts succeeded")
            state.execution_log.append(LogEntry(
                agent="executor",
                action="subset_validated",
                detail=f"{len(scripts_to_run)} scripts passed",
            ))

        elif phase == "full_run":
            vprint("executor", "full run PASSED — splits generated")
            state.execution_log.append(LogEntry(
                agent="executor",
                action="full_run_complete",
                detail=f"splits.json generated",
            ))

        elif phase == "ecg_stats_run":
            vprint("executor", "ECG stats run PASSED — stat files generated")
            state.execution_log.append(LogEntry(
                agent="executor",
                action="ecg_stats_complete",
                detail=f"{len(scripts_to_run)} stat scripts passed",
            ))
    else:
        state.execution_log.append(LogEntry(
            agent="executor",
            action="failed",
            detail=str(state.v4_refinement_context.failed_script if state.v4_refinement_context else "unknown"),
        ))

    return state


# ── Helpers ────────────────────────────────────────────────────────────────

def _select_scripts(state: GraphState, phase: str) -> dict:
    """Return the subset of generated scripts for the current phase."""
    all_scripts = dict(state.v4_generated_scripts)
    # Also include ecg_stats scripts
    all_scripts.update(state.v4_ecg_stats_scripts)

    if phase == "subset_validation":
        # Scripts 00-03
        return {
            name: rec
            for name, rec in all_scripts.items()
            if name[:2] in ("00", "01", "02", "03")
        }
    elif phase == "full_run":
        # Script 04 only
        return {
            name: rec
            for name, rec in all_scripts.items()
            if name.startswith("04")
        }
    elif phase == "ecg_stats_run":
        # Scripts 10-14
        return {
            name: rec
            for name, rec in all_scripts.items()
            if name[:2] in ("10", "11", "12", "13", "14")
        }
    else:
        # Fallback: run everything
        return all_scripts


def _run_script(
    script_name: str,
    record: Any,
    state: GraphState,
) -> ExecutionResult:
    """Execute a single script and return an ExecutionResult."""
    import subprocess

    path = Path(record.path)
    if not path.exists():
        return ExecutionResult(
            script_name=script_name,
            exit_code=-1,
            stderr=f"Script file not found: {record.path}",
            verification_passed=False,
            verification_notes="Script file not found",
        )

    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=record.timeout,
            cwd=str(path.parent),
        )
        duration = time.monotonic() - start

        stdout = proc.stdout[:8000]
        stderr = proc.stderr[:2000]
        exit_code = proc.returncode

    except subprocess.TimeoutExpired:
        duration = record.timeout
        stdout = ""
        stderr = f"Script timed out after {record.timeout}s"
        exit_code = -1
    except Exception as exc:
        duration = time.monotonic() - start
        stdout = ""
        stderr = str(exc)
        exit_code = -1

    # Collect generated files early for verification
    generated = collect_generated_files(record.output_dir)

    # Verify output
    subset_size = getattr(state.user_options, "v4_subset_size", state.v4_subset_size)
    verification = verify_script_output(
        script_name=script_name,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        generated_files=generated,
        subset_size=subset_size,
    )

    return ExecutionResult(
        script_name=script_name,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=round(duration, 3),
        generated_files=generated,
        verification_passed=verification.passed,
        verification_notes=verification.notes or "; ".join(verification.issues),
    )


def _build_execution_summary(state: GraphState) -> str:
    """Build a text summary of all execution results for LLM context."""
    parts = ["## V4 Execution Summary\n"]

    for result in state.v4_execution_results:
        status = "PASS" if result.verification_passed else "FAIL"
        parts.append(
            f"### {result.script_name} [{status}]\n"
            f"Exit code: {result.exit_code}, "
            f"Duration: {result.duration_seconds:.1f}s\n"
            f"Output:\n{result.stdout[:2000]}\n"
        )
        if result.generated_files:
            parts.append(f"Generated files: {list(result.generated_files.keys())}\n")

    # Merge all generated files in summary
    if state.v4_generated_files:
        parts.append(f"\n## All Generated Files\n{list(state.v4_generated_files.keys())}\n")

    return "\n".join(parts)


def _write_log(state: GraphState, script_name: str, exec_result: ExecutionResult) -> None:
    """Write execution log JSON file."""
    try:
        from cardiomas.tools.v4_output_tools import write_execution_log
        log_dir = str(Path(state.v4_output_dir) / "logs")
        write_execution_log.invoke({
            "log_dir": log_dir,
            "script_name": script_name,
            "execution_result": exec_result.model_dump(mode="json"),
        })
    except Exception as e:
        logger.debug(f"executor: log write failed for {script_name}: {e}")
