"""
Approval Gate (V4) — Human-in-the-loop checkpoint.

After subset validation, this node checks whether the user has approved
proceeding to the full dataset run. If auto-approve is set, it bypasses
the interactive step.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from cardiomas.schemas.state import ApprovalSummary, GraphState, LogEntry
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def approval_gate_node(state: GraphState) -> GraphState:
    """Human-in-the-loop gate. Checks state.v4_approval_status.

    Routing outcomes (via state.next_agent):
    - "approved"  → set phase=full_run, next_agent=executor
    - "rejected"  → next_agent=end_saved
    - "pending" + v4_auto_approve → auto-approve, same as "approved"
    - "pending" (interactive) → print summary, save checkpoint, next_agent=end_saved
    """
    state.execution_log.append(LogEntry(agent="approval_gate", action="start"))
    status = state.v4_approval_status
    auto = getattr(state.user_options, "v4_auto_approve", False)

    vprint("approval_gate", f"approval_status={status} auto_approve={auto}")

    if status == "approved" or (status == "pending" and auto):
        if status == "pending":
            vprint("approval_gate", "auto-approve enabled — proceeding to full run")
            state.v4_approval_status = "approved"

        state.v4_pipeline_phase = "full_run"
        state.execution_log.append(LogEntry(
            agent="approval_gate",
            action="approved",
            detail="Approved — transitioning to full_run phase.",
        ))
        state.next_agent = "executor"
        vprint("approval_gate", "approved — routing to executor (full run)")

    elif status == "rejected":
        state.execution_log.append(LogEntry(
            agent="approval_gate",
            action="rejected",
            detail="Rejected — saving subset results only.",
        ))
        state.next_agent = "end_saved"
        vprint("approval_gate", "rejected — saving subset results only")

    else:
        # "pending" without auto-approve — show summary and stop
        summary = _build_approval_summary(state)
        state.v4_approval_summary = summary
        _print_approval_request(state, summary)
        state.next_agent = "end_saved"
        state.execution_log.append(LogEntry(
            agent="approval_gate",
            action="pending",
            detail="Awaiting human approval — saved checkpoint for re-run with --approve.",
        ))
        vprint("approval_gate", "pending — saved checkpoint; re-run with --approve to continue")

    return state


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_approval_summary(state: GraphState) -> ApprovalSummary:
    """Build an ApprovalSummary from execution results."""
    info = state.dataset_info
    dataset_name = (info.name if info else None) or "unknown"
    subset_size = getattr(state.user_options, "v4_subset_size", state.v4_subset_size)

    # Extract records found from explore output
    records_found = 0
    for result in state.v4_execution_results:
        if result.script_name.startswith("00"):
            m = re.search(r"TOTAL_FILES\s*=\s*(\d+)", result.stdout)
            if m:
                records_found = int(m.group(1))
                break

    # Extract column names from metadata output
    columns_found: list[str] = []
    for result in state.v4_execution_results:
        if result.script_name.startswith("01"):
            m = re.search(r"COLUMNS\s*=\s*(\[.*?\])", result.stdout, re.DOTALL)
            if m:
                try:
                    columns_found = json.loads(m.group(1))
                except Exception:
                    pass
            break

    # Extract label distribution from stats output
    label_excerpt = ""
    stats_csv = state.v4_generated_files.get("stats.csv", "")
    if stats_csv:
        label_excerpt = stats_csv[:500]

    # Extract split sizes from splits_subset.json
    split_sizes: dict[str, int] = {}
    splits_json = state.v4_generated_files.get("splits_subset.json", "")
    if splits_json:
        try:
            data = json.loads(splits_json)
            splits = data.get("splits", data)
            split_sizes = {k: len(v) for k, v in splits.items() if isinstance(v, list)}
        except Exception:
            pass

    # Track passed/failed scripts
    scripts_passed = [r.script_name for r in state.v4_execution_results if r.verification_passed]
    scripts_failed = [r.script_name for r in state.v4_execution_results if not r.verification_passed]

    return ApprovalSummary(
        dataset_name=dataset_name,
        subset_size=subset_size,
        records_found=records_found,
        columns_found=columns_found,
        label_distribution_excerpt=label_excerpt,
        split_sizes=split_sizes,
        scripts_passed=scripts_passed,
        scripts_failed=scripts_failed,
        output_dir=state.v4_output_dir,
    )


def _print_approval_request(state: GraphState, summary: ApprovalSummary) -> None:
    """Pretty-print the subset validation results and approval prompt."""
    try:
        from rich.console import Console
        from rich.rule import Rule
        from rich.table import Table
        console = Console()
    except ImportError:
        _print_plain_approval_request(summary)
        return

    console.print()
    console.print(Rule("[bold yellow]CardioMAS V4 — Subset Validation Complete[/bold yellow]"))
    console.print()
    console.print(f"[bold]Dataset:[/bold] {summary.dataset_name}")
    console.print(f"[bold]Subset size:[/bold] {summary.subset_size}")
    console.print(f"[bold]Files found:[/bold] {summary.records_found}")
    console.print()

    if summary.columns_found:
        console.print(f"[bold]Columns detected ({len(summary.columns_found)}):[/bold]")
        console.print(f"  {summary.columns_found[:15]}")
        console.print()

    if summary.split_sizes:
        table = Table(title="Subset Splits")
        table.add_column("Split", style="cyan")
        table.add_column("Records", justify="right")
        for split_name, count in summary.split_sizes.items():
            table.add_row(split_name, str(count))
        console.print(table)
        console.print()

    if summary.label_distribution_excerpt:
        console.print("[bold]Label distribution (stats.csv excerpt):[/bold]")
        console.print(f"  {summary.label_distribution_excerpt[:300]}")
        console.print()

    console.print(f"[bold]Scripts passed:[/bold] {summary.scripts_passed}")
    if summary.scripts_failed:
        console.print(f"[bold red]Scripts failed:[/bold red] {summary.scripts_failed}")
    console.print()
    console.print(f"[dim]Output directory: {summary.output_dir}[/dim]")
    console.print()
    console.print("[yellow]Subset validation complete.[/yellow]")
    console.print(
        "Review the results above, then re-run with [bold cyan]--approve[/bold cyan] "
        "to proceed to full dataset processing."
    )
    console.print(
        "Or use [bold cyan]--auto-approve[/bold cyan] to skip this gate entirely."
    )
    console.print()


def _print_plain_approval_request(summary: ApprovalSummary) -> None:
    """Fallback plain-text approval request when Rich is not available."""
    print()
    print("=" * 60)
    print("CardioMAS V4 — Subset Validation Complete")
    print("=" * 60)
    print(f"Dataset:      {summary.dataset_name}")
    print(f"Subset size:  {summary.subset_size}")
    print(f"Files found:  {summary.records_found}")
    if summary.columns_found:
        print(f"Columns:      {summary.columns_found[:10]}")
    if summary.split_sizes:
        print(f"Splits:       {summary.split_sizes}")
    print(f"Scripts OK:   {summary.scripts_passed}")
    if summary.scripts_failed:
        print(f"Scripts FAIL: {summary.scripts_failed}")
    print(f"Output dir:   {summary.output_dir}")
    print()
    print("Re-run with --approve to proceed to full dataset processing.")
    print()
