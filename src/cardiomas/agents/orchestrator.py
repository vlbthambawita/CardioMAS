"""
Dynamic Orchestrator Agent (V2)

Runs at pipeline entry and after every worker agent as a hub-and-spoke supervisor.
Sets state.next_agent to control routing. Records reasoning for full transparency.
"""
from __future__ import annotations

import logging
import uuid

from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def orchestrator_agent(state: GraphState) -> GraphState:
    """Entry point and supervisor. Decides next agent based on current pipeline state."""
    from cardiomas import config as cfg
    from cardiomas.recorder import SessionRecorder

    last = state.last_completed_agent

    # ── Initial call ──────────────────────────────────────────────────────
    if last == "":
        return _initial_routing(state, cfg, SessionRecorder)

    # ── Supervisor call (post-agent) ──────────────────────────────────────
    vprint("orchestrator", f"supervisor: [{last}] completed")

    # Handle errors: retry once, then abort
    if state.errors:
        last_error = state.errors[-1]
        retry_count = state.retry_counts.get(last, 0)
        if retry_count < 1:
            state.retry_counts[last] = retry_count + 1
            vprint("orchestrator", f"[yellow]error in {last} — retrying ({retry_count + 1}/1)[/yellow]")
            reason = f"Error in {last}: {last_error[:120]}. Retrying once."
            _record(state, last, reason, SessionRecorder)
            # Clear the error before retry
            state.errors = state.errors[:-1]
            state.next_agent = last
            return state
        else:
            vprint("orchestrator", f"[red]{last} failed after 1 retry — aborting pipeline[/red]")
            reason = f"{last} failed after 1 retry: {last_error[:120]}. Aborting."
            _record(state, "end_with_error", reason, SessionRecorder)
            state.next_agent = "end_with_error"
            return state

    # Normal routing
    next_a, reason = _route(state, last)
    _record(state, next_a, reason, SessionRecorder)
    state.next_agent = next_a
    return state


# ── Initial routing ────────────────────────────────────────────────────────

def _initial_routing(state: GraphState, cfg, SessionRecorder) -> GraphState:
    # Init session
    if not state.session_id:
        state.session_id = str(uuid.uuid4())
        rec = SessionRecorder.reset()
        dataset_name = _dataset_name(state.dataset_source)
        rec.start_session(
            dataset_name=dataset_name,
            user_options=state.user_options.model_dump(),
            raw_requirement=state.user_options.requirement,
        )

    state.execution_log.append(LogEntry(
        agent="orchestrator", action="start", detail=state.dataset_source
    ))
    vprint("orchestrator", f"pipeline start — source: {state.dataset_source}")
    vprint("orchestrator", f"session: {state.session_id}")

    # HF cache check
    if not state.user_options.force_reanalysis:
        dataset_name = _dataset_name(state.dataset_source)
        vprint("orchestrator", f"checking HF cache for '{dataset_name}'…")
        try:
            from cardiomas.tools.publishing_tools import check_hf_repo
            result = check_hf_repo.invoke({"repo_id": cfg.HF_REPO_ID, "dataset_name": dataset_name})
            if result.get("exists"):
                state.existing_hf_splits = result.get("metadata")
                state.publish_status = "already_published"
                state.execution_log.append(
                    LogEntry(agent="orchestrator", action="cache_hit", detail=dataset_name)
                )
                vprint("orchestrator", f"cache hit — '{dataset_name}' already on HF")
                _record(state, "return_existing",
                        f"HF cache hit for '{dataset_name}'. Returning existing splits.",
                        SessionRecorder)
                state.next_agent = "return_existing"
                return state
            else:
                vprint("orchestrator", "no cache hit — running full pipeline")
        except Exception as e:
            logger.warning(f"HF cache check failed: {e} — continuing")
    else:
        vprint("orchestrator", "--force-reanalysis — skipping cache check")

    # Choose entry agent
    if state.user_options.requirement:
        next_a = "nl_requirement"
        reason = "Natural language requirement provided — parsing before discovery."
    else:
        next_a = "discovery"
        reason = "No NL requirement — starting with discovery agent."

    _record(state, next_a, reason, SessionRecorder)
    state.next_agent = next_a
    return state


# ── Routing logic ──────────────────────────────────────────────────────────

def _route(state: GraphState, last: str) -> tuple[str, str]:
    """Pure logic: determine next agent given last completed agent and current state."""
    opts = state.user_options

    if last == "nl_requirement":
        return "discovery", "NL requirement parsed — proceeding to discovery."

    if last == "discovery":
        # Local path with no URL → skip paper (no paper to find for local datasets)
        has_local = bool(opts.local_path) and not _is_url(opts.dataset_source)
        if has_local:
            # V4 mode: go to data_engineer instead of analysis
            if _has_local_path(state):
                return "data_engineer", "Local-only path — generating exploration scripts (V4)."
            return "analysis", "Local-only path — skipping paper agent, going to analysis."
        return "paper", "Dataset identified — searching for associated publication."

    if last == "paper":
        # V4 mode: go to data_engineer instead of analysis
        return "data_engineer", "Paper analysis done — generating dataset exploration scripts (V4)."

    if last == "data_engineer":
        return "executor", "Scripts generated — executing on subset for validation (V4)."

    if last == "executor":
        # Phase-based routing from executor
        phase = state.v4_pipeline_phase
        refinement = state.v4_refinement_context

        if refinement and state.errors:
            # Script failed — check if we should refine
            failed_script = refinement.failed_script
            retry_key = f"executor_refinement_{failed_script}"
            count = state.retry_counts.get(retry_key, 0)
            max_refinements = getattr(opts, "v4_max_refinements", 2)
            if count < max_refinements:
                state.retry_counts[retry_key] = count + 1
                # Clear the error to allow data_engineer to retry
                state.errors = [e for e in state.errors if "executor:" not in e]
                return "data_engineer", (
                    f"Script '{failed_script}' failed (attempt {count + 1}/{max_refinements}) "
                    "— sending back to data_engineer for refinement."
                )
            return "end_with_error", (
                f"Script '{failed_script}' failed after {max_refinements} refinement attempts."
            )

        if phase == "subset_validation" and state.v4_subset_validated:
            return "analysis", "Subset validation passed — proceeding to analysis (V4)."

        if phase == "full_run":
            return "analysis", "Full run complete — proceeding to analysis (V4)."

        if phase == "ecg_stats_run":
            return "splitter", "ECG stats complete — generating reproducible splits (V4)."

        return "end_with_error", f"Unexpected executor state: phase={phase}"

    if last == "analysis":
        # V4: after analysis, check phase
        phase = state.v4_pipeline_phase
        if phase == "subset_validation" and state.v4_subset_validated:
            return "approval_gate", "Subset validated — requesting approval before full run (V4)."
        if phase == "full_run":
            skip_ecg = getattr(opts, "v4_skip_ecg_stats", False)
            if skip_ecg:
                return "splitter", "Full run analysis complete — skipping ECG stats (V4)."
            return "ecg_stats", "Full run analysis complete — generating ECG statistical scripts (V4)."
        # V3 fallback (no V4 phases set)
        return "splitter", "Analysis complete — generating reproducible splits."

    if last == "ecg_stats":
        return "executor", "ECG stat scripts generated — executing full-dataset stats (V4)."

    if last == "splitter":
        return "security", "Splits generated — running security audit."

    if last == "security":
        if state.security_audit and not state.security_audit.passed:
            return "end_with_error", (
                "Security audit FAILED. Blocking issues detected — "
                "not saving output or pushing to HuggingFace."
            )
        return "coder", "Security audit passed — generating reproducibility scripts."

    if last == "coder":
        if opts.push_to_hf:
            return "publisher", "Scripts generated — publishing to HuggingFace as requested."
        return "end_saved", "Scripts generated — saving locally (no --push flag)."

    if last == "publisher":
        return "end_saved", "Published to HuggingFace — pipeline complete."

    return "end_saved", "Pipeline complete."


def _has_local_path(state: GraphState) -> bool:
    """Return True if a local dataset path is set."""
    opts = state.user_options
    return bool(
        opts.local_path
        or (state.dataset_info and state.dataset_info.local_path)
        or (not _is_url(opts.dataset_source))
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _record(state: GraphState, next_agent: str, reason: str, SessionRecorder) -> None:
    last = state.last_completed_agent or "start"
    entry = f"[{last}] → [{next_agent}]: {reason}"
    state.orchestrator_reasoning.append(entry)
    state.execution_log.append(LogEntry(
        agent="orchestrator",
        action=f"route→{next_agent}",
        detail=reason[:120],
    ))
    vprint("orchestrator", f"  → {next_agent}: {reason[:90]}")
    try:
        SessionRecorder.get().add_orchestrator_reasoning(entry)
    except Exception:
        pass


def _dataset_name(source: str) -> str:
    return source.rstrip("/").split("/")[-1].lower()


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))
