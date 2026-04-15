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
            return "analysis", "Local-only path — skipping paper agent, going to analysis."
        return "paper", "Dataset identified — searching for associated publication."

    if last == "paper":
        return "analysis", "Paper analysis complete — proceeding to data analysis."

    if last == "analysis":
        return "splitter", "Analysis complete — generating reproducible splits."

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
