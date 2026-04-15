"""
LangGraph workflow — V2 hub-and-spoke pattern.

Every worker agent routes back to the orchestrator after completing.
The orchestrator sets state.next_agent, which drives the routing.
Checkpoints are saved after every node.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from langgraph.graph import StateGraph, END

from cardiomas.schemas.state import GraphState, UserOptions
from cardiomas.agents.orchestrator import orchestrator_agent
from cardiomas.agents.nl_requirement import nl_requirement_agent
from cardiomas.agents.discovery import discovery_agent
from cardiomas.agents.paper import paper_agent
from cardiomas.agents.analysis import analysis_agent
from cardiomas.agents.splitter import splitter_agent
from cardiomas.agents.security import security_agent
from cardiomas.agents.coder import coder_agent
from cardiomas.agents.publisher import publisher_agent

logger = logging.getLogger(__name__)

# All non-terminal agents that loop back to orchestrator
_WORKER_AGENTS = [
    "nl_requirement", "discovery", "paper",
    "analysis", "splitter", "security",
    "coder", "publisher",
]

# All valid routing targets from orchestrator
_ALL_TARGETS = _WORKER_AGENTS + ["return_existing", "end_saved", "end_with_error"]


# ── State serialisation ───────────────────────────────────────────────────

def _state_to_dict(state: GraphState) -> dict:
    return state.model_dump(mode="json")


def _dict_to_state(d: dict) -> GraphState:
    return GraphState(**d)


# ── Passthrough terminal nodes ────────────────────────────────────────────

def _passthrough_return_existing(state: GraphState) -> GraphState:
    from cardiomas.schemas.state import LogEntry
    state.execution_log.append(LogEntry(agent="orchestrator", action="return_existing"))
    return state


def _passthrough_end_error(state: GraphState) -> GraphState:
    from cardiomas.schemas.state import LogEntry
    state.execution_log.append(LogEntry(agent="orchestrator", action="blocked_by_security"))
    _save_session_log(state, "error")
    return state


def _passthrough_end_saved(state: GraphState) -> GraphState:
    from cardiomas.schemas.state import LogEntry
    state.publish_status = "saved_locally"
    state.execution_log.append(LogEntry(
        agent="orchestrator",
        action="saved_locally",
        detail=state.local_output_dir,
    ))
    _save_session_log(state, "ok")
    return state


# ── Session log persistence ───────────────────────────────────────────────

def _save_session_log(state: GraphState, status: str) -> None:
    from cardiomas.recorder import SessionRecorder
    recorder = SessionRecorder.get()
    recorder.finish_session(status=status)
    if state.local_output_dir:
        try:
            recorder.save(Path(state.local_output_dir))
        except Exception as e:
            logger.warning(f"Failed to save session log: {e}")


# ── Checkpoint ────────────────────────────────────────────────────────────

def _save_checkpoint(state_dict: dict) -> None:
    path = state_dict.get("checkpoint_path", "")
    if not path:
        return
    try:
        Path(path).write_text(json.dumps(state_dict, default=str))
    except Exception as e:
        logger.debug(f"Checkpoint save failed: {e}")


# ── Agent wrappers ────────────────────────────────────────────────────────

def _make_worker_wrapper(agent_name: str, fn):
    """Wrap a worker agent: set last_completed_agent and save checkpoint."""
    def _wrapper(state_dict: dict) -> dict:
        state = _dict_to_state(state_dict)
        result = fn(state)
        result.last_completed_agent = agent_name
        result_dict = _state_to_dict(result)
        _save_checkpoint(result_dict)
        return result_dict
    _wrapper.__name__ = fn.__name__
    return _wrapper


def _wrap_simple(fn):
    """Wrap a passthrough/orchestrator node."""
    def _wrapper(state_dict: dict) -> dict:
        state = _dict_to_state(state_dict)
        result = fn(state)
        result_dict = _state_to_dict(result)
        _save_checkpoint(result_dict)
        return result_dict
    _wrapper.__name__ = fn.__name__
    return _wrapper


# ── Graph construction ────────────────────────────────────────────────────

def build_workflow():
    """Build and compile the V2 hub-and-spoke LangGraph StateGraph."""
    graph = StateGraph(dict)

    def _route_from_orchestrator(state_dict: dict) -> str:
        state = _dict_to_state(state_dict)
        target = state.next_agent
        if target not in _ALL_TARGETS:
            logger.warning(f"Unknown next_agent '{target}' — defaulting to end_saved")
            return "end_saved"
        return target

    # ── Nodes ─────────────────────────────────────────────────────────────
    graph.add_node("orchestrator", _wrap_simple(orchestrator_agent))
    graph.add_node("nl_requirement", _make_worker_wrapper("nl_requirement", nl_requirement_agent))
    graph.add_node("discovery",      _make_worker_wrapper("discovery",      discovery_agent))
    graph.add_node("paper",          _make_worker_wrapper("paper",          paper_agent))
    graph.add_node("analysis",       _make_worker_wrapper("analysis",       analysis_agent))
    graph.add_node("splitter",       _make_worker_wrapper("splitter",       splitter_agent))
    graph.add_node("security",       _make_worker_wrapper("security",       security_agent))
    graph.add_node("coder",          _make_worker_wrapper("coder",          coder_agent))
    graph.add_node("publisher",      _make_worker_wrapper("publisher",      publisher_agent))
    graph.add_node("return_existing",_wrap_simple(_passthrough_return_existing))
    graph.add_node("end_saved",      _wrap_simple(_passthrough_end_saved))
    graph.add_node("end_with_error", _wrap_simple(_passthrough_end_error))

    # ── Entry ─────────────────────────────────────────────────────────────
    graph.set_entry_point("orchestrator")

    # ── Orchestrator → any agent (hub) ────────────────────────────────────
    graph.add_conditional_edges(
        "orchestrator",
        _route_from_orchestrator,
        {t: t for t in _ALL_TARGETS},
    )

    # ── Worker agents → orchestrator (spokes) ─────────────────────────────
    for agent in _WORKER_AGENTS:
        graph.add_edge(agent, "orchestrator")

    # ── Terminal nodes → END ─────────────────────────────────────────────
    graph.add_edge("return_existing", END)
    graph.add_edge("end_saved", END)
    graph.add_edge("end_with_error", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────

def run_pipeline(
    dataset_source: str,
    options: UserOptions | None = None,
) -> GraphState:
    """Run the full CardioMAS pipeline and return the final GraphState."""
    if options is None:
        options = UserOptions(dataset_source=dataset_source)

    # Ensure checkpoint directory exists
    dataset_name = dataset_source.rstrip("/").split("/")[-1].lower()
    checkpoint_dir = Path(options.output_dir) / dataset_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoint_dir / "session_checkpoint.json")

    initial = GraphState(
        dataset_source=dataset_source,
        user_options=options,
        checkpoint_path=checkpoint_path,
    )
    workflow = build_workflow()
    final_dict = workflow.invoke(_state_to_dict(initial))
    return _dict_to_state(final_dict)


def resume_pipeline(checkpoint_file: str) -> GraphState:
    """Resume a pipeline from a saved checkpoint.

    Picks up from the last completed agent.
    """
    data = json.loads(Path(checkpoint_file).read_text())
    last_agent = data.get("last_completed_agent", "")
    if not last_agent:
        raise ValueError(
            f"Checkpoint '{checkpoint_file}' has no last_completed_agent — "
            "cannot determine resume point. Run cardiomas analyze from scratch."
        )
    logger.info(f"Resuming pipeline from after '{last_agent}'")
    state = GraphState(**data)
    workflow = build_workflow()
    final_dict = workflow.invoke(_state_to_dict(state))
    return _dict_to_state(final_dict)
