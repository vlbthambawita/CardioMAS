from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from cardiomas.schemas.state import GraphState, UserOptions
from cardiomas.agents.orchestrator import orchestrator_agent
from cardiomas.agents.discovery import discovery_agent
from cardiomas.agents.paper import paper_agent
from cardiomas.agents.analysis import analysis_agent
from cardiomas.agents.splitter import splitter_agent
from cardiomas.agents.security import security_agent
from cardiomas.agents.publisher import publisher_agent

logger = logging.getLogger(__name__)


def _route_after_orchestrator(state: GraphState) -> str:
    if state.existing_hf_splits and not state.user_options.force_reanalysis:
        return "return_existing"
    return "discovery"


def _route_after_security(state: GraphState) -> str:
    if state.security_audit and not state.security_audit.passed:
        return "end_with_error"
    if state.user_options.push_to_hf:
        return "publisher"
    return "end_saved"  # default: saved locally, no HF push


def _passthrough_return_existing(state: GraphState) -> GraphState:
    """Short-circuit when existing splits found on HF."""
    from cardiomas.schemas.state import LogEntry
    state.execution_log.append(LogEntry(agent="orchestrator", action="return_existing"))
    return state


def _passthrough_end_error(state: GraphState) -> GraphState:
    """End with security error."""
    from cardiomas.schemas.state import LogEntry
    state.execution_log.append(LogEntry(agent="orchestrator", action="blocked_by_security"))
    return state


def _passthrough_end_saved(state: GraphState) -> GraphState:
    """Default end: outputs saved locally, no HF push requested."""
    from cardiomas.schemas.state import LogEntry
    state.publish_status = "saved_locally"
    state.execution_log.append(LogEntry(
        agent="orchestrator", action="saved_locally",
        detail=state.local_output_dir,
    ))
    return state


def _state_to_dict(state: GraphState) -> dict:
    return state.model_dump()


def _dict_to_state(d: dict) -> GraphState:
    return GraphState(**d)


def build_workflow():
    """Build and compile the LangGraph StateGraph."""
    # LangGraph needs dict-based state; wrap our Pydantic model
    graph = StateGraph(dict)

    def wrap(fn):
        def _wrapper(state_dict: dict) -> dict:
            state = _dict_to_state(state_dict)
            result = fn(state)
            return _state_to_dict(result)
        _wrapper.__name__ = fn.__name__
        return _wrapper

    def route_after_orch(state_dict: dict) -> str:
        return _route_after_orchestrator(_dict_to_state(state_dict))

    def route_after_sec(state_dict: dict) -> str:
        return _route_after_security(_dict_to_state(state_dict))

    graph.add_node("orchestrator", wrap(orchestrator_agent))
    graph.add_node("return_existing", wrap(_passthrough_return_existing))
    graph.add_node("discovery", wrap(discovery_agent))
    graph.add_node("paper", wrap(paper_agent))
    graph.add_node("analysis", wrap(analysis_agent))
    graph.add_node("splitter", wrap(splitter_agent))
    graph.add_node("security", wrap(security_agent))
    graph.add_node("publisher", wrap(publisher_agent))
    graph.add_node("end_with_error", wrap(_passthrough_end_error))
    graph.add_node("end_saved", wrap(_passthrough_end_saved))

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orch,
        {"return_existing": "return_existing", "discovery": "discovery"},
    )
    graph.add_edge("return_existing", END)
    graph.add_edge("discovery", "paper")
    graph.add_edge("paper", "analysis")
    graph.add_edge("analysis", "splitter")
    graph.add_edge("splitter", "security")
    graph.add_conditional_edges(
        "security",
        route_after_sec,
        {"publisher": "publisher", "end_with_error": "end_with_error", "end_saved": "end_saved"},
    )
    graph.add_edge("publisher", END)
    graph.add_edge("end_with_error", END)
    graph.add_edge("end_saved", END)

    return graph.compile()


def run_pipeline(
    dataset_source: str,
    options: UserOptions | None = None,
) -> GraphState:
    """Run the full CardioMAS pipeline and return the final GraphState."""
    if options is None:
        options = UserOptions(dataset_source=dataset_source)

    initial = GraphState(dataset_source=dataset_source, user_options=options)
    workflow = build_workflow()
    final_dict = workflow.invoke(_state_to_dict(initial))
    return _dict_to_state(final_dict)
