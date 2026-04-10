"""
LangGraph graph definition.

Flow:
    START → scrape → [error?] → END
                   ↓ (ok)
                 extract → ecg_expert → write → END
"""

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import scrape_node, extract_node, ecg_expert_node, write_node


def _route_after_scrape(state: AgentState) -> str:
    """If scraping failed, skip all downstream nodes and bail out early."""
    if state.get("error"):
        return "end"
    return "extract"


def build_graph():
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("scrape", scrape_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("ecg_expert", ecg_expert_node)
    workflow.add_node("write", write_node)

    # Entry point
    workflow.add_edge(START, "scrape")

    # Conditional routing after scrape
    workflow.add_conditional_edges(
        "scrape",
        _route_after_scrape,
        {"extract": "extract", "end": END},
    )

    # Happy path: extract → ecg_expert → write
    workflow.add_edge("extract", "ecg_expert")
    workflow.add_edge("ecg_expert", "write")
    workflow.add_edge("write", END)

    return workflow.compile()
