"""
LangGraph graph definition.

Flow:
    START → scrape → [error?] → END
                   ↓ (ok)
                 extract → write → END
"""

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import scrape_node, extract_node, write_node


def _route_after_scrape(state: AgentState) -> str:
    """If scraping failed, skip extraction and bail out early."""
    if state.get("error"):
        return "end"
    return "extract"


def build_graph():
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("scrape", scrape_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("write", write_node)

    # Entry point
    workflow.add_edge(START, "scrape")

    # Conditional routing after scrape
    workflow.add_conditional_edges(
        "scrape",
        _route_after_scrape,
        {"extract": "extract", "end": END},
    )

    # Fixed edges for the happy path
    workflow.add_edge("extract", "write")
    workflow.add_edge("write", END)

    return workflow.compile()
