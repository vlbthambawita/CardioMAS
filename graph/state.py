from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    """Shared state that flows through all graph nodes."""

    # Input
    url: str

    # Populated by ScrapeNode
    title: str
    raw_text: str
    links: List[dict]

    # Populated by ExtractNode
    summary: str
    sections: str
    key_facts: str

    # Populated by WriteNode
    output_path: str

    # Set by any node on failure — triggers early exit
    error: Optional[str]
