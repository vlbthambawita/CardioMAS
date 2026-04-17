"""Short-term distilled key-facts scratchpad for the ReAct++ agent.

The raw ``observations`` list grows to include full tool output (folder trees,
WFDB metadata reports, website text) which can reach thousands of tokens.
The scratchpad stores one condensed sentence per successful tool call and is
shown to the LLM at each iteration as "Running knowledge", letting the model
reason about accumulated facts without re-reading verbose output.
"""
from __future__ import annotations


class Scratchpad:
    """Lightweight in-memory store of distilled tool findings."""

    def __init__(self) -> None:
        self._facts: list[tuple[str, str]] = []  # (tool_name, condensed_fact)

    def add(self, tool: str, raw_summary: str) -> None:
        """Add the first sentence / key line from a tool's summary."""
        # Take the first non-empty line as the key fact
        for line in raw_summary.splitlines():
            condensed = " ".join(line.split())[:180]
            if condensed:
                self._facts.append((tool, condensed))
                return

    def to_string(self) -> str:
        """Return a compact block suitable for injection into an LLM prompt."""
        if not self._facts:
            return ""
        lines = ["Running knowledge (distilled from tool results so far):"]
        for tool, fact in self._facts:
            lines.append(f"  [{tool}] {fact}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return not self._facts

    def __len__(self) -> int:
        return len(self._facts)
