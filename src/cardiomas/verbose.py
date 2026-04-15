"""
Global verbose output controller.

Usage in agents/tools:
    from cardiomas.verbose import vprint, vprint_llm

vprint() writes a line only when verbose mode is on.
vprint_llm() prints the LLM prompt and response in a styled panel.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

_console = Console(stderr=False)
_enabled: bool = False

AGENT_COLORS = {
    "orchestrator": "cyan",
    "discovery":    "blue",
    "paper":        "magenta",
    "analysis":     "yellow",
    "splitter":     "green",
    "security":     "red",
    "publisher":    "bright_green",
}


def enable(on: bool = True) -> None:
    global _enabled
    _enabled = on


def is_enabled() -> bool:
    return _enabled


def vprint(agent: str, message: str) -> None:
    if not _enabled:
        return
    color = AGENT_COLORS.get(agent, "white")
    _console.print(f"  [{color}][{agent}][/{color}] {message}")


def vprint_llm(agent: str, prompt: str, response: str) -> None:
    if not _enabled:
        return
    color = AGENT_COLORS.get(agent, "white")
    _console.print(Rule(f"[{color}]{agent} — LLM call[/{color}]"))
    _console.print(Panel(
        Text(prompt[:800] + ("…" if len(prompt) > 800 else ""), style="dim"),
        title="[dim]prompt[/dim]",
        border_style="dim",
    ))
    _console.print(Panel(
        Text(response[:1200] + ("…" if len(response) > 1200 else ""), style="bright_white"),
        title=f"[{color}]response[/{color}]",
        border_style=color,
    ))
