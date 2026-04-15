from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from cardiomas.verbose import vprint, vprint_llm

logger = logging.getLogger(__name__)


def load_skill(skill_name: str) -> str:
    """Load an agent skill .md file as a system prompt string."""
    skills_dir = Path(__file__).parent.parent / "skills"
    path = skills_dir / f"{skill_name}.md"
    if path.exists():
        return path.read_text()
    return f"You are an expert agent for {skill_name}. Be precise and cite sources."


def run_agent(
    llm: BaseChatModel,
    skill_name: str,
    user_message: str,
    extra_context: str = "",
) -> str:
    """Run an agent with a skill system prompt and return the text response."""
    system_prompt = load_skill(skill_name)
    if extra_context:
        system_prompt += f"\n\n## Context\n{extra_context}"

    full_prompt = f"{user_message}\n\n---\nContext:\n{extra_context[:600]}" if extra_context else user_message
    vprint(skill_name, f"calling LLM ({llm.__class__.__name__})…")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    try:
        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        vprint_llm(skill_name, full_prompt, text)
        return text
    except Exception as e:
        logger.error(f"Agent {skill_name} failed: {e}")
        vprint(skill_name, f"[red]ERROR: {e}[/red]")
        return f"ERROR: {e}"
