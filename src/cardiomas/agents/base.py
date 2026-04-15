from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

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
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    try:
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"Agent {skill_name} failed: {e}")
        return f"ERROR: {e}"
