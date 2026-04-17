"""Upfront planning step for the ReAct++ agent loop.

Before the first ReAct iteration the planner makes a single LLM call and
returns an ordered list of tool names (2-5 steps). This plan is injected as
the first observation hint so the agent starts with a global strategy rather
than reasoning from scratch each iteration.

The plan is a *soft guide* — the agent can deviate when observations change.
"""
from __future__ import annotations

import json
from collections.abc import Generator

from cardiomas.inference.base import ChatClient, ChatRequest
from cardiomas.inference.prompts import react_planner_messages
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.runtime import AgentEvent
from cardiomas.tools.registry import ToolRegistry


def generate_plan(
    query: str,
    config: RuntimeConfig,
    registry: ToolRegistry,
    chat_client: ChatClient,
) -> Generator[AgentEvent, None, list[str]]:
    """One LLM call that returns an ordered list of tool names to try.

    Yields status events and returns the plan as a list of tool name strings.
    Falls back to an empty list on any failure so the loop still runs.
    """
    assert config.llm is not None
    model = config.llm.resolved_router_model
    registered = {spec.name for spec in registry.specs()}

    messages = react_planner_messages(query, registry.specs())
    try:
        request = ChatRequest(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=config.llm.router_max_tokens,
            json_mode=True,
            keep_alive=config.llm.keep_alive,
        )
        response = chat_client.chat(request)
        data = json.loads(response.content)
        raw_plan = data.get("plan", [])
        reasoning = str(data.get("reasoning", ""))
        if isinstance(raw_plan, list):
            plan = [t for t in raw_plan if isinstance(t, str) and t in registered][:5]
            if plan:
                yield AgentEvent(
                    type="status", stage="react",
                    message=f"Upfront plan ({len(plan)} steps): {' → '.join(plan)}. {reasoning}",
                    data={"plan": plan, "reasoning": reasoning},
                )
                return plan
    except Exception as exc:
        yield AgentEvent(
            type="status", stage="react",
            message=f"Planner LLM failed, proceeding without upfront plan: {exc}",
        )
    return []
