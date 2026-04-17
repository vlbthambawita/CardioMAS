from __future__ import annotations

import json
import re
from pathlib import Path

from cardiomas.inference.base import ChatClient, ChatRequest
from cardiomas.inference.prompts import planner_messages, prompt_preview
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.runtime import AgentDecision, LLMTrace, PlanStep
from cardiomas.tools.registry import ToolRegistry


def plan_query(
    query: str,
    config: RuntimeConfig,
    registry: ToolRegistry,
    chat_client: ChatClient | None = None,
) -> tuple[AgentDecision, list[LLMTrace], list[str]]:
    if config.planner_uses_ollama and chat_client is not None and config.llm is not None:
        return _plan_with_ollama(query, config, registry, chat_client)
    if config.planner_uses_ollama and chat_client is None:
        decision = _heuristic_plan(query, config, registry)
        return decision, [], ["Ollama planner was requested, but no chat client was available; using heuristic planner."]
    return _heuristic_plan(query, config, registry), [], []


def _plan_with_ollama(
    query: str,
    config: RuntimeConfig,
    registry: ToolRegistry,
    chat_client: ChatClient,
) -> tuple[AgentDecision, list[LLMTrace], list[str]]:
    assert config.llm is not None
    dataset_path = _first_dataset_path(config)
    urls = re.findall(r"https?://\S+", query)
    expression = _extract_expression(query)
    messages = planner_messages(query, registry.specs(), dataset_path, urls, expression)
    trace = LLMTrace(
        stage="planner",
        provider=config.llm.provider,
        model=config.llm.resolved_planner_model,
        prompt_preview=prompt_preview(messages),
    )
    fallback = _heuristic_plan(query, config, registry)

    try:
        response = chat_client.chat(
            ChatRequest(
                model=config.llm.resolved_planner_model,
                messages=messages,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                json_mode=True,
                keep_alive=config.llm.keep_alive,
            )
        )
        trace.response_preview = _trim(response.content)
        raw = json.loads(response.content)
        decision = AgentDecision.model_validate(raw)
        sanitized = _sanitize_decision(decision, query, config, registry, dataset_path, urls, expression)
        if not sanitized.steps:
            trace.ok = False
            trace.error = "Planner response did not yield executable steps."
            return fallback, [trace], ["Ollama planner returned no valid steps; using heuristic planner."]
        return sanitized, [trace], []
    except Exception as exc:
        trace.ok = False
        trace.error = str(exc)
        return fallback, [trace], [f"Ollama planner failed; using heuristic planner. {exc}"]


def _heuristic_plan(query: str, config: RuntimeConfig, registry: ToolRegistry) -> AgentDecision:
    lower = query.lower()
    steps: list[PlanStep] = []
    notes: list[str] = []
    available = {spec.name for spec in registry.specs()}

    urls = re.findall(r"https?://\S+", query)
    if urls and "fetch_webpage" in available:
        steps.append(PlanStep(tool_name="fetch_webpage", reason="Query includes a direct URL.", args={"url": urls[0]}))

    if _looks_like_math(query) and "calculate" in available:
        expression = _extract_expression(query)
        if expression:
            steps.append(PlanStep(tool_name="calculate", reason="Query contains an arithmetic expression.", args={"expression": expression}))

    dataset_path = _first_dataset_path(config)
    if dataset_path and "inspect_dataset" in available and _needs_dataset_inspection(lower):
        steps.append(PlanStep(tool_name="inspect_dataset", reason="Query asks about local dataset structure or metadata.", args={"dataset_path": dataset_path}))

    if "retrieve_corpus" in available:
        steps.append(PlanStep(tool_name="retrieve_corpus", reason="Ground the answer with corpus retrieval.", args={"query": query, "top_k": config.retrieval.top_k}))

    if not steps and "inspect_dataset" in available and dataset_path:
        steps.append(PlanStep(tool_name="inspect_dataset", reason="Fallback to dataset inspection because no other tool matched.", args={"dataset_path": dataset_path}))
        notes.append("No specialized tool matched the query; using dataset inspection fallback.")

    strategy = "multi_tool" if len(steps) > 1 else "single_tool"
    return AgentDecision(strategy=strategy, steps=steps, notes=notes)


def _sanitize_decision(
    decision: AgentDecision,
    query: str,
    config: RuntimeConfig,
    registry: ToolRegistry,
    dataset_path: str,
    urls: list[str],
    expression: str,
) -> AgentDecision:
    available = {spec.name for spec in registry.specs()}
    steps: list[PlanStep] = []

    for step in decision.steps:
        if step.tool_name not in available:
            continue
        args = dict(step.args)
        if step.tool_name == "retrieve_corpus":
            args["query"] = query
            args["top_k"] = _safe_top_k(args.get("top_k"), config.retrieval.top_k)
        elif step.tool_name == "inspect_dataset":
            if not dataset_path:
                continue
            args["dataset_path"] = dataset_path
        elif step.tool_name == "calculate":
            args["expression"] = str(args.get("expression") or expression)
            if not args["expression"]:
                continue
        elif step.tool_name == "fetch_webpage":
            url = str(args.get("url") or (urls[0] if urls else ""))
            if not url:
                continue
            args["url"] = url
        steps.append(
            PlanStep(
                tool_name=step.tool_name,
                reason=step.reason or "Selected by the Ollama planner.",
                args=args,
            )
        )

    return AgentDecision(
        strategy="multi_tool" if len(steps) > 1 else "single_tool",
        steps=steps,
        notes=[note for note in decision.notes if isinstance(note, str)],
    )


def _looks_like_math(query: str) -> bool:
    return bool(re.search(r"\d+\s*[-+*/%]\s*\d+", query)) or "calculate" in query.lower()


def _extract_expression(query: str) -> str:
    match = re.search(r"(\d[\d\s\.\+\-\*\/%\(\)]*\d)", query)
    return match.group(1).replace(" ", "") if match else ""


def _needs_dataset_inspection(lower: str) -> bool:
    hints = ["dataset", "file", "files", "column", "columns", "metadata", "csv", "folder", "directory"]
    return any(hint in lower for hint in hints)


def _first_dataset_path(config: RuntimeConfig) -> str:
    for source in config.sources:
        if source.path and source.kind in {"dataset_dir", "local_dir"}:
            return str(Path(source.path))
    return ""


def _safe_top_k(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _trim(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    trimmed = compact[:limit].rsplit(" ", 1)[0]
    return f"{trimmed}..."
