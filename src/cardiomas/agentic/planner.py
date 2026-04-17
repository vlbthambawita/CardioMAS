from __future__ import annotations

from collections.abc import Generator
import json
import re
from pathlib import Path

from cardiomas.inference.base import ChatClient, ChatRequest
from cardiomas.inference.prompts import planner_messages, prompt_preview
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.runtime import AgentDecision, AgentEvent, LLMTrace, PlanStep
from cardiomas.tools.registry import ToolRegistry


def plan_query(
    query: str,
    config: RuntimeConfig,
    registry: ToolRegistry,
    chat_client: ChatClient | None = None,
) -> tuple[AgentDecision, list[LLMTrace], list[str]]:
    generator = plan_query_events(query, config, registry, chat_client=chat_client)
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value


def plan_query_events(
    query: str,
    config: RuntimeConfig,
    registry: ToolRegistry,
    chat_client: ChatClient | None = None,
) -> Generator[AgentEvent, None, tuple[AgentDecision, list[LLMTrace], list[str]]]:
    yield AgentEvent(type="status", stage="planner", message="Planning started.")
    if config.planner_uses_ollama and chat_client is not None and config.llm is not None:
        result = yield from _plan_with_ollama_events(query, config, registry, chat_client)
        yield AgentEvent(type="status", stage="planner", message=f"Planner completed with {len(result[0].steps)} step(s).")
        return result
    if config.planner_uses_ollama and chat_client is None:
        decision = _heuristic_plan(query, config, registry)
        warning = "Ollama planner was requested, but no chat client was available; using heuristic planner."
        yield AgentEvent(type="status", stage="planner", message=warning)
        yield AgentEvent(type="status", stage="planner", message=f"Planner completed with {len(decision.steps)} step(s).")
        return decision, [], [warning]
    decision = _heuristic_plan(query, config, registry)
    yield AgentEvent(type="status", stage="planner", message=f"Planner completed with {len(decision.steps)} step(s).")
    return decision, [], []


def _plan_with_ollama_events(
    query: str,
    config: RuntimeConfig,
    registry: ToolRegistry,
    chat_client: ChatClient,
) -> Generator[AgentEvent, None, tuple[AgentDecision, list[LLMTrace], list[str]]]:
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
        request = ChatRequest(
            model=config.llm.resolved_planner_model,
            messages=messages,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            json_mode=True,
            keep_alive=config.llm.keep_alive,
        )
        yield AgentEvent(
            type="llm_stream_start",
            stage="planner",
            message="Planner LLM stream started.",
            data={"model": config.llm.resolved_planner_model},
        )
        streamed_content = ""
        for chunk in chat_client.chat_stream(request):
            if chunk.content:
                streamed_content += chunk.content
                yield AgentEvent(
                    type="llm_token",
                    stage="planner",
                    content=chunk.content,
                    data={"model": chunk.model},
                )
        yield AgentEvent(
            type="llm_stream_end",
            stage="planner",
            message="Planner LLM stream ended.",
            data={"model": config.llm.resolved_planner_model},
        )
        trace.response_preview = _trim(streamed_content)
        raw = _normalize_planner_payload(json.loads(streamed_content))
        decision = AgentDecision.model_validate(raw)
        sanitized = _sanitize_decision(decision, query, config, registry, dataset_path, urls, expression)
        if not sanitized.steps:
            trace.ok = False
            trace.error = "Planner response did not yield executable steps."
            warning = "Ollama planner returned no valid steps; using heuristic planner."
            yield AgentEvent(type="status", stage="planner", message=warning)
            return fallback, [trace], [warning]
        return sanitized, [trace], []
    except Exception as exc:
        trace.ok = False
        trace.error = str(exc)
        warning = f"Ollama planner failed; using heuristic planner. {exc}"
        yield AgentEvent(type="status", stage="planner", message=warning)
        return fallback, [trace], [warning]


def _plan_with_ollama(
    query: str,
    config: RuntimeConfig,
    registry: ToolRegistry,
    chat_client: ChatClient,
) -> tuple[AgentDecision, list[LLMTrace], list[str]]:
    generator = _plan_with_ollama_events(query, config, registry, chat_client)
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value


def _is_script_only_mode(config: RuntimeConfig) -> bool:
    return (
        config.autonomy.dataset_mode == "script_only"
        and bool(_first_dataset_path(config))
        and config.autonomy.enabled
        and config.autonomy.allow_tool_codegen
    )


def _heuristic_plan(query: str, config: RuntimeConfig, registry: ToolRegistry) -> AgentDecision:
    lower = query.lower()
    steps: list[PlanStep] = []
    notes: list[str] = []
    available = {spec.name for spec in registry.specs()}

    if _is_script_only_mode(config) and "generate_python_artifact" in available:
        dataset_path = _first_dataset_path(config)
        target_path = _extract_local_path(query)
        return AgentDecision(
            strategy="single_tool",
            steps=[
                PlanStep(
                    tool_name="generate_python_artifact",
                    reason="Dataset query in script_only mode — generating standalone analysis script.",
                    args={"task": query, "dataset_path": dataset_path, "target_path": target_path, "artifact_name": ""},
                )
            ],
            notes=["script_only mode: routing exclusively to generate_python_artifact."],
        )

    urls = re.findall(r"https?://\S+", query)
    if urls and "fetch_webpage" in available:
        steps.append(PlanStep(tool_name="fetch_webpage", reason="Query includes a direct URL.", args={"url": urls[0]}))

    if _looks_like_math(query) and "calculate" in available:
        expression = _extract_expression(query)
        if expression:
            steps.append(PlanStep(tool_name="calculate", reason="Query contains an arithmetic expression.", args={"expression": expression}))

    dataset_path = _first_dataset_path(config)
    target_path = _extract_local_path(query)
    if dataset_path and "generate_python_artifact" in available and _needs_dynamic_code_generation(lower, target_path):
        steps.append(
            PlanStep(
                tool_name="generate_python_artifact",
                reason="Query asks for dataset reading or analysis that should be handled by a generated Python artifact.",
                args={"task": query, "dataset_path": dataset_path, "target_path": target_path},
            )
        )

    if dataset_path and "inspect_dataset" in available and _needs_dataset_inspection(lower) and "generate_python_artifact" not in available:
        steps.append(PlanStep(tool_name="inspect_dataset", reason="Query asks about local dataset structure or metadata.", args={"dataset_path": dataset_path}))

    if dataset_path and "generate_shell_artifact" in available and _needs_shell_script(lower):
        steps.append(
            PlanStep(
                tool_name="generate_shell_artifact",
                reason="Query asks for a shell script or batch command.",
                args={"task": query, "dataset_path": dataset_path, "execute": _asks_to_run_script(lower)},
            )
        )

    if "retrieve_corpus" in available:
        steps.append(PlanStep(tool_name="retrieve_corpus", reason="Ground the answer with corpus retrieval.", args={"query": query, "top_k": config.retrieval.top_k}))

    if not steps and dataset_path and "generate_python_artifact" in available:
        steps.append(
            PlanStep(
                tool_name="generate_python_artifact",
                reason="Fallback to a generated Python artifact for the dataset query.",
                args={"task": query, "dataset_path": dataset_path, "target_path": target_path},
            )
        )
        notes.append("No specialized tool matched the query; using generated dataset code as the fallback.")
    elif not steps and "inspect_dataset" in available and dataset_path:
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

    if _is_script_only_mode(config) and "generate_python_artifact" in available:
        pa_steps = [s for s in decision.steps if s.tool_name == "generate_python_artifact"]
        if not pa_steps:
            return _heuristic_plan(query, config, registry)
        decision = AgentDecision(strategy="single_tool", steps=pa_steps[:1], notes=decision.notes)

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
            args = {"dataset_path": dataset_path}
        elif step.tool_name == "calculate":
            args["expression"] = str(args.get("expression") or expression)
            if not args["expression"]:
                continue
        elif step.tool_name == "fetch_webpage":
            url = str(args.get("url") or (urls[0] if urls else ""))
            if not url:
                continue
            args["url"] = url
        elif step.tool_name == "generate_python_artifact":
            args = {
                "task": query,
                "dataset_path": dataset_path,
                "target_path": str(step.args.get("target_path") or _extract_local_path(query)),
                "artifact_name": str(step.args.get("artifact_name") or ""),
            }
        elif step.tool_name == "generate_shell_artifact":
            if not dataset_path:
                continue
            args = {
                "task": query,
                "dataset_path": dataset_path,
                "artifact_name": str(step.args.get("artifact_name") or ""),
                "execute": bool(step.args.get("execute")) and _asks_to_run_script(query.lower()),
            }
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


def _needs_dataset_statistics(lower: str) -> bool:
    hints = ["statistics", "statistical", "distribution", "missing", "mean", "median", "summary stats", "class count", "counts"]
    return any(hint in lower for hint in hints)


def _needs_shell_script(lower: str) -> bool:
    hints = ["shell script", "bash script", "script", "command script"]
    return any(hint in lower for hint in hints)


def _needs_file_read(lower: str) -> bool:
    hints = ["read file", "inspect file", "open file", "show file", "read dataset file"]
    return any(hint in lower for hint in hints)


def _needs_dynamic_code_generation(lower: str, target_path: str) -> bool:
    if _needs_shell_script(lower):
        return False
    if target_path:
        return True
    return _needs_file_read(lower) or _needs_dataset_statistics(lower) or _needs_dataset_inspection(lower) or any(
        hint in lower
        for hint in [
            "analyze",
            "analysis",
            "summarize",
            "summary",
            "label",
            "class",
            "metadata",
            "columns",
            "rows",
            "missing",
            "hea",
            "header",
        ]
    )


def _asks_to_run_script(lower: str) -> bool:
    return any(hint in lower for hint in ["run the script", "execute the script", "and run it", "execute it"])


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


def _normalize_planner_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Planner response must be a JSON object.")

    normalized = dict(payload)
    notes = normalized.get("notes", [])
    if notes is None:
        normalized["notes"] = []
    elif isinstance(notes, str):
        normalized["notes"] = [notes]
    elif not isinstance(notes, list):
        normalized["notes"] = [str(notes)]

    steps = normalized.get("steps", [])
    if isinstance(steps, dict):
        normalized["steps"] = [steps]
    elif not isinstance(steps, list):
        normalized["steps"] = []

    return normalized


def _trim(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    trimmed = compact[:limit].rsplit(" ", 1)[0]
    return f"{trimmed}..."


def _extract_local_path(query: str) -> str:
    matches = re.findall(r"(/[^\s\"']+)", query)
    for match in matches:
        if Path(match).exists():
            return match
    return ""
