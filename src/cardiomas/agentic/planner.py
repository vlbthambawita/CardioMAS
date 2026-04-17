from __future__ import annotations

import re
from pathlib import Path

from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.runtime import AgentDecision, PlanStep
from cardiomas.tools.registry import ToolRegistry


def plan_query(query: str, config: RuntimeConfig, registry: ToolRegistry) -> AgentDecision:
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
