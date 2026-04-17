from __future__ import annotations

from cardiomas.memory.session import SessionStore
from cardiomas.safety.approvals import approval_required
from cardiomas.safety.permissions import tool_allowed
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.runtime import AgentDecision
from cardiomas.schemas.tools import ToolCallRecord, ToolResult
from cardiomas.tools.registry import ToolRegistry


def execute_plan(
    decision: AgentDecision,
    config: RuntimeConfig,
    registry: ToolRegistry,
    session_store: SessionStore,
    session_id: str,
) -> tuple[list[ToolResult], list[ToolCallRecord], list[str]]:
    results: list[ToolResult] = []
    calls: list[ToolCallRecord] = []
    warnings: list[str] = []
    specs = {spec.name: spec for spec in registry.specs()}

    for step in decision.steps:
        spec = specs[step.tool_name]
        allowed, reason = tool_allowed(config, spec)
        if not allowed:
            warning = f"{spec.name}: {reason}"
            warnings.append(warning)
            calls.append(ToolCallRecord(tool_name=spec.name, args=step.args, ok=False, error=reason))
            continue
        if approval_required(config, spec):
            warning = f"{spec.name}: approval required before execution."
            warnings.append(warning)
            calls.append(ToolCallRecord(tool_name=spec.name, args=step.args, ok=False, error="approval required"))
            continue
        result = registry.execute(step.tool_name, **step.args)
        call = ToolCallRecord(
            tool_name=step.tool_name,
            args=step.args,
            ok=result.ok,
            summary=result.summary,
            error=result.error,
        )
        session_store.append_tool_call(session_id, call)
        results.append(result)
        calls.append(call)
    return results, calls, warnings
