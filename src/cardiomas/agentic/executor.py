from __future__ import annotations

from collections.abc import Generator

from cardiomas.autonomy.recovery import AutonomousToolManager
from cardiomas.memory.session import SessionStore
from cardiomas.safety.approvals import approval_required
from cardiomas.safety.permissions import tool_allowed
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.runtime import AgentDecision, AgentEvent
from cardiomas.schemas.tools import ToolCallRecord, ToolResult
from cardiomas.tools.registry import ToolRegistry


def execute_plan(
    decision: AgentDecision,
    config: RuntimeConfig,
    registry: ToolRegistry,
    session_store: SessionStore,
    session_id: str,
    autonomy_manager: AutonomousToolManager | None = None,
) -> tuple[list[ToolResult], list[ToolCallRecord], list[str]]:
    generator = execute_plan_events(
        decision,
        config,
        registry,
        session_store,
        session_id,
        autonomy_manager=autonomy_manager,
    )
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value


def execute_plan_events(
    decision: AgentDecision,
    config: RuntimeConfig,
    registry: ToolRegistry,
    session_store: SessionStore,
    session_id: str,
    autonomy_manager: AutonomousToolManager | None = None,
) -> Generator[AgentEvent, None, tuple[list[ToolResult], list[ToolCallRecord], list[str]]]:
    results: list[ToolResult] = []
    calls: list[ToolCallRecord] = []
    warnings: list[str] = []
    specs = {spec.name: spec for spec in registry.specs()}
    yield AgentEvent(type="status", stage="executor", message=f"Executing {len(decision.steps)} planned step(s).")

    for index, step in enumerate(decision.steps, start=1):
        spec = specs[step.tool_name]
        yield AgentEvent(
            type="tool_started",
            stage="tool",
            message=f"Starting tool {spec.name} ({index}/{len(decision.steps)}).",
            data={"tool_name": spec.name, "args": step.args, "index": index},
        )
        allowed, reason = tool_allowed(config, spec)
        if not allowed:
            warning = f"{spec.name}: {reason}"
            warnings.append(warning)
            calls.append(ToolCallRecord(tool_name=spec.name, args=step.args, ok=False, error=reason))
            yield AgentEvent(
                type="tool_finished",
                stage="tool",
                message=f"Tool {spec.name} blocked by policy.",
                data={"tool_name": spec.name, "ok": False, "error": reason},
            )
            continue
        if approval_required(config, spec):
            warning = f"{spec.name}: approval required before execution."
            warnings.append(warning)
            calls.append(ToolCallRecord(tool_name=spec.name, args=step.args, ok=False, error="approval required"))
            yield AgentEvent(
                type="tool_finished",
                stage="tool",
                message=f"Tool {spec.name} requires approval.",
                data={"tool_name": spec.name, "ok": False, "error": "approval required"},
            )
            continue
        trace_count_before = len(autonomy_manager.peek_traces()) if autonomy_manager is not None else 0
        try:
            result = registry.execute(step.tool_name, **step.args)
        except Exception as exc:
            warning = f"{spec.name}: execution failed: {exc}"
            warnings.append(warning)
            calls.append(ToolCallRecord(tool_name=step.tool_name, args=step.args, ok=False, error=str(exc)))
            yield AgentEvent(
                type="tool_finished",
                stage="tool",
                message=f"Tool {spec.name} failed.",
                data={"tool_name": spec.name, "ok": False, "error": str(exc)},
            )
            continue
        call = ToolCallRecord(
            tool_name=step.tool_name,
            args=step.args,
            ok=result.ok,
            summary=result.summary,
            error=result.error,
            repaired=spec.generated and result.ok,
        )
        session_store.append_tool_call(session_id, call)
        if not result.ok and result.error:
            warnings.append(f"{step.tool_name}: {result.error}")
        if autonomy_manager is not None:
            new_traces = autonomy_manager.peek_traces()[trace_count_before:]
            for trace in new_traces:
                yield AgentEvent(
                    type="repair_trace",
                    stage="repair",
                    message=f"{trace.action} for {trace.tool_name} attempt {trace.attempt}",
                    data=trace.model_dump(mode="json"),
                )
        results.append(result)
        calls.append(call)
        yield AgentEvent(
            type="tool_finished",
            stage="tool",
            message=f"Tool {spec.name} finished.",
            data={"tool_name": spec.name, "ok": result.ok, "summary": result.summary, "error": result.error},
        )
    yield AgentEvent(type="status", stage="executor", message="Plan execution finished.")
    return results, calls, warnings
