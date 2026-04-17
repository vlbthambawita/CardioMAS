from __future__ import annotations

import json
import re
from collections.abc import Generator
from pathlib import Path

from cardiomas.agentic.aggregator import aggregate_results
from cardiomas.agentic.answer_grader import grade_answer
from cardiomas.agentic.query_decomposer import SubQuery, decompose
from cardiomas.agentic.react_planner import generate_plan
from cardiomas.agentic.responder import compose_answer_events
from cardiomas.agentic.retrieval_grader import grade_chunks
from cardiomas.agentic.router import RouteDecision, route_query
from cardiomas.autonomy.recovery import AutonomousToolManager
from cardiomas.inference.base import ChatClient, ChatRequest
from cardiomas.inference.prompts import orchestrator_messages
from cardiomas.memory.persistent import PersistentMemory
from cardiomas.memory.scratchpad import Scratchpad
from cardiomas.memory.session import SessionStore
from cardiomas.safety.approvals import approval_required
from cardiomas.safety.permissions import tool_allowed
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import Citation, EvidenceChunk
from cardiomas.schemas.runtime import AgentDecision, AgentEvent, LLMTrace, PlanStep, ReActStep, RepairTrace
from cardiomas.schemas.tools import ToolCallRecord, ToolResult
from cardiomas.tools.pre_exec_verifier import verify_tool_args
from cardiomas.tools.registry import ToolRegistry


def run_react_events(
    query: str,
    config: RuntimeConfig,
    registry: ToolRegistry,
    chat_client: ChatClient,
    session_store: SessionStore,
    session_id: str,
    autonomy_manager: AutonomousToolManager | None = None,
    persistent_memory: PersistentMemory | None = None,
) -> Generator[AgentEvent, None, tuple[
    str, list[Citation], list[EvidenceChunk], dict,
    list[ToolCallRecord], list[LLMTrace], list[str], list[ReActStep],
]]:
    """ReAct agent loop replacing the linear planner→executor→aggregator chain.

    Yields AgentEvent objects for streaming, then returns:
    (answer, citations, evidence, aggregate, tool_calls, llm_traces, warnings, react_steps)
    """
    all_tool_results: list[ToolResult] = []
    all_tool_calls: list[ToolCallRecord] = []
    all_llm_traces: list[LLMTrace] = []
    all_warnings: list[str] = []
    all_react_steps: list[ReActStep] = []

    yield AgentEvent(type="status", stage="react", message="ReAct agent started.")

    # ── 1. Persistent memory check ──────────────────────────────────────────
    if persistent_memory is not None:
        cached = persistent_memory.find_similar(query)
        if cached is not None:
            sim = cached.get("_similarity", 0.0)
            yield AgentEvent(
                type="status", stage="react",
                message=f"Found similar past answer (similarity={sim:.2f}). Using as candidate.",
                data={"cached_query": cached.get("query", ""), "similarity": sim},
            )
            # Surface as a warning, still run fresh retrieval at reduced depth
            all_warnings.append(
                f"Similar past answer found (sim={sim:.2f}): {cached.get('answer', '')[:120]}..."
            )

    # ── 2. Route ────────────────────────────────────────────────────────────
    tools = registry.specs()
    route_decision = route_query(query, config, tools, chat_client)
    yield AgentEvent(
        type="status", stage="react",
        message=f"Route: {route_decision.route} — {route_decision.reason}",
        data={"route": route_decision.route},
    )

    # ── 3. Query decomposition ───────────────────────────────────────────────
    if config.agent.query_decomposition:
        sub_queries = decompose(query, config, chat_client)
    else:
        from cardiomas.agentic.query_decomposer import _infer_type
        sub_queries = [SubQuery(text=query, query_type=_infer_type(query))]

    if len(sub_queries) > 1:
        yield AgentEvent(
            type="status", stage="react",
            message=f"Decomposed into {len(sub_queries)} sub-queries.",
            data={"sub_queries": [sq.text for sq in sub_queries]},
        )

    # ── 4. Upfront planning (ReAct++) ────────────────────────────────────────
    react_plan: list[str] = []
    if config.agent.upfront_planning and chat_client is not None:
        react_plan = yield from generate_plan(query, config, registry, chat_client)

    # ── 5. ReAct loop over sub-queries ──────────────────────────────────────
    for sq_index, sub_query in enumerate(sub_queries):
        sq_label = f"[{sq_index + 1}/{len(sub_queries)}] " if len(sub_queries) > 1 else ""
        yield AgentEvent(
            type="status", stage="react",
            message=f"{sq_label}Starting ReAct loop: {sub_query.text[:80]}",
        )

        tool_results, tool_calls, llm_traces, warnings, steps = yield from _react_loop(
            sub_query=sub_query,
            parent_query=query,
            config=config,
            registry=registry,
            chat_client=chat_client,
            session_store=session_store,
            session_id=session_id,
            autonomy_manager=autonomy_manager,
            route_decision=route_decision,
            initial_plan=react_plan,
        )
        all_tool_results.extend(tool_results)
        all_tool_calls.extend(tool_calls)
        all_llm_traces.extend(llm_traces)
        all_warnings.extend(warnings)
        all_react_steps.extend(steps)

    # ── 6. Aggregate evidence ────────────────────────────────────────────────
    evidence, aggregate = aggregate_results(all_tool_results)

    # ── 7. Synthesize answer ─────────────────────────────────────────────────
    answer, citations, resp_traces, resp_warnings = yield from compose_answer_events(
        query=query,
        config=config,
        evidence=evidence,
        aggregate=aggregate,
        warnings=all_warnings,
        chat_client=chat_client,
    )
    all_llm_traces.extend(resp_traces)
    all_warnings.extend(resp_warnings)

    # ── 8. Optional self-reflection ──────────────────────────────────────────
    if config.agent.self_reflection and chat_client is not None:
        verdict = grade_answer(query, answer, evidence, config, chat_client)
        yield AgentEvent(
            type="status", stage="react",
            message=f"Answer grader verdict: {verdict}",
            data={"verdict": verdict},
        )
        if verdict == "hallucinated":
            all_warnings.append("Answer grader flagged potential hallucination — treat answer with caution.")
        elif verdict == "incomplete":
            all_warnings.append("Answer grader flagged incomplete answer — more information may be needed.")

    # ── 9. Store in persistent memory ───────────────────────────────────────
    if persistent_memory is not None and answer:
        grounded = not any("hallucination" in w.lower() for w in all_warnings)
        persistent_memory.store(
            query=query,
            answer=answer,
            grounded=grounded,
            evidence_ids=[c.chunk_id for c in evidence[:10]],
        )

    yield AgentEvent(type="status", stage="react", message="ReAct agent finished.")
    return answer, citations, evidence, aggregate, all_tool_calls, all_llm_traces, all_warnings, all_react_steps


def _react_loop(
    sub_query: SubQuery,
    parent_query: str,
    config: RuntimeConfig,
    registry: ToolRegistry,
    chat_client: ChatClient,
    session_store: SessionStore,
    session_id: str,
    autonomy_manager: AutonomousToolManager | None,
    route_decision: RouteDecision,
    initial_plan: list[str] | None = None,
) -> Generator[AgentEvent, None, tuple[
    list[ToolResult], list[ToolCallRecord], list[LLMTrace], list[str], list[ReActStep],
]]:
    tool_results: list[ToolResult] = []
    tool_calls: list[ToolCallRecord] = []
    llm_traces: list[LLMTrace] = []
    warnings: list[str] = []
    steps: list[ReActStep] = []
    observations: list[dict] = []
    called_tools: list[tuple[str, str]] = []  # (tool_name, args_key) for dedup
    stuck_count: int = 0  # consecutive "stuck" reflections

    # Short-term scratchpad (ReAct++)
    scratchpad = Scratchpad() if config.agent.scratchpad else None

    specs = {spec.name: spec for spec in registry.specs()}
    query = sub_query.text

    # Inject router hint then upfront plan (plan overrides hint when present)
    _hint_first_tool(observations, route_decision, registry, config, query)
    if initial_plan:
        observations.append({
            "tool": "_planner",
            "observation": (
                f"Upfront plan ({len(initial_plan)} steps): "
                f"{' → '.join(initial_plan)}. "
                "Follow this sequence but adapt if observations change."
            ),
        })

    assert config.llm is not None
    model = config.llm.resolved_responder_model or config.llm.model
    step_reflection = config.agent.step_reflection

    for iteration in range(1, config.agent.max_iterations + 1):
        yield AgentEvent(
            type="status", stage="react",
            message=f"Iteration {iteration}/{config.agent.max_iterations}",
        )

        # ── Orchestrator LLM call ──────────────────────────────────────────
        scratchpad_text = scratchpad.to_string() if scratchpad and not scratchpad.is_empty() else ""
        messages = orchestrator_messages(
            query, registry.specs(), observations,
            scratchpad_text=scratchpad_text,
            step_reflection=step_reflection,
        )
        trace = LLMTrace(
            stage=f"react-iter-{iteration}",
            provider=config.llm.provider,
            model=model,
        )
        try:
            request = ChatRequest(
                model=model,
                messages=messages,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                json_mode=True,
                keep_alive=config.llm.keep_alive,
            )
            yield AgentEvent(
                type="llm_stream_start", stage=f"react-iter-{iteration}",
                message="Orchestrator thinking...",
                data={"model": model},
            )
            streamed_content = ""
            for chunk in chat_client.chat_stream(request):
                if chunk.content:
                    streamed_content += chunk.content
                    yield AgentEvent(
                        type="llm_token", stage=f"react-iter-{iteration}",
                        content=chunk.content, data={},
                    )
            yield AgentEvent(
                type="llm_stream_end", stage=f"react-iter-{iteration}",
                message="", data={},
            )
            if not streamed_content.strip():
                fallback = chat_client.chat(request)
                streamed_content = fallback.content
            trace.response_preview = streamed_content[:200]
            thought_data = _parse_thought(streamed_content)
        except Exception as exc:
            trace.ok = False
            trace.error = str(exc)
            llm_traces.append(trace)
            warnings.append(f"Orchestrator LLM failed at iteration {iteration}: {exc}")
            break

        llm_traces.append(trace)
        thought = thought_data.get("thought", "")
        action = str(thought_data.get("action", "answer")).strip()
        action_args = dict(thought_data.get("args", {}))

        # Step reflection (ReAct++) — "sufficient" collapses to action="answer"
        reflection = (
            str(thought_data.get("reflection", "making_progress")).strip()
            if step_reflection else "making_progress"
        )
        if reflection == "sufficient":
            action = "answer"

        yield AgentEvent(
            type="status", stage="react",
            message=f"Thought: {thought[:120]}",
            data={"action": action, "args": action_args},
        )

        step = ReActStep(
            iteration=iteration,
            thought=thought,
            action=action,
            action_args=action_args,
        )

        # ── Stop condition: LLM says answer ──────────────────────────────
        if action == "answer" or action not in specs:
            step.observation = "Stopping: action is 'answer' or tool not available."
            steps.append(step)
            break

        # ── Dedup: skip identical repeated calls ─────────────────────────
        call_key = _call_key(action, action_args)
        if call_key in called_tools:
            step.observation = f"Skipping repeated call to {action} with same args."
            step.ok = False
            observations.append({"tool": action, "observation": step.observation})
            steps.append(step)
            continue
        called_tools.append(call_key)

        # ── Permission check ──────────────────────────────────────────────
        spec = specs[action]
        allowed, deny_reason = tool_allowed(config, spec)
        if not allowed or approval_required(config, spec):
            msg = deny_reason if not allowed else "approval required"
            step.ok = False
            step.error = msg
            step.observation = f"Tool {action} blocked: {msg}"
            warnings.append(f"{action}: {msg}")
            observations.append({"tool": action, "observation": step.observation})
            steps.append(step)
            continue

        # ── Tool arg verification (ReAct++) ──────────────────────────────
        if config.agent.tool_verification:
            valid, verify_error = verify_tool_args(action, action_args)
            if not valid:
                step.ok = False
                step.error = verify_error
                step.observation = f"Tool verification failed: {verify_error}"
                warnings.append(f"{action}: {verify_error}")
                observations.append({"tool": action, "observation": step.observation})
                steps.append(step)
                continue

        # ── Execute tool ──────────────────────────────────────────────────
        yield AgentEvent(
            type="tool_started", stage="tool",
            message=f"Calling {action}",
            data={"tool_name": action, "args": action_args},
        )
        trace_count_before = len(autonomy_manager.peek_traces()) if autonomy_manager else 0
        try:
            result = registry.execute(action, **action_args)
        except Exception as exc:
            step.ok = False
            step.error = str(exc)
            step.observation = f"Tool execution failed: {exc}"
            warnings.append(f"{action}: {exc}")
            observations.append({"tool": action, "observation": step.observation, "error": str(exc)})
            steps.append(step)
            yield AgentEvent(
                type="tool_finished", stage="tool",
                message=f"{action} failed.",
                data={"tool_name": action, "ok": False, "error": str(exc)},
            )
            continue

        # Emit repair traces
        if autonomy_manager is not None:
            for trace_item in autonomy_manager.peek_traces()[trace_count_before:]:
                yield AgentEvent(
                    type="repair_trace", stage="repair",
                    message=f"{trace_item.action} for {trace_item.tool_name} attempt {trace_item.attempt}",
                    data=trace_item.model_dump(mode="json"),
                )

        call_record = ToolCallRecord(
            tool_name=action,
            args=action_args,
            ok=result.ok,
            summary=result.summary,
            error=result.error,
            repaired=spec.generated and result.ok,
        )
        session_store.append_tool_call(session_id, call_record)
        tool_calls.append(call_record)
        tool_results.append(result)

        yield AgentEvent(
            type="tool_finished", stage="tool",
            message=f"{action} finished.",
            data={"tool_name": action, "ok": result.ok, "summary": result.summary[:200]},
        )

        if not result.ok:
            step.ok = False
            step.error = result.error
            step.observation = f"Tool returned error: {result.error}"
            warnings.append(f"{action}: {result.error}")
            observations.append({"tool": action, "observation": step.observation})
            steps.append(step)
            continue

        # ── Retrieval grading ─────────────────────────────────────────────
        observation_text = result.summary or "(tool ran, no summary)"
        if action == "retrieve_corpus" and config.agent.retrieval_grading and result.evidence:
            graded = grade_chunks(query, result.evidence, config, chat_client)
            yield AgentEvent(
                type="status", stage="react",
                message=f"Retrieval grader: {graded.verdict} ({graded.reason[:80]})",
                data={"verdict": graded.verdict, "relevant_count": graded.relevant_count},
            )
            observation_text = (
                f"Retrieved {len(result.evidence)} chunks, graded={graded.verdict}. "
                f"{graded.reason}"
            )
            if graded.verdict == "insufficient":
                step.observation = observation_text
                observations.append({"tool": action, "observation": observation_text, "graded": "insufficient"})
                steps.append(step)
                continue  # Loop back for another retrieval attempt

        step.observation = observation_text
        observations.append({"tool": action, "args": _summarize_args(action_args), "observation": observation_text})
        steps.append(step)

        # Update scratchpad with distilled key fact (ReAct++)
        if scratchpad is not None:
            scratchpad.add(action, observation_text)

        # Stuck detection: inject recovery hint after ≥2 consecutive stuck reflections
        if reflection == "stuck":
            stuck_count += 1
            if stuck_count >= 2:
                observations.append({
                    "tool": "_reflection",
                    "observation": (
                        "You appear stuck — the last two steps gave no new information. "
                        "Try a different tool, different arguments, or say action='answer' "
                        "if you have enough for a partial answer."
                    ),
                })
                stuck_count = 0
        else:
            stuck_count = 0

    return tool_results, tool_calls, llm_traces, warnings, steps


def _hint_first_tool(
    observations: list[dict],
    route: RouteDecision,
    registry: ToolRegistry,
    config: RuntimeConfig,
    query: str,
) -> None:
    """Inject a routing hint as the first observation so the LLM starts efficiently."""
    if route.route == "code" and registry.has("generate_python_artifact"):
        dataset_path = _first_dataset_path(config)
        observations.append({
            "tool": "_router",
            "observation": (
                f"Router suggests: start with generate_python_artifact for computational analysis. "
                f"Dataset path: {dataset_path}"
            ),
        })
    elif route.route == "web" and registry.has("fetch_webpage"):
        observations.append({
            "tool": "_router",
            "observation": "Router suggests: start with fetch_webpage to retrieve URL content.",
        })
    elif route.route == "retrieval" and registry.has("retrieve_corpus"):
        observations.append({
            "tool": "_router",
            "observation": f"Router suggests: start with retrieve_corpus to find relevant knowledge.",
        })


def _parse_thought(content: str) -> dict:
    """Parse LLM JSON response into thought dict. Fallback to 'answer' on failure."""
    try:
        return json.loads(content)
    except Exception:
        # Try extracting JSON from within the text
        match = re.search(r"\{[^{}]+\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {"thought": content[:200], "action": "answer", "args": {}}


def _call_key(action: str, args: dict) -> tuple[str, str]:
    return (action, json.dumps(sorted(args.items()), default=str))


def _summarize_args(args: dict, limit: int = 100) -> str:
    text = ", ".join(f"{k}={str(v)[:30]}" for k, v in args.items())
    return text[:limit]


def _first_dataset_path(config: RuntimeConfig) -> str:
    for source in config.sources:
        if source.path and source.kind in {"dataset_dir", "local_dir"}:
            return str(Path(source.path))
    return ""


def make_react_decision(react_steps: list[ReActStep]) -> AgentDecision:
    """Build a synthetic AgentDecision for QueryResult compatibility."""
    steps = [
        PlanStep(
            tool_name=s.action if s.action != "answer" else "_answer",
            reason=s.thought[:120] if s.thought else "ReAct iteration",
            args=s.action_args,
        )
        for s in react_steps
    ]
    return AgentDecision(
        strategy="react",
        steps=steps,
        notes=[f"Ran {len(react_steps)} ReAct iteration(s)."],
    )
