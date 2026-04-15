from __future__ import annotations

import logging

from cardiomas.agents.base import AgentOutputError, run_structured_agent
from cardiomas.schemas.agent_outputs import NLRequirementOutput
from cardiomas.schemas.requirement import ParsedRequirement
from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def nl_requirement_agent(state: GraphState) -> GraphState:
    """Parse a natural language requirement string into structured UserOptions."""
    from cardiomas.llm_factory import get_llm_for_agent
    from cardiomas.recorder import SessionRecorder

    requirement = state.user_options.requirement
    if not requirement:
        state.execution_log.append(LogEntry(
            agent="nl_requirement", action="skip", detail="no requirement provided"
        ))
        vprint("nl_requirement", "no --requirement provided — skipping")
        rec = SessionRecorder.get()
        rec.start_step("nl_requirement", "skip")
        rec.end_step(skipped=True, skip_reason="no requirement text")
        return state

    state.execution_log.append(LogEntry(agent="nl_requirement", action="start"))
    rec = SessionRecorder.get()
    rec.start_step("nl_requirement", "parse_requirement", inputs={"requirement": requirement[:200]})
    vprint("nl_requirement", f"parsing: \"{requirement[:100]}{'…' if len(requirement) > 100 else ''}\"")

    llm = get_llm_for_agent(
        "nl_requirement",
        prefer_cloud=state.user_options.use_cloud_llm,
        agent_llm_map=state.user_options.agent_llm_map,
    )

    # Provide dataset context if already known
    context = ""
    if state.dataset_info:
        context = (
            f"Dataset: {state.dataset_info.name}\n"
            f"Records: {state.dataset_info.num_records}\n"
            f"ID field: {state.dataset_info.ecg_id_field}\n"
            f"Metadata fields: {', '.join(state.dataset_info.metadata_fields[:20])}\n"
        )

    prompt = (
        f'Parse this natural language split requirement:\n\n"{requirement}"\n\n'
        "Rules:\n"
        "- split_ratios values must sum to 1.0; default 70/15/15 if not specified\n"
        "- stratify_by: exact field name string, or null\n"
        "- exclusion_filters: list of {\"field\": \"...\", \"op\": \"notna|eq|ne|gt|lt\", \"value\": ...}\n"
        "- patient_level: true unless user says 'record-level' or 'sample-level'\n"
        "- seed: integer or null\n"
        "- notes: anything you cannot parse precisely\n"
        "- llm_reasoning: brief explanation of your decisions"
    )

    try:
        output: NLRequirementOutput = run_structured_agent(
            llm, "nl_requirement", prompt, NLRequirementOutput, extra_context=context
        )
    except AgentOutputError as exc:
        logger.error(f"nl_requirement_agent: structured output failed — {exc}")
        state.errors.append(f"nl_requirement: {exc}")
        output = NLRequirementOutput(
            notes=f"Automatic parsing failed: {exc}. Original requirement preserved.",
        )

    parsed = ParsedRequirement(
        split_ratios=output.split_ratios,
        stratify_by=output.stratify_by,
        exclusion_filters=output.exclusion_filters,
        patient_level=output.patient_level,
        seed=output.seed,
        notes=output.notes,
        raw_input=requirement,
        llm_reasoning=output.llm_reasoning,
    )
    state.parsed_requirement = parsed

    # Apply parsed values to UserOptions (only override defaults, not explicit user flags)
    opts = state.user_options
    updates: dict = {}
    if parsed.seed is not None and opts.seed == 42:
        updates["seed"] = parsed.seed
    if parsed.stratify_by and opts.stratify_by is None:
        updates["stratify_by"] = parsed.stratify_by
    if opts.custom_split is None:
        total = sum(parsed.split_ratios.values())
        if 0.98 <= total <= 1.02:
            updates["custom_split"] = {k: v / total for k, v in parsed.split_ratios.items()}
        else:
            logger.warning(f"Parsed ratios sum to {total:.3f} — not applying to custom_split")
    if updates:
        state.user_options = opts.model_copy(update=updates)

    if parsed.notes:
        vprint("nl_requirement", f"[yellow]unparsed notes: {parsed.notes[:120]}[/yellow]")

    state.execution_log.append(LogEntry(
        agent="nl_requirement", action="parsed",
        detail=f"ratios={parsed.split_ratios} stratify={parsed.stratify_by}",
    ))
    vprint("nl_requirement", f"parsed — ratios={parsed.split_ratios}, stratify_by={parsed.stratify_by}")

    rec.end_step(
        outputs={"split_ratios": parsed.split_ratios, "stratify_by": parsed.stratify_by},
        reasoning=parsed.llm_reasoning[:200],
    )
    return state
