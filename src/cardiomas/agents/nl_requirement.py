from __future__ import annotations

import json
import logging

from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


def nl_requirement_agent(state: GraphState) -> GraphState:
    """Parse a natural language requirement string into structured UserOptions."""
    from cardiomas.agents.base import run_agent
    from cardiomas.llm_factory import get_llm_for_agent
    from cardiomas.recorder import SessionRecorder
    from cardiomas.schemas.requirement import ParsedRequirement

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

    prefer_cloud = state.user_options.use_cloud_llm
    llm = get_llm_for_agent(
        "nl_requirement",
        prefer_cloud=prefer_cloud,
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

    user_msg = f"""Parse this natural language split requirement into structured JSON.

Requirement: "{requirement}"

Return ONLY valid JSON with this exact schema (no extra text before or after):
{{
  "split_ratios": {{"train": 0.7, "val": 0.15, "test": 0.15}},
  "stratify_by": null,
  "exclusion_filters": [],
  "patient_level": true,
  "seed": null,
  "notes": "",
  "raw_input": "{requirement.replace('"', "'")}",
  "llm_reasoning": "explain your parsing decisions here"
}}

Rules:
- split_ratios must sum to 1.0; default 70/15/15 if not specified
- stratify_by: field name string or null
- exclusion_filters: list of {{"field": "...", "op": "notna|eq|ne|gt|lt", "value": ...}}
- patient_level: true unless user says "record-level" or "sample-level"
- seed: integer or null
- notes: anything you cannot parse precisely"""

    response = run_agent(llm, "nl_requirement", user_msg, extra_context=context)

    # Parse JSON from response
    parsed: ParsedRequirement
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            parsed = ParsedRequirement(
                split_ratios=data.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15}),
                stratify_by=data.get("stratify_by"),
                exclusion_filters=data.get("exclusion_filters", []),
                patient_level=data.get("patient_level", True),
                seed=data.get("seed"),
                notes=data.get("notes", ""),
                raw_input=requirement,
                llm_reasoning=data.get("llm_reasoning", ""),
            )
        else:
            raise ValueError("No JSON object found in response")
    except Exception as e:
        logger.warning(f"nl_requirement: JSON parse failed ({e}) — using defaults with note")
        parsed = ParsedRequirement(
            raw_input=requirement,
            notes=f"Automatic parsing failed: {e}. Original requirement preserved for record.",
            llm_reasoning=response[:400],
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
        # Validate ratios sum
        total = sum(parsed.split_ratios.values())
        if 0.99 <= total <= 1.01:
            updates["custom_split"] = {k: v / total for k, v in parsed.split_ratios.items()}
        else:
            logger.warning(f"Parsed ratios sum to {total:.3f} — not applying to custom_split")
    if updates:
        state.user_options = opts.model_copy(update=updates)

    if parsed.notes:
        vprint("nl_requirement", f"[yellow]unparsed notes: {parsed.notes[:120]}[/yellow]")

    state.execution_log.append(LogEntry(
        agent="nl_requirement", action="parsed",
        detail=f"ratios={parsed.split_ratios} stratify={parsed.stratify_by}"
    ))
    vprint("nl_requirement", f"parsed — ratios={parsed.split_ratios}, stratify_by={parsed.stratify_by}")

    rec.end_step(
        outputs={"split_ratios": parsed.split_ratios, "stratify_by": parsed.stratify_by},
        reasoning=parsed.llm_reasoning[:200],
    )
    return state
