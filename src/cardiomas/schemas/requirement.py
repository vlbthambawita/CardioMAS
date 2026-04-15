from __future__ import annotations

from pydantic import BaseModel, Field


class ParsedRequirement(BaseModel):
    """Structured representation of a natural language split requirement."""

    split_ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    stratify_by: str | None = None
    exclusion_filters: list[dict] = Field(default_factory=list)
    # e.g. [{"field": "age", "op": "notna"}, {"field": "rhythm", "op": "eq", "value": "SR"}]
    patient_level: bool = True
    seed: int | None = None
    notes: str = ""              # anything the agent could not parse
    raw_input: str = ""          # original user text (for reproducibility log)
    llm_reasoning: str = ""      # agent's explanation of its parsing decisions
