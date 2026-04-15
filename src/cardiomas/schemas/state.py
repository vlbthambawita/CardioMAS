from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from cardiomas.schemas.dataset import DatasetInfo
from cardiomas.schemas.split import SplitManifest
from cardiomas.schemas.audit import SecurityAudit

# Forward-declared to avoid circular imports; resolved at runtime
DatasetMapType = Any  # cardiomas.mappers.schemas.DatasetMap


class UserOptions(BaseModel):
    dataset_source: str
    local_path: str | None = None
    output_dir: str = "output"
    force_reanalysis: bool = False
    use_cloud_llm: bool = False
    seed: int = 42
    custom_split: dict[str, float] | None = None
    ignore_official: bool = False
    stratify_by: str | None = None
    verbose: bool = False
    push_to_hf: bool = False          # explicit opt-in; requires HF_TOKEN
    requirement: str | None = None    # natural language requirement input (V2)
    agent_llm_map: dict[str, str] = Field(default_factory=dict)  # per-agent LLM overrides (V2)

    # ── V4 options ─────────────────────────────────────────────────────────
    v4_auto_approve: bool = False    # skip human approval gate
    v4_subset_size: int = 100        # override subset size
    v4_max_refinements: int = 2      # max refinement rounds per script
    v4_skip_ecg_stats: bool = False  # skip ECG statistics phase
    v4_plot_format: str = "png"      # "png" | "pdf" | "svg"


# ── V4 new Pydantic models ─────────────────────────────────────────────────

class ScriptRecord(BaseModel):
    """Metadata about a generated script."""
    name: str                      # e.g. "00_explore_structure.py"
    path: str                      # absolute path to the script file
    purpose: str                   # human-readable description
    output_dir: str                # directory where script writes its outputs
    timeout: int = 300             # execution timeout in seconds
    phase: str = "subset"          # "subset" | "full"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    sha256: str = ""               # SHA-256 of script content


class ExecutionResult(BaseModel):
    """Captured output from one script execution."""
    script_name: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    generated_files: dict[str, str] = Field(default_factory=dict)
    # {filename -> text_content} for script-generated files only
    verification_passed: bool = False
    verification_notes: str = ""
    executed_at: datetime = Field(default_factory=datetime.utcnow)


class RefinementContext(BaseModel):
    """Context passed back to data_engineer when a script fails."""
    failed_script: str
    error_message: str
    stdout_excerpt: str
    suggested_fix: str = ""        # LLM-populated on re-entry
    attempt: int = 1


class ApprovalSummary(BaseModel):
    """Human-readable summary shown at approval gate."""
    dataset_name: str
    subset_size: int
    records_found: int
    columns_found: list[str] = Field(default_factory=list)
    label_distribution_excerpt: str = ""
    split_sizes: dict[str, int] = Field(default_factory=dict)
    scripts_passed: list[str] = Field(default_factory=list)
    scripts_failed: list[str] = Field(default_factory=list)
    output_dir: str = ""


class LogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent: str
    action: str
    detail: str = ""


class GraphState(BaseModel):
    dataset_source: str
    dataset_info: DatasetInfo | None = None
    download_status: str = "pending"
    paper_findings: dict[str, Any] | None = None
    analysis_report: dict[str, Any] | None = None
    existing_hf_splits: dict | None = None
    proposed_splits: SplitManifest | None = None
    security_audit: SecurityAudit | None = None
    publish_status: str = "pending"
    local_output_dir: str = ""   # set by splitter_agent after saving locally
    user_options: UserOptions = Field(default_factory=UserOptions.model_construct)
    execution_log: list[LogEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # ── V2 orchestrator fields ─────────────────────────────────────────────
    session_id: str = ""
    next_agent: str = ""
    last_completed_agent: str = ""
    agent_skip_reasons: dict[str, str] = Field(default_factory=dict)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    orchestrator_reasoning: list[str] = Field(default_factory=list)

    # ── V2 NL requirement ─────────────────────────────────────────────────
    parsed_requirement: Any | None = None   # ParsedRequirement | None

    # ── V2 coder agent ────────────────────────────────────────────────────
    generated_scripts: dict[str, str] = Field(default_factory=dict)   # {name: path}
    script_execution_log: list[dict] = Field(default_factory=list)
    script_verified: bool = False

    # ── V2 checkpoint / resume ────────────────────────────────────────────
    checkpoint_path: str = ""
    last_completed_agent_at: str = ""  # ISO timestamp of last agent completion

    # ── V3 dataset semantic map (Phase 2) ─────────────────────────────────
    dataset_map: Any | None = None   # DatasetMap | None (avoid circular import)

    # ── V4 indirect execution ──────────────────────────────────────────────
    # Output directory structure for V4 artifacts
    v4_output_dir: str = ""          # e.g. "output/ptb-xl/v4/"
    v4_subset_size: int = 100        # number of records for subset validation

    # Script management
    v4_generated_scripts: dict[str, ScriptRecord] = Field(default_factory=dict)
    # keyed by script name, e.g. "00_explore_structure.py"

    # Execution results
    v4_execution_results: list[ExecutionResult] = Field(default_factory=list)
    v4_execution_summary: str = ""   # aggregated text summary for LLM context
    v4_generated_files: dict[str, str] = Field(default_factory=dict)
    # {filename -> text_content} for machine-readable generated outputs

    # Pipeline phase tracking
    v4_pipeline_phase: str = "subset_validation"
    # values: "subset_validation" | "full_run" | "ecg_stats_run"
    v4_subset_validated: bool = False

    # Human-in-the-loop approval
    v4_approval_status: str = "pending"   # "pending" | "approved" | "rejected"
    v4_approval_summary: ApprovalSummary | None = None

    # Iterative refinement
    v4_refinement_context: RefinementContext | None = None
    v4_refinement_rounds: dict[str, int] = Field(default_factory=dict)
    # {script_name -> number_of_refinement_rounds}

    # ECG statistics outputs
    v4_ecg_stats_dir: str = ""       # path to directory with all stat outputs
    v4_ecg_stats_scripts: dict[str, ScriptRecord] = Field(default_factory=dict)
