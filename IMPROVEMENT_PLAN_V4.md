# CardioMAS V4 — Implementation Plan: Indirect Execution Architecture

> **Branch:** `feature/v4-indirect-execution`
> **Base:** current `feature/v3-rag-structured-reasoning` (all V3 phases complete)
> **Status:** Draft — approved for implementation

---

## Table of Contents

1. [Root Diagnosis: Why V3 Violates the V4 Core Rule](#1-root-diagnosis)
2. [Architecture Overview](#2-architecture-overview)
3. [New Agent Specifications](#3-new-agent-specifications)
4. [Workflow Graph Topology](#4-workflow-graph-topology)
5. [New State Schema Fields](#5-new-state-schema-fields)
6. [New Tools and Modules](#6-new-tools-and-modules)
7. [Changes to Existing Agents](#7-changes-to-existing-agents)
8. [Implementation Phases](#8-implementation-phases)
9. [File-Level Change List](#9-file-level-change-list)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. Root Diagnosis

### V3 Violations of the V4 Core Rule

The fundamental V4 constraint is: **agents must NOT directly read dataset files**. Instead, all dataset understanding must come through executing scripts and reading their outputs.

In V3 (current state), the following agents directly read files:

| Agent | Direct File Access | V4 Violation |
|---|---|---|
| `analysis_agent` | `read_csv_metadata.invoke()` on CSVs; `DatasetMapper` reads headers, HDF5 keys, WFDB headers | **Hard violation** — agent calls tools that open files directly |
| `splitter_agent` | `pd.read_csv(csv_path)` in `_load_record_ids()` fallback | **Hard violation** — agent opens files directly in Python |
| `coder_agent` | No direct reads, but generates scripts assuming path availability; execution runs inline | **Soft violation** — execution happens in-process without structured output capture |
| `discovery_agent` | None — reads webpage HTML only | Compliant |
| `security_agent` | None — reads from `state.dataset_map` built by others | Compliant (if analysis is fixed) |

The `DatasetMapper` in `mappers/dataset_mapper.py` — called by `analysis_agent` — is the primary offender. It uses Python to directly open and inspect files. Under V4, this logic must be relocated into a *script* that the `data_engineer` agent generates and the `executor` agent runs.

---

## 2. Architecture Overview

### V3 Architecture (Current)

```
orchestrator (hub)
    ├── nl_requirement
    ├── discovery
    ├── paper
    ├── analysis  ← directly reads CSVs/WFDB/HDF5 via DatasetMapper
    ├── splitter  ← directly reads CSVs to load record IDs
    ├── security  ← uses data from analysis (indirect via state)
    ├── coder     ← generates scripts, runs inline subprocess
    └── publisher
```

### V4 Architecture (Target)

```
orchestrator (hub) — extended routing table
    ├── nl_requirement           (unchanged)
    ├── discovery                (unchanged)
    ├── paper                    (unchanged)
    │
    ├── data_engineer  [NEW]     ← generates exploration + preprocessing + split scripts
    ├── executor       [NEW]     ← runs scripts step-by-step, captures all output
    ├── analysis                 [REFACTORED] ← consumes executor output only, no file access
    ├── ecg_stats      [NEW]     ← generates ECG statistics + quality + plot scripts
    │
    ├── splitter                 [REFACTORED] ← reads IDs from executor-produced JSON only
    ├── security                 (minor update — now reads patient map from executor output)
    ├── coder                    [REFACTORED] ← generates final reproducibility scripts only
    └── publisher                (unchanged)
```

### Core Data Flow

```
1. data_engineer generates:
   - 00_explore_structure.py   → stdout: file tree, format counts
   - 01_extract_metadata.py    → stdout: column names, sample rows
   - 02_compute_statistics.py  → writes: stats.csv, class_dist.csv
   - 03_generate_splits.py     → writes: splits.json, split_manifest.json
   (on subset first: N=100 records)

2. executor runs each script in sequence:
   - captures stdout, stderr, exit_code
   - reads generated files as text
   - stores all output in GraphState

3. analysis reads executor output only (no file I/O)
   - parses stdout from scripts as context
   - produces AnalysisOutput via LLM

4. human approval gate (after subset validation)
   - summarise subset results, request --approve flag
   - if approved (or auto-approve), proceed to full run

5. ecg_stats generates:
   - 10_class_distribution.py  → writes: class_dist.csv, class_dist.png
   - 11_per_lead_statistics.py → writes: lead_stats.csv
   - 12_signal_quality.py      → writes: quality_report.csv
   - 13_clinical_plausibility.py → writes: clinical_flags.csv
   - 14_publication_table.py   → writes: table1.md, table1.tex

6. executor runs ecg_stats scripts (full dataset)

7. splitter reads executor-produced splits.json only

8. coder generates final verify_splits.py (reproducibility)

9. publisher (unchanged)
```

---

## 3. New Agent Specifications

### 3.1 Data Engineering Agent (`data_engineer`)

**File:** `src/cardiomas/agents/data_engineer.py`

**Purpose:** Generates all Python/shell scripts needed to explore dataset structure, extract metadata, preprocess, and split. Never reads dataset files directly.

**Inputs (from `GraphState`):**
- `state.dataset_info` — dataset name, source, ID field hint, num_records estimate
- `state.user_options` — `local_path`, `seed`, `custom_split`, `stratify_by`
- `state.parsed_requirement` — parsed NL constraints
- `state.paper_findings` — split methodology from paper agent
- `state.v4_output_dir` — base output directory for all scripts
- `state.v4_subset_size` — number of records for subset validation (default 100)

**Outputs (to `GraphState`):**
- `state.v4_generated_scripts` — `dict[str, ScriptRecord]` keyed by script name
- `state.v4_pipeline_phase` — set to `"subset_validation"`
- `state.execution_log` — appended entries

**Agent Function Signature:**
```python
def data_engineer_agent(state: GraphState) -> GraphState:
    ...
```

**Scripts Generated (with naming convention `{NN}_{name}.py`):**

| Script | Purpose | Key Output |
|---|---|---|
| `00_explore_structure.py` | Walk directory tree, count files by extension, detect format | stdout: structured JSON summary |
| `01_extract_metadata.py` | Open first CSV/TSV, print column names, dtypes, 5 sample rows | stdout: column schema JSON |
| `02_compute_statistics.py` | Compute value_counts on label/ID columns | writes: `stats.csv` |
| `03_generate_splits_subset.py` | Generate splits on first `SUBSET_SIZE` records | writes: `splits_subset.json` |

**Script Properties (enforced by LLM prompt):**
- All parameters as `ALL_CAPS` constants at the top
- `DATASET_PATH`, `OUTPUT_DIR`, `SUBSET_SIZE`, `SEED` constants
- Stdlib + `pandas` + `numpy` only — no cardiomas dependency
- All output written to `OUTPUT_DIR` (passed as constant)
- All stdout structured as `KEY=VALUE` or JSON for machine parsing
- Complete docstring with: purpose, inputs, outputs, author (`CardioMAS V4`)
- Graceful error handling — always exits with code 0 unless fatal

**LLM Skill File:** `src/cardiomas/skills/data_engineer.md`

**Context Built for LLM:**
```python
context = {
    "dataset_name": info.name,
    "dataset_path": opts.local_path,
    "id_field_hint": info.ecg_id_field,
    "num_records_estimate": info.num_records,
    "subset_size": state.v4_subset_size,
    "seed": opts.seed,
    "output_dir": str(scripts_dir),
    "split_ratios": split_ratios,
    "stratify_by": opts.stratify_by,
    "paper_methodology": paper.get("split_methodology", ""),
}
```

**Iterative Refinement:** The `data_engineer` agent does NOT execute scripts. It passes control to `executor`, which returns execution results. The orchestrator may route back to `data_engineer` with `state.v4_refinement_context` populated — this is the generate→execute→analyze→refine loop.

---

### 3.2 Execution Agent (`executor`)

**File:** `src/cardiomas/agents/executor.py`

**Purpose:** Runs scripts step-by-step. Captures all stdout, stderr, exit codes. Reads generated files as text. Stores everything in state. Verifies correctness after each script.

**Inputs (from `GraphState`):**
- `state.v4_generated_scripts` — scripts to run (in order by script name)
- `state.v4_pipeline_phase` — `"subset_validation"` or `"full_run"`
- `state.v4_output_dir` — working directory

**Outputs (to `GraphState`):**
- `state.v4_execution_results` — `list[ExecutionResult]`
- `state.v4_subset_validated` — `bool`
- `state.v4_execution_summary` — aggregated text summary for LLM context
- `state.v4_generated_files` — `dict[str, str]` — {filename: text_content} for key output files
- `state.errors` — populated if any script fails after max retries

**Agent Function Signature:**
```python
def executor_agent(state: GraphState) -> GraphState:
    ...
```

**Execution Loop (per script):**
```python
for script_name in sorted(state.v4_generated_scripts.keys()):
    script_record = state.v4_generated_scripts[script_name]

    # 1. Run the script
    result = _run_script(script_record.path, timeout=script_record.timeout)

    # 2. Capture outputs
    exec_result = ExecutionResult(
        script_name=script_name,
        exit_code=result["exit_code"],
        stdout=result["stdout"][:8000],
        stderr=result["stderr"][:2000],
        duration_seconds=result["duration"],
        generated_files=_collect_generated_files(script_record.output_dir),
    )

    # 3. Verify (script-specific checks)
    verification = _verify_script_output(script_name, exec_result)
    exec_result.verification_passed = verification.passed
    exec_result.verification_notes = verification.notes

    state.v4_execution_results.append(exec_result)

    # 4. On failure: populate refinement context, break
    if not verification.passed or exec_result.exit_code != 0:
        state.v4_refinement_context = RefinementContext(
            failed_script=script_name,
            error_message=exec_result.stderr,
            stdout_excerpt=exec_result.stdout[:2000],
            suggested_fix="",  # LLM will fill in
        )
        state.errors.append(f"executor: script '{script_name}' failed: {exec_result.stderr[:200]}")
        break
```

**Subset Validation Check:**
After all subset scripts pass, the executor sets `state.v4_subset_validated = True` and populates `state.v4_execution_summary` with:
- Total records found (from `00_explore_structure.py` stdout)
- Column schema (from `01_extract_metadata.py` stdout)
- Label distribution summary (from `02_compute_statistics.py` output)
- Subset split sizes (from `03_generate_splits_subset.py` output)

This summary becomes the context for the `analysis` agent — **the analysis agent never sees raw file content**.

**File Reading (Generated Files Only):**
```python
def _collect_generated_files(output_dir: str) -> dict[str, str]:
    """Read text content of files generated by the script — not dataset files."""
    result = {}
    output_path = Path(output_dir)
    for f in output_path.glob("*.csv"):
        if f.stat().st_size < 100_000:  # 100KB cap
            result[f.name] = f.read_text()
    for f in output_path.glob("*.json"):
        if f.stat().st_size < 500_000:  # 500KB cap
            result[f.name] = f.read_text()
    return result
```

**No LLM calls in executor** — this agent is purely deterministic execution + output capture.

---

### 3.3 ECG Statistics Agent (`ecg_stats`)

**File:** `src/cardiomas/agents/ecg_stats.py`

**Purpose:** ECG-domain expert. Generates scripts for clinical-grade statistical analysis. Produces CSV summaries, publication-ready tables, and plots. Operates exclusively on script generation — all execution is delegated to `executor`.

**Inputs (from `GraphState`):**
- `state.v4_execution_summary` — field schema and basic stats from executor
- `state.analysis_report` — `AnalysisOutput` from analysis agent
- `state.v4_output_dir` — output directory

**Outputs (to `GraphState`):**
- `state.v4_generated_scripts` — augmented with ECG stat scripts
- `state.v4_pipeline_phase` — set to `"ecg_stats_run"`

**Agent Function Signature:**
```python
def ecg_stats_agent(state: GraphState) -> GraphState:
    ...
```

**Scripts Generated:**

| Script | ECG Domain | Output Files |
|---|---|---|
| `10_class_distribution.py` | Label frequency analysis; handles multi-label SCP codes | `class_dist.csv`, `class_dist.png` |
| `11_per_lead_statistics.py` | Amplitude range, mean, std per lead (reads .hea headers only — no signal data) | `lead_stats.csv` |
| `12_signal_quality.py` | Record completeness check (counts records per patient, flags duplicates, counts signals) | `quality_report.csv` |
| `13_clinical_plausibility.py` | Checks for age outliers (0–120), HR range, known SCP code validity | `clinical_flags.csv` |
| `14_publication_table.py` | Aggregates all CSV outputs into Table 1 markdown + LaTeX | `table1.md`, `table1.tex` |

**ECG Expert LLM Skill File:** `src/cardiomas/skills/ecg_stats.md`

The skill file encodes ECG domain knowledge:
- Standard ECG lead names (I, II, III, aVR, aVF, aVL, V1–V6)
- Known SCP code families (MI codes: `STEMI`, `NSTEMI`; rhythm codes: `SR`, `AFIB`)
- Clinical plausibility ranges: HR 30–250 bpm, PR 120–200 ms, QRS 60–120 ms
- Multi-label handling: `scp_codes` is a dict in PTB-XL; `label` in CPSC is integer

**Script Constraints (enforced):**
- Scripts read ONLY from generated CSV outputs and metadata files (not raw signal files)
- For `11_per_lead_statistics.py`: reads `.hea` header lines as text (`open(hea_file).readlines()`) — not `wfdb.rdheader()` — so it works without the wfdb library and does not load signals
- All plots saved to disk, not displayed interactively

---

## 4. Workflow Graph Topology

### New Graph Nodes

```python
_WORKER_AGENTS_V4 = [
    "nl_requirement",
    "discovery",
    "paper",
    "data_engineer",   # NEW — replaces analysis for file discovery
    "executor",        # NEW — runs scripts, captures output
    "analysis",        # KEPT but refactored — reads executor output only
    "ecg_stats",       # NEW — generates ECG statistical scripts
    "splitter",        # KEPT but refactored — reads executor splits.json
    "security",        # KEPT — minor update
    "coder",           # KEPT — generates reproducibility scripts only
    "publisher",       # KEPT — unchanged
]

_ALL_TARGETS_V4 = _WORKER_AGENTS_V4 + [
    "return_existing",
    "approval_gate",   # NEW — human-in-the-loop node
    "end_saved",
    "end_with_error",
]
```

### Updated Graph Construction in `workflow.py`

```python
def build_workflow():
    graph = StateGraph(dict)

    # ── New V4 nodes ──────────────────────────────────────────────────────
    graph.add_node("data_engineer", _make_worker_wrapper("data_engineer", data_engineer_agent))
    graph.add_node("executor",      _make_worker_wrapper("executor",      executor_agent))
    graph.add_node("ecg_stats",     _make_worker_wrapper("ecg_stats",     ecg_stats_agent))
    graph.add_node("approval_gate", _wrap_simple(_passthrough_approval_gate))

    # ── Existing nodes (unchanged registration) ───────────────────────────
    graph.add_node("nl_requirement", ...)
    graph.add_node("discovery", ...)
    graph.add_node("paper", ...)
    graph.add_node("analysis", ...)
    graph.add_node("splitter", ...)
    graph.add_node("security", ...)
    graph.add_node("coder", ...)
    graph.add_node("publisher", ...)

    # ── Conditional routing from executor ─────────────────────────────────
    # executor may need to route back to data_engineer (refinement loop)
    # or forward to analysis (success)
    graph.add_conditional_edges(
        "executor",
        _route_from_executor,  # checks state.v4_refinement_context, phase, validated flag
        {
            "data_engineer": "data_engineer",   # script failed → refine
            "analysis": "analysis",              # subset validated → analysis
            "ecg_stats": "ecg_stats",            # after full run → ecg_stats
            "orchestrator": "orchestrator",      # orchestrator-driven routing
        },
    )

    # ── Conditional approval gate ─────────────────────────────────────────
    graph.add_conditional_edges(
        "approval_gate",
        _route_from_approval_gate,  # checks state.v4_approval_status
        {
            "executor": "executor",              # approved → start full run
            "end_saved": "end_saved",            # rejected / timeout → end
            "orchestrator": "orchestrator",      # pending (should not happen in compiled graph)
        },
    )
```

### Routing Table (Extended Orchestrator `_route()`)

```python
def _route(state: GraphState, last: str) -> tuple[str, str]:
    # V3 routes preserved...
    if last == "nl_requirement":
        return "discovery", "..."
    if last == "discovery":
        ...

    # V4 new routes:
    if last == "paper":
        return "data_engineer", "Paper analysis done — generating dataset exploration scripts."

    if last == "data_engineer":
        return "executor", "Scripts generated — executing on subset for validation."

    if last == "analysis":
        # After analysis reads executor output:
        # If subset was just validated → go to approval gate
        if state.v4_pipeline_phase == "subset_validation" and state.v4_subset_validated:
            return "approval_gate", "Subset validated — requesting human approval before full run."
        # If full run already approved → go to ecg_stats
        if state.v4_pipeline_phase == "full_run":
            return "ecg_stats", "Analysis complete — generating ECG statistical scripts."
        return "end_with_error", "Unexpected pipeline phase in analysis."

    if last == "ecg_stats":
        return "executor", "ECG stat scripts generated — executing full-dataset stats."

    if last == "splitter":
        return "security", "Splits generated from executor output — running security audit."

    if last == "security":
        if state.security_audit and not state.security_audit.passed:
            return "end_with_error", "Security audit FAILED."
        return "coder", "Security passed — generating reproducibility scripts."

    if last == "coder":
        if opts.push_to_hf:
            return "publisher", "Scripts verified — publishing."
        return "end_saved", "All complete — saved locally."

    if last == "publisher":
        return "end_saved", "Published."
```

### Executor Internal Routing (`_route_from_executor`)

This is a **conditional edge function** (not going through the orchestrator hub) to keep the refinement loop tight:

```python
def _route_from_executor(state_dict: dict) -> str:
    state = _dict_to_state(state_dict)

    # Refinement: script failed, send back to data_engineer
    if state.v4_refinement_context and state.errors:
        retry_key = f"executor_refinement_{state.v4_refinement_context.failed_script}"
        count = state.retry_counts.get(retry_key, 0)
        if count < 2:  # max 2 refinement rounds
            return "data_engineer"
        return "orchestrator"  # orchestrator will route to end_with_error

    # Phase-based routing
    if state.v4_pipeline_phase == "subset_validation" and state.v4_subset_validated:
        return "orchestrator"  # orchestrator → analysis → approval_gate

    if state.v4_pipeline_phase == "ecg_stats_run":
        return "orchestrator"  # orchestrator → splitter

    return "orchestrator"
```

### Approval Gate Node

```python
def _passthrough_approval_gate(state: GraphState) -> GraphState:
    """Human-in-the-loop gate. Checks state.v4_approval_status."""
    from cardiomas.schemas.state import LogEntry

    status = state.v4_approval_status

    if status == "approved":
        # Transition to full run phase
        state.v4_pipeline_phase = "full_run"
        state.execution_log.append(LogEntry(
            agent="approval_gate",
            action="approved",
            detail="User approved subset validation — proceeding to full run.",
        ))
        # The graph routes executor back into full-run mode on next invocation
        state.next_agent = "executor"

    elif status == "rejected":
        state.execution_log.append(LogEntry(
            agent="approval_gate",
            action="rejected",
            detail="User rejected or did not approve — saving subset results only.",
        ))
        state.next_agent = "end_saved"

    else:
        # "pending" — auto-approve if --auto-approve flag set
        if state.user_options.v4_auto_approve:
            state.v4_approval_status = "approved"
            state.v4_pipeline_phase = "full_run"
            state.next_agent = "executor"
        else:
            # Save checkpoint and wait for human input
            _print_approval_request(state)
            state.next_agent = "end_saved"  # save and exit; human re-runs with --approve

    return state
```

### CLI: Human Approval Flow

```bash
# First run (generates subset validation)
cardiomas analyze /data/ptb-xl --local-path /data/ptb-xl

# Output shows:
# Subset validation complete. Review results in output/ptb-xl/v4/
# Re-run with --approve to proceed to full dataset processing.

# Second run (with approval)
cardiomas analyze /data/ptb-xl --local-path /data/ptb-xl --approve
# Loads checkpoint, sets v4_approval_status="approved", resumes from approval_gate
```

---

## 5. New State Schema Fields

All new fields are added to `GraphState` in `src/cardiomas/schemas/state.py` under a clearly marked `# ── V4 indirect execution ──` section.

### New Pydantic Models

**In `src/cardiomas/schemas/state.py`:**

```python
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
    # {filename → text_content} for script-generated files only
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
```

### New `GraphState` Fields

```python
class GraphState(BaseModel):
    # ... all existing fields unchanged ...

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
    # {filename → text_content} for machine-readable generated outputs

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
    # {script_name → number_of_refinement_rounds}

    # ECG statistics outputs
    v4_ecg_stats_dir: str = ""       # path to directory with all stat outputs
    v4_ecg_stats_scripts: dict[str, ScriptRecord] = Field(default_factory=dict)
```

### New `UserOptions` Fields

```python
class UserOptions(BaseModel):
    # ... all existing fields unchanged ...

    # ── V4 options ─────────────────────────────────────────────────────────
    v4_auto_approve: bool = False    # skip human approval gate
    v4_subset_size: int = 100        # override subset size
    v4_max_refinements: int = 2      # max refinement rounds per script
    v4_skip_ecg_stats: bool = False  # skip ECG statistics phase
    v4_plot_format: str = "png"      # "png" | "pdf" | "svg"
```

---

## 6. New Tools and Modules

### 6.1 Enhanced Script Runner Tool

**File:** `src/cardiomas/tools/code_tools.py` — add new tool

```python
@tool
def execute_script_with_env(
    script_path: str,
    env_vars: dict[str, str] | None = None,
    timeout: int = 300,
    working_dir: str = "",
    capture_files: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a Python script with optional environment variables and file capture.

    Args:
        script_path: Absolute path to script
        env_vars: Additional environment variables (e.g. DATA_PATH overrides)
        timeout: Execution timeout in seconds (default 300)
        working_dir: Working directory for script execution
        capture_files: List of filenames to read after execution
            (relative to working_dir). Content returned as text.

    Returns dict with:
        exit_code, stdout, stderr, duration_seconds,
        captured_files: {filename: text_content}
    """
```

### 6.2 Script Output Directory Manager

**File:** `src/cardiomas/tools/v4_output_tools.py` — new file

```python
@tool
def setup_v4_output_dir(dataset_name: str, base_dir: str = "output") -> dict[str, Any]:
    """Create the V4 structured output directory tree.

    Creates:
        {base_dir}/{dataset_name}/v4/
            scripts/subset/     ← subset validation scripts
            scripts/full/       ← full-run scripts
            scripts/ecg_stats/  ← ECG statistics scripts
            outputs/subset/     ← subset script outputs
            outputs/full/       ← full-run script outputs
            outputs/ecg_stats/  ← ECG stat outputs
            logs/               ← execution logs

    Returns: dict with all directory paths.
    """

@tool
def read_generated_file(file_path: str, max_bytes: int = 500_000) -> dict[str, Any]:
    """Read a script-generated output file (CSV, JSON, or text).

    ONLY reads files within the V4 output directory structure.
    Will refuse to read files with ECG data suffixes (.dat, .h5, .hdf5, .edf, .npy).
    """

@tool
def list_generated_files(output_dir: str) -> dict[str, Any]:
    """List all files generated in a V4 output directory."""

@tool
def write_execution_log(
    log_dir: str,
    script_name: str,
    execution_result: dict,
) -> dict[str, Any]:
    """Persist an execution result to a structured JSON log file."""
```

### 6.3 Script Verification Engine

**File:** `src/cardiomas/tools/script_verification.py` — new file

```python
class ScriptVerification(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    notes: str = ""

def verify_explore_output(stdout: str) -> ScriptVerification:
    """Verify 00_explore_structure.py output contains expected keys."""
    # Checks: TOTAL_FILES=N, FORMATS={...}, ROOT=...

def verify_metadata_output(stdout: str) -> ScriptVerification:
    """Verify 01_extract_metadata.py output contains column schema."""
    # Checks: COLUMNS=[...], DTYPES={...}, SAMPLE_ROWS=N

def verify_stats_output(
    stdout: str,
    generated_files: dict[str, str],
) -> ScriptVerification:
    """Verify 02_compute_statistics.py produced stats.csv."""
    # Checks: stats.csv exists, has rows, LABEL_FIELD= in stdout

def verify_subset_splits_output(
    stdout: str,
    generated_files: dict[str, str],
    expected_subset_size: int,
) -> ScriptVerification:
    """Verify 03_generate_splits_subset.py produced valid splits.json."""
    # Checks: splits_subset.json exists and parseable, sizes sum to <= subset_size

def verify_ecg_stats_output(
    script_name: str,
    stdout: str,
    generated_files: dict[str, str],
) -> ScriptVerification:
    """Generic verifier for ECG statistics scripts."""
    # Checks expected output files exist for each script
```

### 6.4 New Skill Files

**`src/cardiomas/skills/data_engineer.md`** — complete LLM system prompt:
```markdown
# Data Engineering Agent (V4)

You are a Python data engineering expert. Your task is to generate
self-contained Python scripts that EXPLORE ECG dataset structure WITHOUT
loading raw signal data. The scripts run sequentially; each depends only
on constants at the top and writes structured output.

## Absolute Rules
1. Scripts MUST NOT load raw ECG signals (.dat, .h5 signal arrays, .edf signals).
   They may read .hea header lines as text (open+readlines), CSV metadata, and
   filenames — nothing else.
2. All parameters as ALL_CAPS constants at the top.
3. All output to OUTPUT_DIR constant (never to DATASET_PATH).
4. Print KEY=VALUE or structured JSON to stdout for machine parsing.
5. stdlib + pandas + numpy only — no cardiomas, no wfdb, no h5py imports.
6. Graceful errors: catch all exceptions, print WARNING: message, exit 0.
7. Complete docstring header: script name, purpose, inputs, outputs, CardioMAS V4.

## Output Directory Layout
OUTPUT_DIR/
    stats.csv              (from 02_compute_statistics.py)
    class_dist.csv         (from 10_class_distribution.py)
    splits_subset.json     (from 03_generate_splits_subset.py)
    splits.json            (from 04_generate_splits_full.py)

## Script Naming
NN_descriptive_name.py where NN is 00-09 for exploration,
10-19 for ECG stats, 04 for final splits.
```

**`src/cardiomas/skills/ecg_stats.md`** — ECG domain expert prompt with known SCP code families, lead naming conventions, plausibility ranges.

**`src/cardiomas/skills/executor.md`** — not needed (executor has no LLM calls).

---

## 7. Changes to Existing Agents

### 7.1 `analysis_agent` — Refactored

**File:** `src/cardiomas/agents/analysis.py`

**Remove:**
- All direct calls to `read_csv_metadata`, `compute_statistics`, `list_dataset_files`
- All calls to `DatasetMapper` / `_build_dataset_map()`
- `_find_data_root()` helper
- All `pd.read_csv()` or file-open operations

**Add:**
- Reads `state.v4_execution_summary` as the primary context source
- Reads `state.v4_generated_files` for CSV content (script-generated outputs only)
- Parses structured KEY=VALUE stdout from executor results

**New function signature:**
```python
def analysis_agent(state: GraphState) -> GraphState:
    """Consume executor script outputs and produce AnalysisOutput via LLM.
    Does NOT read dataset files directly."""
```

**New context assembly:**
```python
def _build_analysis_context(state: GraphState) -> str:
    """Build LLM context purely from executor outputs."""
    parts = []

    # 1. Execution summary (text aggregated by executor)
    if state.v4_execution_summary:
        parts.append(f"## Script Execution Summary\n{state.v4_execution_summary}")

    # 2. Key stdout excerpts per script
    for result in state.v4_execution_results:
        parts.append(
            f"### Script: {result.script_name}\n"
            f"Exit code: {result.exit_code}\n"
            f"Output:\n{result.stdout[:2000]}\n"
        )

    # 3. Generated CSV content (stats.csv, class_dist.csv)
    for fname, content in state.v4_generated_files.items():
        if fname.endswith(".csv"):
            parts.append(f"### Generated file: {fname}\n{content[:3000]}\n")

    return "\n\n".join(parts)
```

**DatasetMap removal:** The `analysis_agent` no longer builds a `DatasetMap`. Instead, it populates `state.analysis_report` from parsed executor stdout. The `state.dataset_map` field is left `None` in V4.

### 7.2 `splitter_agent` — Refactored

**File:** `src/cardiomas/agents/splitter.py`

**Remove:**
- `_load_record_ids()` — entire function (reads CSVs directly)
- `pd.read_csv()` calls
- DatasetMapper integration
- Fallback to `find_dataset_files.invoke()`

**Add:**
- `_load_ids_from_executor_output()` — reads `splits.json` generated by `04_generate_splits_full.py` (the full-run split script)
- If `splits.json` present in `state.v4_generated_files`, deserialize and use directly

**New `_load_record_ids()`:**
```python
def _load_record_ids_v4(state: GraphState) -> list[str]:
    """Load record IDs exclusively from executor-generated splits.json."""
    # Priority 1: splits.json in v4_generated_files
    splits_json = state.v4_generated_files.get("splits.json", "")
    if splits_json:
        try:
            data = json.loads(splits_json)
            all_ids = []
            for split_ids in data.get("splits", {}).values():
                all_ids.extend(split_ids)
            return sorted(set(all_ids))
        except json.JSONDecodeError as e:
            state.errors.append(f"splitter: could not parse splits.json from executor: {e}")

    # Priority 2: splits_subset.json (only for subset phase — should not reach splitter in this case)
    # Priority 3: Hard failure — no synthetic fallback
    raise SplitDataMissingError(
        "No executor-generated splits.json available. "
        "Ensure executor ran 04_generate_splits_full.py successfully."
    )
```

**New exception:**
```python
class SplitDataMissingError(RuntimeError):
    """Raised when no executor-generated splits are available."""
```

The existing `SplitIntegrityError` and `deterministic_split()` call remain unchanged.

### 7.3 `orchestrator_agent` — Updated Routing

**File:** `src/cardiomas/agents/orchestrator.py`

**Changes:**
1. `_route()` extended with V4 routes (see Section 4)
2. `_WORKER_AGENTS` list in `workflow.py` updated (see Section 4)
3. Config module updated to register V4 agent LLM overrides
4. Agent LLM override env vars: `AGENT_LLM_DATA_ENGINEER`, `AGENT_LLM_EXECUTOR` (no LLM), `AGENT_LLM_ECG_STATS`

### 7.4 `coder_agent` — Scoped Down

**File:** `src/cardiomas/agents/coder.py`

**Remove:**
- `generate_splits.py` generation (now done by `data_engineer`)
- `explore_dataset.py` generation (now done by `data_engineer`)

**Keep:**
- `verify_splits.py` generation — standalone reproducibility verifier
- Execution of `verify_splits.py` for SHA-256 verification
- Script verification logic via `verify_script_sha256()`

**The coder agent's new sole purpose** is to generate the `verify_splits.py` reproducibility checker and verify it passes. It reads the executor-produced `splits.json` as its reference, not a self-generated one.

**New context for `verify_splits.py` generation:**
```python
context = (
    f"Dataset: {dataset_name}\n"
    f"Reference splits.json: {v4_splits_path}\n"  # path to executor-generated file
    f"Seed: {seed}\n"
    f"Strategy: deterministic\n"
    f"Note: This script verifies the splits produced by the V4 pipeline.\n"
    f"It must read splits.json from the V4 output directory.\n"
)
```

### 7.5 `security_agent` — Minor Update

**File:** `src/cardiomas/agents/security.py`

**Change:** `_get_patient_mapping()` updated to read from `state.v4_generated_files` instead of `state.dataset_map`. Specifically, if `patient_map.json` is listed in generated files (produced by `01_extract_metadata.py`), parse it.

```python
def _get_patient_mapping_v4(state: GraphState) -> dict[str, list[str]] | None:
    patient_map_json = state.v4_generated_files.get("patient_map.json", "")
    if patient_map_json:
        try:
            return json.loads(patient_map_json)
        except Exception:
            pass
    # Fall back to V3 dataset_map if available (backwards compat)
    return _get_patient_mapping_v3(state)
```

### 7.6 `config.py` — New Agent Registrations

```python
for _agent in (
    "orchestrator", "nl_requirement", "discovery", "paper",
    "data_engineer", "executor", "analysis", "ecg_stats",  # V4 additions
    "splitter", "security", "coder", "publisher",
):
    ...
```

---

## 8. Implementation Phases

### Phase V4-1: State Schema + New Tools (Build first — no agent changes)

**Goal:** Lay the Pydantic foundation and tool infrastructure before touching agents.

**Deliverables:**
1. Add all new `GraphState` fields (`ScriptRecord`, `ExecutionResult`, `RefinementContext`, `ApprovalSummary`, `v4_*` fields) to `state.py`
2. Add `v4_auto_approve`, `v4_subset_size`, etc. to `UserOptions`
3. Create `src/cardiomas/tools/v4_output_tools.py` with all 4 tools
4. Add `execute_script_with_env` to `code_tools.py`
5. Create `src/cardiomas/tools/script_verification.py` with all verification functions
6. Create `src/cardiomas/skills/data_engineer.md`
7. Create `src/cardiomas/skills/ecg_stats.md`
8. Unit tests for all new schema models and tools

**Test coverage:** `tests/test_v4_schemas.py`, `tests/test_v4_tools.py`

---

### Phase V4-2: Data Engineering Agent + Executor Agent

**Goal:** Get scripts generated and executed; get executor output flowing into state.

**Deliverables:**
1. `src/cardiomas/agents/data_engineer.py` — full implementation
2. `src/cardiomas/agents/executor.py` — full implementation (no LLM)
3. Update `config.py` — register `data_engineer`, `executor` agent names
4. Add `data_engineer` and `executor` nodes to `workflow.py`
5. Add `_route_from_executor()` conditional edge function
6. Add V4 routes to `orchestrator._route()`
7. Wire `data_engineer → executor` edge pair
8. Unit tests: mock LLM, verify script generation and execution capture

**Test coverage:** `tests/test_data_engineer.py`, `tests/test_executor.py`

---

### Phase V4-3: Analysis Agent Refactor

**Goal:** Cut `analysis_agent` completely off from file system access.

**Deliverables:**
1. Refactor `src/cardiomas/agents/analysis.py` — remove all direct file reads
2. Implement `_build_analysis_context(state)` that reads only from `state.v4_*` fields
3. Remove `_build_dataset_map()`, `_find_data_root()`, `_verify_analysis_output()` calls
4. Remove imports: `DatasetMapper`, `read_csv_metadata`, `compute_statistics`, `list_dataset_files`
5. Verify `state.analysis_report` still populated correctly from LLM output
6. Refactor `splitter_agent` — implement `_load_record_ids_v4()`, add `SplitDataMissingError`
7. Update `security_agent` — `_get_patient_mapping_v4()`

**Test coverage:** `tests/test_v4_analysis.py` (mock `v4_execution_results` in state)

---

### Phase V4-4: Approval Gate + Full Run Routing

**Goal:** Implement the human-in-the-loop approval flow.

**Deliverables:**
1. Add `_passthrough_approval_gate()` node to `workflow.py`
2. Add `approval_gate` to `_ALL_TARGETS`
3. Add `_route_from_approval_gate()` conditional edge
4. Add `--approve` and `--auto-approve` flags to `cli/main.py`
5. Implement `_print_approval_request()` — pretty-print subset summary for human review
6. Implement `resume_pipeline()` update — supports resuming from approval gate
7. Add `v4_auto_approve` to `UserOptions` and wire through CLI
8. Add data_engineer script for full-run: `04_generate_splits_full.py`

**Test coverage:** `tests/test_v4_approval.py`

---

### Phase V4-5: ECG Statistics Agent

**Goal:** Add clinical-grade ECG statistical analysis as a pipeline phase.

**Deliverables:**
1. `src/cardiomas/agents/ecg_stats.py` — full implementation
2. Add ECG stats scripts to `data_engineer` prompts (or as separate agent — separate is cleaner)
3. Register `ecg_stats` node in `workflow.py`
4. Add V4 routing: `after analysis (full_run) → ecg_stats → executor → splitter`
5. Add `--skip-ecg-stats` CLI flag
6. Add `v4_ecg_stats_dir` populated by executor after ECG stats run

**Test coverage:** `tests/test_ecg_stats.py`

---

### Phase V4-6: Coder Agent Scope-Down + Integration

**Goal:** Scope coder agent to reproducibility verification only; end-to-end integration.

**Deliverables:**
1. Refactor `coder_agent` — remove `generate_splits.py` and `explore_dataset.py` generation
2. Update coder to generate `verify_splits.py` referencing V4 executor-produced `splits.json`
3. Update coder's `execute_script_with_env` call to point at V4 split file
4. Full end-to-end integration test with a synthetic ECG dataset fixture
5. Update `CLAUDE.md` with V4 architecture
6. Update `cli/main.py` with all V4 flags

**Test coverage:** `tests/test_v4_integration.py`

---

## 9. File-Level Change List

### New Files

| File | Type | Phase |
|---|---|---|
| `src/cardiomas/agents/data_engineer.py` | New agent | V4-2 |
| `src/cardiomas/agents/executor.py` | New agent (no LLM) | V4-2 |
| `src/cardiomas/agents/ecg_stats.py` | New agent | V4-5 |
| `src/cardiomas/tools/v4_output_tools.py` | New tools module | V4-1 |
| `src/cardiomas/tools/script_verification.py` | New verification module | V4-1 |
| `src/cardiomas/skills/data_engineer.md` | New skill file | V4-1 |
| `src/cardiomas/skills/ecg_stats.md` | New skill file | V4-1 |
| `tests/test_v4_schemas.py` | New tests | V4-1 |
| `tests/test_v4_tools.py` | New tests | V4-1 |
| `tests/test_data_engineer.py` | New tests | V4-2 |
| `tests/test_executor.py` | New tests | V4-2 |
| `tests/test_v4_analysis.py` | New tests | V4-3 |
| `tests/test_v4_approval.py` | New tests | V4-4 |
| `tests/test_ecg_stats.py` | New tests | V4-5 |
| `tests/test_v4_integration.py` | End-to-end test | V4-6 |
| `tests/fixtures/synthetic_ecg/` | Test fixtures (CSV + .hea stubs) | V4-2 |

### Modified Files

| File | Change Summary | Phase |
|---|---|---|
| `src/cardiomas/schemas/state.py` | Add `ScriptRecord`, `ExecutionResult`, `RefinementContext`, `ApprovalSummary`, all `v4_*` fields to `GraphState`; add `v4_*` fields to `UserOptions` | V4-1 |
| `src/cardiomas/tools/code_tools.py` | Add `execute_script_with_env` tool | V4-1 |
| `src/cardiomas/graph/workflow.py` | Add `data_engineer`, `executor`, `ecg_stats`, `approval_gate` nodes; add `_route_from_executor()`, `_route_from_approval_gate()` conditional edges; update `_WORKER_AGENTS_V4` and `_ALL_TARGETS_V4`; update `build_workflow()` | V4-2 |
| `src/cardiomas/agents/orchestrator.py` | Extend `_route()` with V4 routes; update skills/orchestrator.md routing table | V4-2 |
| `src/cardiomas/agents/analysis.py` | Remove all direct file access; add `_build_analysis_context(state)`; remove DatasetMapper integration | V4-3 |
| `src/cardiomas/agents/splitter.py` | Replace `_load_record_ids()` with `_load_record_ids_v4()`; add `SplitDataMissingError`; remove all CSV reads | V4-3 |
| `src/cardiomas/agents/security.py` | Add `_get_patient_mapping_v4()` that reads from `v4_generated_files` | V4-3 |
| `src/cardiomas/agents/coder.py` | Remove `generate_splits.py` and `explore_dataset.py` generation; scope to `verify_splits.py` only | V4-6 |
| `src/cardiomas/config.py` | Register `data_engineer`, `executor`, `ecg_stats` in agent LLM override loop | V4-2 |
| `src/cardiomas/cli/main.py` | Add `--approve`, `--auto-approve`, `--v4-subset-size`, `--skip-ecg-stats` flags; update output display for V4 artifacts | V4-4 |
| `src/cardiomas/skills/orchestrator.md` | Update routing table to include V4 agents and approval gate | V4-2 |
| `pyproject.toml` | Add `matplotlib` to core dependencies (needed for plot scripts); version bump for V4 | V4-6 |
| `CLAUDE.md` | Update architecture section with V4 diagram | V4-6 |

### Files Removed or Left Unchanged

| File | Status | Reason |
|---|---|---|
| `src/cardiomas/mappers/` (entire module) | **Left in place, unused by V4** | `analysis_agent` no longer calls it; can be deprecated in V5 |
| `src/cardiomas/agents/verification.py` | **Left in place, used by executor** | `verify_split_integrity()` called from executor; `verify_script_sha256()` still used by coder |
| `src/cardiomas/agents/nl_requirement.py` | Unchanged | V4 does not affect NL requirement parsing |
| `src/cardiomas/agents/discovery.py` | Unchanged | V4 does not affect dataset discovery |
| `src/cardiomas/agents/paper.py` | Unchanged | V4 does not affect paper analysis |
| `src/cardiomas/agents/publisher.py` | Unchanged | V4 does not affect publishing |
| `src/cardiomas/rag/` | Unchanged | Paper RAG (Phase 3) unaffected |
| `src/cardiomas/splitters/strategies.py` | Unchanged | `deterministic_split()` still used |
| `src/cardiomas/tools/research_tools.py` | Unchanged | |
| `src/cardiomas/tools/publishing_tools.py` | Unchanged | |
| `src/cardiomas/tools/security_tools.py` | Unchanged | |
| `src/cardiomas/tools/split_tools.py` | Unchanged | |
| `src/cardiomas/tools/shell_tools.py` | Unchanged | Used by data_engineer for context, not for direct reads |
| `src/cardiomas/tools/data_tools.py` | **Effectively deprecated** | No agent calls these tools in V4; leave in place for V5 removal |

---

## 10. Testing Strategy

### Unit Tests by Phase

**Phase V4-1 (`test_v4_schemas.py`):**
- `ScriptRecord` Pydantic validation
- `ExecutionResult` serialisation round-trip
- `v4_*` fields default values in `GraphState`
- `ApprovalSummary` construction

**Phase V4-1 (`test_v4_tools.py`):**
- `setup_v4_output_dir` creates correct directory tree
- `read_generated_file` refuses ECG signal extensions
- `execute_script_with_env` captures stdout and generated files correctly
- All 5 verification functions in `script_verification.py`

**Phase V4-2 (`test_data_engineer.py`):**
```python
def test_data_engineer_generates_four_scripts(tmp_path):
    # Mock LLM returning valid Python code
    # Assert 4 ScriptRecords in state.v4_generated_scripts
    # Assert each script file exists on disk
    # Assert no cardiomas import in generated code

def test_data_engineer_refinement_context_propagated(tmp_path):
    # Populate state.v4_refinement_context
    # Assert LLM receives refinement context in prompt
```

**Phase V4-2 (`test_executor.py`):**
```python
def test_executor_runs_scripts_in_order(tmp_path):
    # Create simple test scripts that write to OUTPUT_DIR
    # Assert execution results in correct order
    # Assert generated_files captured

def test_executor_sets_refinement_on_failure(tmp_path):
    # Script that exits with code 1
    # Assert state.v4_refinement_context populated
    # Assert state.errors contains failure message

def test_executor_marks_subset_validated(tmp_path):
    # All 4 subset scripts pass
    # Assert state.v4_subset_validated == True
    # Assert state.v4_execution_summary non-empty
```

**Phase V4-3 (`test_v4_analysis.py`):**
```python
def test_analysis_uses_only_executor_output(mock_state):
    # Populate state.v4_execution_results with mock data
    # Run analysis_agent with mock LLM
    # Assert no file I/O occurred (mock Path.open)
    # Assert analysis_report populated correctly

def test_splitter_reads_from_v4_generated_files(tmp_path):
    # Populate state.v4_generated_files with valid splits.json
    # Run splitter_agent
    # Assert proposed_splits populated from that JSON
    # Assert no pd.read_csv called
```

**Phase V4-4 (`test_v4_approval.py`):**
```python
def test_approval_gate_auto_approve():
    # state.user_options.v4_auto_approve = True
    # Assert next_agent = "executor"
    # Assert v4_pipeline_phase = "full_run"

def test_approval_gate_pending_ends_pipeline():
    # state.v4_approval_status = "pending", auto_approve=False
    # Assert next_agent = "end_saved"
```

**Phase V4-6 (`test_v4_integration.py`):**
```python
def test_full_v4_pipeline_with_synthetic_dataset(tmp_path):
    """End-to-end test using tests/fixtures/synthetic_ecg/."""
    # Create synthetic CSV with 500 records
    # Run pipeline with v4_auto_approve=True, v4_subset_size=50
    # Assert all 4 phases complete
    # Assert splits.json exists in v4 output dir
    # Assert no direct file reads occurred in analysis or splitter
    # Assert verify_splits.py generated and passes
```

### Backward Compatibility

- All existing 26 tests in `test_cli.py`, `test_security.py`, `test_splitters.py`, `test_v2.py` must continue to pass without modification
- `UserOptions.v4_*` fields all have defaults — existing code paths unaffected
- `GraphState.v4_*` fields all have defaults — existing checkpoint loading unaffected
- V4 is only activated when `local_path` is provided and dataset files exist; discovery-only (URL) mode is unaffected

---

## Summary Decision Table

| V3 Agent | V4 Fate | Reason |
|---|---|---|
| `orchestrator` | Extended routing | Needs V4 route entries |
| `nl_requirement` | Unchanged | Not dataset-file related |
| `discovery` | Unchanged | Reads webpage, not files |
| `paper` | Unchanged | Reads PDF, not dataset |
| `analysis` | **Refactored** | Remove all direct file I/O |
| `splitter` | **Refactored** | Remove CSV reads, use executor output |
| `security` | Minor update | Add V4 patient map source |
| `coder` | **Scoped down** | Only generates verify_splits.py |
| `publisher` | Unchanged | Not file-access related |
| **`data_engineer`** | **New** | Script generation for exploration |
| **`executor`** | **New** | Script execution + output capture |
| **`ecg_stats`** | **New** | ECG-domain statistics scripts |
| **Approval gate** | **New node** | Human-in-the-loop checkpoint |

---

### Critical Files for Implementation

- `/work/vajira/DL2026/CardioMAS/src/cardiomas/schemas/state.py`
- `/work/vajira/DL2026/CardioMAS/src/cardiomas/graph/workflow.py`
- `/work/vajira/DL2026/CardioMAS/src/cardiomas/agents/analysis.py`
- `/work/vajira/DL2026/CardioMAS/src/cardiomas/agents/splitter.py`
- `/work/vajira/DL2026/CardioMAS/src/cardiomas/agents/orchestrator.py`
