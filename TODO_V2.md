# CardioMAS V2 — Development Plan

Branch: `dev/v2-dynamic-orchestrator` → merge to `master` when complete.

All features build on the existing latest architecture (`src/cardiomas/`).
Reproducibility remains the top-level non-negotiable constraint throughout.

---

## Summary of V2 Features

| # | Feature | Priority |
|---|---|---|
| 1 | Dynamic orchestrator (Option B) | High |
| 2 | Natural language requirement input agent | High |
| 3 | Coding agent — generate + execute analysis/split scripts | High |
| 4 | Per-agent LLM configuration | Medium |
| 5 | Full conversation + reasoning recorder | High |
| 6 | Verbose LLM name display | Low |
| 7 | Context compression for long sessions | Medium |

---

## 1. Dynamic Orchestrator (Option B)

### Problem with V1

The `orchestrator_agent` is only an entry node. Routing between agents is hardwired in `workflow.py` graph edges. The orchestrator does not inspect state and cannot skip or reorder agents based on what is actually available.

### V2 Behaviour

The orchestrator runs **between every agent** as a supervisor. After each agent completes, it evaluates the updated `GraphState` and decides what to run next, whether to retry, or whether to skip.

```
User input
    │
    ▼
┌─────────────────────────────────┐
│         ORCHESTRATOR            │  ◄── runs after every agent step
│  - reads GraphState             │
│  - decides next agent           │
│  - logs reasoning               │
│  - can skip, retry, or abort    │
└────────────┬────────────────────┘
             │ dispatches
    ┌────────┼──────────────────┐
    ▼        ▼                  ▼
  NLReq   Discovery   Analysis  ...
```

### Decision logic (must be documented in `skills/orchestrator.md`)

| State condition | Orchestrator decision |
|---|---|
| No URL, local path provided | Skip Discovery + Paper; go to Analysis |
| URL only, no local path | Run Discovery + Paper; skip Analysis |
| Paper found with official splits | Inform Splitter to use official splits |
| No local metadata files found | Warn user; use registry estimates for splits |
| Security audit failed | Halt; do not save or push |
| `push_to_hf=False` | Skip Publisher entirely |
| Any agent sets an error | Orchestrator decides: retry once or abort |

### Implementation

**Files to create/modify:**

- `agents/orchestrator.py` — rewrite as a true supervisor with a routing LLM call
- `graph/workflow.py` — change from fixed edge sequence to orchestrator-driven conditional edges
- `schemas/state.py` — add `next_agent: str`, `agent_skip_reasons: dict[str, str]`, `retry_counts: dict[str, int]`
- `skills/orchestrator.md` — full decision tree documented as LLM system prompt

**New `GraphState` fields:**

```python
next_agent: str = ""                        # orchestrator sets this after each step
agent_skip_reasons: dict[str, str] = {}    # {agent_name: reason_skipped}
retry_counts: dict[str, int] = {}          # {agent_name: n_retries}
orchestrator_reasoning: list[str] = []     # one entry per routing decision
```

**LangGraph pattern** — replace fixed edges with a single hub-and-spoke pattern:

```
[each agent] → orchestrator → [conditional edge to next agent or END]
```

The orchestrator makes one LLM call per routing step, writes its reasoning to
`orchestrator_reasoning`, and sets `next_agent`. Routing is then deterministic
from that field (no LLM non-determinism in the graph edges themselves).

---

## 2. Natural Language Requirement Input Agent

### Purpose

Allow users to describe their needs in plain English. A dedicated agent translates that into structured `UserOptions` before the pipeline starts.

### CLI usage

```bash
cardiomas analyze /data/ptb-xl/ \
  --requirement "I need patient-level splits stratified by diagnosis, \
                 with 70% training, 15% validation, 15% test. \
                 Exclude records with missing age. Seed 99."
```

### Python API usage

```python
mas.analyze(
    "/data/ptb-xl/",
    requirement="80/10/10 split, stratify by rhythm label, no paediatric patients"
)
```

### Agent design

**New agent:** `agents/nl_requirement.py`
**New skill:** `skills/nl_requirement.md`

The agent receives the raw natural language string and the current `DatasetInfo`
(column names, available fields) and returns a structured `ParsedRequirement`
Pydantic model:

```python
class ParsedRequirement(BaseModel):
    split_ratios: dict[str, float]
    stratify_by: str | None
    exclusion_filters: list[dict]      # e.g. [{"field": "age", "op": "notna"}]
    patient_level: bool
    seed: int | None
    notes: str                          # anything the agent could not parse
    raw_input: str                      # original user text (for reproducibility log)
    llm_reasoning: str                  # agent's explanation of its parsing
```

`ParsedRequirement` is stored in `GraphState` and passed to the Splitter and
Analysis agents. If `notes` is non-empty, the CLI warns the user that some
requirements could not be parsed.

**Reproducibility:** `raw_input` and `llm_reasoning` are saved into
`split_metadata.json` so splits can always be traced back to the original
human request.

### Implementation

- `schemas/state.py` — add `parsed_requirement: ParsedRequirement | None`
- `agents/nl_requirement.py` — new agent
- `skills/nl_requirement.md` — parsing instructions
- `cli/main.py` — add `--requirement TEXT` option to `analyze`
- `api.py` — add `requirement: str | None` param to `analyze()`

---

## 3. Coding Agent — Generate & Execute Split Scripts

### Purpose

Instead of (or in addition to) running splits inside the Python process, a
coding agent writes a self-contained Python script that the **user can inspect,
modify, and re-run locally** to reproduce the exact same splits.

### Output

```
output/
└── ptb-xl/
    ├── splits.json
    ├── split_metadata.json
    ├── analysis_report.md
    └── scripts/
        ├── generate_splits.py     ← runnable, no CardioMAS dependency
        ├── verify_splits.py       ← re-runs and compares to splits.json
        └── explore_dataset.py     ← EDA script with statistics + plots
```

### Script requirements

- **`generate_splits.py`** must be fully self-contained (only stdlib + pandas +
  numpy + scikit-learn). No `cardiomas` import. All parameters (seed, ratios,
  field names, filters) hardcoded as constants at the top of the file.
- The script prints a SHA-256 of its output to stdout so the user can verify
  against `split_metadata.json`.
- Header comment block documents: dataset name, CardioMAS version, date,
  `raw_requirement` if provided.

### Agent design

**New agent:** `agents/coder.py`
**New skill:** `skills/coder.md`

The coder agent receives:
- `DatasetInfo` (field names, id field, num records)
- `AnalysisReport` (CSV paths, column types, statistics)
- `ParsedRequirement` or `UserOptions` (ratios, stratify field, filters, seed)
- `SplitManifest` (the already-computed splits to use as ground truth)

It writes three scripts, executes `generate_splits.py` in a subprocess, and
verifies that the output SHA-256 matches `split_metadata.json`.

**Execution safety:**
- Scripts run in a restricted subprocess with a timeout (default 120 s)
- No network access in subprocess
- Only reads from `local_path`, writes only to `output/{dataset}/scripts/`

### New GraphState fields

```python
generated_scripts: dict[str, str] = {}    # {script_name: file_path}
script_execution_log: list[dict] = []     # [{script, stdout, stderr, exit_code, sha256}]
script_verified: bool = False
```

### Implementation

- `agents/coder.py`
- `skills/coder.md`
- `tools/code_tools.py` — `write_script_tool`, `execute_script_tool`, `verify_script_output_tool`
- `schemas/state.py` — add script fields
- `graph/workflow.py` — orchestrator dispatches coder after splitter

---

## 4. Per-Agent LLM Configuration

### Purpose

Allow different agents to use different LLMs. Example: use a fast local model
(Gemma 2B) for discovery and paper search, and a more capable model (llama3.1:8b
or cloud) for analysis and split strategy reasoning.

### Configuration

**`.env` / environment variables:**

```bash
# Default for all agents (fallback)
OLLAMA_MODEL=llama3.1:8b

# Per-agent overrides (optional)
AGENT_LLM_ORCHESTRATOR=llama3.1:8b
AGENT_LLM_NL_REQUIREMENT=gemma3:4b
AGENT_LLM_DISCOVERY=gemma3:4b
AGENT_LLM_PAPER=llama3.1:8b
AGENT_LLM_ANALYSIS=llama3.1:8b
AGENT_LLM_SPLITTER=gemma3:4b
AGENT_LLM_SECURITY=gemma3:4b
AGENT_LLM_CODER=deepseek-coder:6.7b
AGENT_LLM_PUBLISHER=gemma3:4b
```

**CLI:**

```bash
cardiomas analyze /data/ptb-xl/ \
  --llm-coder deepseek-coder:6.7b \
  --llm-analysis llama3.1:8b
```

**Python API:**

```python
mas = CardioMAS(
    agent_llms={
        "coder":    "deepseek-coder:6.7b",
        "analysis": "llama3.1:70b",
        "default":  "gemma3:4b",
    }
)
```

### Implementation

- `config.py` — add `AGENT_LLM_*` env var loading; expose `get_agent_llm(agent_name)` function
- `llm_factory.py` — `get_llm_for_agent(agent_name, prefer_cloud)` — reads per-agent config
- `agents/base.py` — `run_agent()` calls `get_llm_for_agent()` instead of `get_llm()`
- `schemas/state.py` — add `agent_llm_map: dict[str, str]` to `UserOptions`
- `cli/main.py` — add `--llm-<agent>` options

### Verbose LLM name display

Every `vprint_llm()` call in `verbose.py` must include the model name:

```
──────────── paper — LLM call [llama3.1:8b @ ollama] ────────────
```

- `verbose.py` — update `vprint_llm()` signature to accept `model_name: str`
- `agents/base.py` — pass `llm.model` (or equivalent) to `vprint_llm()`

---

## 5. Conversation & Reasoning Recorder

### Purpose

Full transparency: every LLM prompt, response, and agent reasoning step is
saved to disk alongside the splits. Users and reviewers can audit exactly what
the system decided and why.

### Output structure

```
output/
└── ptb-xl/
    ├── splits.json
    ├── split_metadata.json
    ├── analysis_report.md
    ├── scripts/
    │   └── ...
    └── session_log/
        ├── session.json           ← machine-readable full log
        ├── conversation.md        ← human-readable narrative
        └── reasoning_trace.md     ← orchestrator decisions + agent reasoning
```

### `session.json` schema

```python
class LLMCall(BaseModel):
    timestamp: datetime
    agent: str
    model: str
    system_prompt: str
    user_message: str
    response: str
    duration_ms: int
    token_counts: dict | None        # if available from API

class AgentStep(BaseModel):
    timestamp: datetime
    agent: str
    action: str
    inputs: dict                     # relevant state fields consumed
    outputs: dict                    # fields written to state
    llm_calls: list[LLMCall]
    reasoning: str                   # agent's stated reason for decisions
    skipped: bool = False
    skip_reason: str = ""

class SessionLog(BaseModel):
    session_id: str                  # UUID
    cardiomas_version: str
    started_at: datetime
    completed_at: datetime | None
    dataset_name: str
    raw_requirement: str | None
    user_options: dict
    agent_steps: list[AgentStep]
    orchestrator_reasoning: list[str]
    final_status: str
    errors: list[str]
```

### `conversation.md` format

Human-readable markdown showing each agent as a conversation turn:

```markdown
## Orchestrator → Discovery Agent

**Reason dispatched:** Source is a local path with no URL. Registry lookup
will be attempted before any network calls.

---

## Discovery Agent

**Input:** `/data/ptb-xl/`
**Registry hit:** Found `ptb-xl` in built-in registry.
**Output:** DatasetInfo populated (21,799 records, 12-lead, 500 Hz)

---

## Orchestrator → Paper Agent

**Reason dispatched:** Dataset has a known paper URL. Extracting split
methodology to check for official train/val/test definition.
...
```

### `reasoning_trace.md` format

Focused on the orchestrator's routing decisions:

```markdown
## Step 1: After Discovery
- State: dataset_info populated, local path present
- Decision: proceed to Paper agent
- Reason: Paper URL known from registry; checking for official splits
  before generating custom ones (reproducibility priority)

## Step 2: After Paper
- State: official splits found in paper (Section 2.3)
- Decision: proceed to Analysis, then Splitter with use_official=True
- Reason: Official splits take priority over generated splits per
  split_strategy.md rules
...
```

### Implementation

- `schemas/session.py` — `LLMCall`, `AgentStep`, `SessionLog` Pydantic models
- `recorder.py` — singleton `SessionRecorder` that agents write to
- `agents/base.py` — `run_agent()` writes every LLM call to recorder
- `agents/orchestrator.py` — writes routing reasoning to recorder after each decision
- `graph/workflow.py` — initialises recorder at pipeline start; saves log at end
- `schemas/state.py` — add `session_id: str` to `GraphState`

---

## 6. Context Compression for Long Sessions

### Purpose

When a CardioMAS session processes large datasets (many files, long papers, big
metadata CSVs), LLM context windows fill up. The system must handle this
gracefully without losing state or breaking reproducibility.

### Strategy

**At the agent level (`agents/base.py`):**

If a prompt + context exceeds a configurable token limit, a compression step
runs first:

1. Send the full context to a cheap/fast model with the instruction:
   *"Summarise the following for use by a [agent_name] agent. Preserve all
   field names, numbers, URLs, and section references exactly."*
2. Use the compressed summary as context instead of the raw text.
3. Log that compression occurred (with original length) in `session.json`.

```python
CONTEXT_COMPRESS_THRESHOLD = 6000   # chars (configurable via env)
CONTEXT_COMPRESS_MODEL = "gemma3:4b"  # fast cheap model for compression
```

**At the CLI level:**

`cardiomas analyze` saves a `session_checkpoint.json` after each agent
completes. If the process is interrupted or hits a limit, the user can resume:

```bash
cardiomas resume output/ptb-xl/session_checkpoint.json
```

The resume command loads the checkpoint into `GraphState` and re-enters the
pipeline at the last incomplete agent.

### Implementation

- `agents/base.py` — add `_compress_context()` helper
- `config.py` — `CONTEXT_COMPRESS_THRESHOLD`, `CONTEXT_COMPRESS_MODEL`
- `graph/workflow.py` — checkpoint save after each node
- `cli/main.py` — add `cardiomas resume CHECKPOINT_FILE` command
- `schemas/state.py` — add `checkpoint_path: str`, `last_completed_agent: str`

---

## 7. Implementation Order

### Phase 1 — Infrastructure (do first, everything depends on it)

1. Per-agent LLM config (`config.py`, `llm_factory.py`)
2. Verbose LLM name in `verbose.py` + `agents/base.py`
3. Session recorder (`schemas/session.py`, `recorder.py`, `agents/base.py`)
4. New `GraphState` fields for all V2 features

### Phase 2 — Natural Language Input

5. `ParsedRequirement` schema
6. `nl_requirement` agent + skill file
7. CLI `--requirement` option + API `requirement=` param

### Phase 3 — Dynamic Orchestrator

8. Rewrite `orchestrator_agent` as supervisor
9. Update `workflow.py` hub-and-spoke routing
10. `skills/orchestrator.md` decision tree

### Phase 4 — Coding Agent

11. `code_tools.py` (write, execute, verify)
12. `coder` agent + skill file
13. Script templates (generate_splits, verify_splits, explore_dataset)
14. Subprocess execution + SHA-256 verification

### Phase 5 — Context Compression & Resume

15. `_compress_context()` in `base.py`
16. Checkpoint save/load in `workflow.py`
17. `cardiomas resume` CLI command

### Phase 6 — Tests & Docs

18. Unit tests for each new agent (mock LLM)
19. Unit tests for session recorder
20. Unit tests for script generation + execution
21. Update `CLAUDE.md`
22. Update `README.md`
23. Tag `v0.5.0` and merge to `master`

---

## 8. New File Map

```
src/cardiomas/
├── agents/
│   ├── orchestrator.py       REWRITE — dynamic supervisor
│   ├── nl_requirement.py     NEW
│   ├── coder.py              NEW
│   └── base.py               MODIFY — per-agent LLM, recorder, compression
├── schemas/
│   ├── state.py              MODIFY — new fields
│   ├── session.py            NEW — LLMCall, AgentStep, SessionLog
│   └── requirement.py        NEW — ParsedRequirement
├── tools/
│   └── code_tools.py         NEW — write_script, execute_script, verify_script
├── skills/
│   ├── orchestrator.md       REWRITE — full decision tree
│   ├── nl_requirement.md     NEW
│   └── coder.md              NEW
├── graph/
│   └── workflow.py           MODIFY — hub-and-spoke, checkpointing
├── config.py                 MODIFY — per-agent LLM env vars
├── llm_factory.py            MODIFY — get_llm_for_agent()
├── recorder.py               NEW — SessionRecorder singleton
├── verbose.py                MODIFY — show model name
└── cli/
    └── main.py               MODIFY — --requirement, --llm-*, resume command
```

---

## 9. Constraints

- **Reproducibility** — `ParsedRequirement.raw_input`, seed, strategy, and
  all filter parameters must be recorded in `split_metadata.json`. Scripts
  generated by the coder agent must produce byte-identical output to
  `splits.json` when run independently.
- **No raw data ever saved** — coder agent scripts must never write signal
  arrays, only record IDs.
- **Security gate unchanged** — publisher still blocked if security audit fails.
- **Backward compatibility** — all V1 CLI flags and Python API signatures
  continue to work unchanged. V2 adds, never removes.
