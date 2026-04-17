# LLM-Driven Code Generation Plan

## Problem

`build_standalone_script()` in `coding/tool_builder.py` generates identical code for every query.
`_render_standalone_script_source()` is a pure Python f-string template — no LLM is ever called.
The "repair loop" in `recovery.py` calls `build_standalone_script(last_error=last_error)`, but since
there is no LLM, the error is just embedded as a Python comment. No actual reasoning or fixing happens.

**Result:** every query against PTB-XL (or any dataset) produces the same generic file-scanner script
regardless of what the user actually asked.

---

## Goal

Replace the static template with a **multi-step LLM-driven pipeline** that:

1. Discovers actual dataset structure (no LLM — fast filesystem probe)
2. Reasons about what the query requires (LLM planning call)
3. Synthesizes task-specific Python code (LLM code generation call)
4. Executes the script (existing Phase 2 subprocess)
5. On failure, uses LLM to repair the code with full error context (LLM repair call)

The result should be a **different, purpose-built script for every distinct query** — like a code agent
specialised for medical dataset analysis.

---

## New Config Fields

Add to `LLMConfig` in `src/cardiomas/schemas/config.py`:

```python
code_model: str = ""               # defaults to model if empty
code_max_tokens: int = 4000        # scripts are long; 800 is not enough
code_temperature: float = 0.2      # low but not zero: allow syntactic flexibility

@property
def resolved_code_model(self) -> str:
    return self.code_model or self.model
```

Update `ptbxl_code_only.yaml` with a minimal `llm:` block:

```yaml
llm:
  provider: ollama
  base_url: http://localhost:11434
  planner_mode: heuristic       # planner stays heuristic; code uses LLM
  model: gemma4:e2b
  code_max_tokens: 4000
  code_temperature: 0.2
```

> **Note:** `planner_mode: heuristic` means the planner does not need Ollama at planning time.
> Ollama is only called during code synthesis and repair.

---

## New Module: `src/cardiomas/coding/llm_coder.py`

This is the core of the change. Public API:

```python
def synthesize_script(
    task: str,
    dataset_path: str,
    config: RuntimeConfig,
    chat_client: ChatClient,
    last_error: str = "",
    previous_code: str = "",
) -> str:
    """Return a complete, standalone Python script as a string."""
```

### Internal pipeline

#### Step 1 — Dataset discovery (no LLM)

```python
def discover_dataset_structure(dataset_path: str) -> dict:
    """
    Fast filesystem probe. Returns:
    {
        "total_files": int,
        "extension_counts": {"csv": 3, ".tsv": 1, ...},
        "tabular_files": [{"path": "...", "columns": [...], "row_count": int}, ...],   # up to 5 files
        "sample_files": ["..."],   # up to 30 paths relative to dataset root
        "dataset_root": "...",
    }
    """
```

For each CSV/TSV found (up to 5), open it and read the header + count rows. This gives the LLM
**actual column names** from the real dataset — critical for writing correct code.

#### Step 2 — Computation planning (LLM call, free text)

```python
def _plan_computation(
    task: str,
    dataset_context: dict,
    chat_client: ChatClient,
    config: RuntimeConfig,
) -> str:
    """Ask LLM: what should the script compute and how? Returns natural language plan."""
```

System prompt:
```
You are a medical dataset analysis expert and Python programmer.
Your job is to plan what a standalone Python script must compute to answer a user's query.
Be specific: name the exact files to read, columns to use, and operations to perform.
Output a numbered list of steps. Do not write code yet.
```

User prompt:
```
Dataset root: {dataset_root}
Total files: {total_files} ({extension_counts})

Tabular files found:
  {for each tabular_file: path, columns list, row count}

Sample file paths:
  {sample_files}

User query: "{task}"

List the exact steps the script must take to answer this query. Be specific about:
- Which file(s) to read
- Which columns to use
- What to compute (counts, distributions, filters, etc.)
- What to print as output
```

> This call uses **free-text output** (no `json_mode`). The plan is used only as internal context
> for the synthesis step.

#### Step 3 — Code synthesis (LLM call, raw Python output)

```python
def _generate_code(
    task: str,
    dataset_path: str,
    dataset_context: dict,
    plan: str,
    chat_client: ChatClient,
    config: RuntimeConfig,
    last_error: str = "",
    previous_code: str = "",
) -> str:
    """Ask LLM to write the Python script. Returns raw Python source."""
```

System prompt:
```
You are an expert Python programmer generating standalone analysis scripts for medical ECG datasets.

Rules:
- Output ONLY valid Python code. No markdown. No explanations. No triple-backtick fences.
- The script must be runnable with: python script.py (no arguments)
- Hardcode DATASET_PATH = "{dataset_path}" and OUTPUT_DIR = Path("{output_dir}")
- Print results to stdout as JSON (import json; print(json.dumps(result, indent=2, default=str)))
- Write results to OUTPUT_DIR / "results.json" as well
- Include: def main(): ... and: if __name__ == "__main__": main()
- Import only: csv, json, math, statistics, pathlib, collections, datetime, numpy, pandas (if available)
- Handle errors with try/except; print {"ok": false, "error": "..."} on failure
- Read real data files — do not produce placeholder output
```

User prompt (when `last_error` is empty — first attempt):
```
Task: "{task}"

Dataset structure:
{dataset_context as formatted text}

Computation plan:
{plan}

Write a complete Python script that executes this plan and answers the task.
Output Python code only.
```

User prompt (when `last_error` is set — repair attempt):
```
Task: "{task}"

The previous script failed. Error:
{last_error}

Previous script:
{previous_code}

Dataset structure:
{dataset_context as formatted text}

Fix the script so it runs without errors and produces the correct output.
Output the complete corrected Python script only.
```

> **Key insight for repair:** the LLM sees the actual broken code + the real error message,
> not just a comment. It can reason about what went wrong and fix it specifically.

#### Step 4 — Code extraction

The LLM might wrap output in markdown fences despite instructions.
Add `_extract_python_code(raw: str) -> str` that strips triple-backtick blocks if present.

#### Fallback

If `chat_client` is None (no `llm:` in config), fall back to the existing
`_render_standalone_script_source()` template with a warning logged to stderr.
This preserves backward compatibility for configs without Ollama.

---

## Modified Files

### `src/cardiomas/coding/tool_builder.py`

Change `build_standalone_script()` signature:

```python
def build_standalone_script(
    task: str,
    dataset_path: str,
    config: RuntimeConfig,
    last_error: str = "",
    previous_code: str = "",
    chat_client: ChatClient | None = None,    # NEW
) -> StandaloneScript:
```

Inside the function, after computing `mode` and `script_name`:

```python
if chat_client is not None:
    code = synthesize_script(task, dataset_path, config, chat_client,
                             last_error=last_error, previous_code=previous_code)
else:
    code = _render_standalone_script_source(...)   # existing template fallback
```

### `src/cardiomas/autonomy/recovery.py`

1. `AutonomousToolManager.__init__` receives `chat_client: ChatClient | None = None`:
   ```python
   def __init__(self, config: RuntimeConfig, chat_client: ChatClient | None = None) -> None:
       self.config = config
       self.chat_client = chat_client
       ...
   ```

2. `_write_standalone_script` passes `chat_client` and `previous_code` to `build_standalone_script`:
   ```python
   script = build_standalone_script(
       task, dataset_path, self.config,
       last_error=last_error,
       previous_code=last_code,
       chat_client=self.chat_client,
   )
   last_code = script.code   # track for repair
   ```
   Add `last_code: str = ""` variable in the attempt loop.

### `src/cardiomas/agentic/runtime.py`

Pass `chat_client` when constructing `AutonomousToolManager`:

```python
self._autonomy_manager = AutonomousToolManager(config, chat_client=self._chat_client)
```

This is the only change in `runtime.py`.

### `examples/ollama/ptbxl_code_only.yaml`

Add `llm:` block as shown in the Config section above.

### `README.md`

Add a note under `dataset_mode: script_only` explaining:
- When `llm:` is configured, scripts are LLM-generated and query-specific
- When `llm:` is absent, a generic template is used as fallback
- `code_max_tokens` should be ≥ 4000 for full scripts (default 800 is for JSON planner/responder)

---

## Implementation Order

1. **`schemas/config.py`** — add `code_model`, `code_max_tokens`, `code_temperature` to `LLMConfig`
2. **`coding/llm_coder.py`** — create from scratch with the full pipeline above
3. **`coding/tool_builder.py`** — add `chat_client` param, dispatch to `llm_coder.synthesize_script`
4. **`autonomy/recovery.py`** — add `chat_client` to `__init__`, thread it into `_write_standalone_script`, track `last_code` for repair
5. **`agentic/runtime.py`** — pass `chat_client` to `AutonomousToolManager`
6. **`examples/ollama/ptbxl_code_only.yaml`** — add `llm:` section
7. **`README.md`** — update `dataset_mode: script_only` section

---

## Tests to Add / Update

**`tests/test_llm_coder.py`** (new):
- `test_discover_dataset_structure_csv` — scans a temp dir with a CSV, verifies column names returned
- `test_discover_dataset_structure_empty` — empty dir handled gracefully
- `test_synthesize_script_calls_llm` — mock `chat_client.chat` returning stub Python code; verify the output is extracted and returned
- `test_synthesize_script_fallback_no_client` — `chat_client=None` → falls back to template without error
- `test_synthesize_script_repair_path` — `last_error` set → LLM receives previous code + error in prompt; mock verifies messages contain both
- `test_extract_python_code_strips_fences` — `_extract_python_code("```python\nprint(1)\n```")` returns `"print(1)\n"`

**`tests/test_recovery.py`** (update if exists, else create):
- `test_write_standalone_script_uses_llm_when_client_provided` — mock `chat_client`, verify `synthesize_script` is called
- `test_write_standalone_script_template_fallback` — `chat_client=None` in manager → template code returned

**Existing tests:** all 26 tests should still pass with no changes (backward compatible via `chat_client=None` default).

---

## Expected Behavior After Implementation

| Query | Before | After |
|---|---|---|
| "How many unique patients?" | Generic file scanner | Script that reads `ptbxl_database.csv`, counts unique `patient_id` values, prints result |
| "What is the age distribution?" | Same generic file scanner | Script that reads the metadata CSV, accesses the `age` column, prints histogram/stats |
| "List all diagnostic labels" | Same generic file scanner | Script that reads the `scp_codes` column or label CSV and prints unique values |
| "How many ECGs per patient on average?" | Same generic file scanner | Script that groups by `patient_id`, counts per patient, prints mean |

The LLM sees **actual column names** from the dataset discovery step, so it can reference real
columns like `patient_id`, `scp_codes`, `age`, `sex`, etc. in the generated code.

---

## Risk and Mitigations

| Risk | Mitigation |
|---|---|
| LLM generates code with markdown fences | `_extract_python_code()` strips them |
| LLM uses unavailable imports | System prompt restricts to safe import set; `verify_python_ast` catches violations |
| LLM halluccinates column names | Discovery step reads actual headers and passes them to LLM |
| LLM exceeds context for large datasets | Discovery caps at 5 tabular files × first 30 columns; sample_files caps at 30 |
| No `llm:` in config (e.g. `ptbxl_code_only.yaml` before update) | Fallback to template with warning |
| Code synthesis too slow | `code_max_tokens` is configurable; code model can differ from chat model |
| Repair loop produces same broken code | LLM sees actual error + previous code; each repair is independent reasoning |
