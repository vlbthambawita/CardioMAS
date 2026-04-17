# Script-First Dataset Query Plan

**Goal:** For any query that involves a configured dataset, the system must generate one or more
standalone Python scripts as the primary way to answer the query. The scripts are saved to a clean
output location. The agent does not attempt to answer from corpus retrieval, inspection fallbacks,
or any other mechanism. Optionally (Phase 2), the agent can execute those scripts and feed their
output back into the corpus as evidence, allowing the responder to synthesize a direct answer from
real computed results.

---

## Scope of Changes

| File | Change type |
|---|---|
| `schemas/config.py` | New config fields: `dataset_mode`, `scripts_dir` |
| `agentic/planner.py` | Strict script-only routing when dataset present |
| `coding/tool_builder.py` | Standalone script format (no `run(payload)` contract) |
| `autonomy/workspace.py` | New `scripts_output_dir()` + `write_standalone_script()` |
| `autonomy/recovery.py` | Adapt `generate_python_artifact` for standalone scripts; remove retry-on-empty-output logic (scripts are always "ok" when written, not when executed) |
| `agentic/responder.py` | New `_compose_script_report()` path: list scripts, don't synthesize answer |
| `inference/prompts.py` | New planner system prompt rule: dataset queries → scripts only |
| `agentic/aggregator.py` | Pass script file paths cleanly through to responder |

Non-dataset queries (web fetch, arithmetic, corpus Q&A) are **not affected**. This mode activates
only when (a) `dataset_mode = "script_only"` and (b) at least one `dataset_dir` or `local_dir`
source is configured.

---

## 1 — Config Changes (`schemas/config.py`)

### 1.1 New field in `AutonomyConfig`

```python
dataset_mode: Literal["script_only", "agentic"] = "agentic"
```

- `"agentic"` — current behavior (unchanged).
- `"script_only"` — any query with a configured dataset routes exclusively to script generation.

### 1.2 New field in `RuntimeConfig`

```python
scripts_dir: str = ""
```

Resolved path (property):

```python
@property
def resolved_scripts_dir(self) -> Path:
    if self.scripts_dir:
        return Path(self.scripts_dir)
    return Path(self.output_dir) / "scripts"
```

### 1.3 New fields in `AutonomyConfig` for Phase 2

```python
execute_for_answer: bool = False          # run script and feed output to responder
execution_timeout_seconds: int = 60       # subprocess timeout
```

`execute_for_answer` is only honoured when `dataset_mode = "script_only"`. When True, the agent
runs the script in a subprocess after writing it, captures stdout, and injects the output as a
high-priority `EvidenceChunk` so the responder can give a direct answer. The script file is still
written to disk regardless — the user can always re-run it.

`execute_for_answer` is gated by the existing `require_approval_for_shell_execution` policy flag.
If that flag is True and no approval has been granted in this session, execution is skipped and
the system falls back to the Phase 1 script-report output.

### 1.4 Example YAML (`examples/ollama/runtime_ptbxl.yaml` addition)

```yaml
autonomy:
  enable_code_agents: true
  allow_tool_codegen: true
  dataset_mode: script_only          # NEW
  execute_for_answer: true           # NEW Phase 2 — run script, feed output to responder
  execution_timeout_seconds: 60      # NEW Phase 2
  require_approval_for_shell_execution: false   # set false to allow auto-execution
  max_repair_attempts: 2
scripts_dir: output/scripts          # NEW — flat, user-accessible location
```

---

## 2 — Planner Changes (`agentic/planner.py`)

### 2.1 New helper: `_is_script_only_mode(config)`

```python
def _is_script_only_mode(config: RuntimeConfig) -> bool:
    return (
        config.autonomy.dataset_mode == "script_only"
        and bool(_first_dataset_path(config))
        and tool_codegen_allowed(config)
    )
```

### 2.2 Changes to `_heuristic_plan()`

When `_is_script_only_mode(config)` is True:

1. **Skip** `retrieve_corpus`, `inspect_dataset`, `calculate`, `fetch_webpage` for dataset-related
   queries (they are not added to `steps`).
2. **Always** add one `generate_python_artifact` step with
   `args={"task": query, "dataset_path": ..., "target_path": "", "artifact_name": ""}`.
3. Return immediately — do not proceed to the general-purpose `retrieve_corpus` append at line 172.

Non-dataset queries (URLs, explicit arithmetic) still follow existing logic even in script_only mode.

**Pseudocode:**

```python
if _is_script_only_mode(config):
    dataset_path = _first_dataset_path(config)
    target_path = _extract_local_path(query)
    steps = [PlanStep(
        tool_name="generate_python_artifact",
        reason="Dataset query in script_only mode — generating standalone analysis script.",
        args={"task": query, "dataset_path": dataset_path,
              "target_path": target_path, "artifact_name": ""},
    )]
    return AgentDecision(strategy="single_tool", steps=steps, notes=[])
```

### 2.3 Changes to `_sanitize_decision()`

Add the same guard at the top of the `generate_python_artifact` branch: if `dataset_mode ==
"script_only"`, drop all other steps from the Ollama-produced plan and keep only the
`generate_python_artifact` step (re-using existing sanitization for its args).

---

## 3 — Script Format Change (`coding/tool_builder.py`)

### 3.1 Current format (to be replaced in script_only mode)

Current: a Python module with a `run(payload: dict) -> dict` function that is dynamically
imported and executed by the agent.

### 3.2 New standalone format

In `script_only` mode, generate a **standalone script** that:

- Has `if __name__ == "__main__":` at entry.
- Hardcodes `DATASET_PATH` from the config (no injection at runtime).
- Hardcodes `OUTPUT_DIR` to `{resolved_scripts_dir}/outputs/`.
- Writes a `results.txt` (and optionally `results.json`) to `OUTPUT_DIR`.
- Prints results to stdout so `python script.py` works as-is.
- Contains a docstring explaining what the script does.
- Has a `# HOW TO RUN:` comment block at the top.

**Template structure:**

```python
#!/usr/bin/env python3
"""
Task:    {task}
Dataset: {dataset_path}
Generated by CardioMAS on {timestamp}

HOW TO RUN:
    python {script_filename}

OUTPUT:
    Results printed to stdout and written to {output_dir}/results.json
"""

import json
import os
from pathlib import Path

DATASET_PATH = "{dataset_path}"
OUTPUT_DIR   = Path("{output_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # --- generated analysis code here ---
    ...
    result = {"answer": ..., "details": ...}
    (OUTPUT_DIR / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

### 3.3 New function `build_standalone_script(task, dataset_path, config, last_error="")`

Add alongside existing `build_generated_tool_package()`. Returns a `StandaloneScript` dataclass:

```python
@dataclass
class StandaloneScript:
    script_name: str          # e.g. "count_unique_patients_a3f2c1.py"
    code: str                 # full Python source
    output_dir: Path          # where results will be written
    description: str          # one-line summary of what the script does
```

Script naming: `{slug}_{hash8}.py` — `slug` is derived from the task (snake_case, ≤40 chars),
`hash8` is SHA-1 of `task + dataset_path` truncated to 8 chars. This is deterministic so the same
query always produces the same filename.

### 3.4 Multi-script support (optional, phase 2)

For complex queries the builder may return a list of `StandaloneScript` objects:

- `01_inspect_structure_{hash}.py` — discovers column names, file layout.
- `02_compute_{slug}_{hash}.py` — performs the actual analysis.

Phase 1 produces a single comprehensive script. Multi-script support is Phase 2.

---

## 4 — Workspace Changes (`autonomy/workspace.py`)

### 4.1 New method: `scripts_output_dir(config)`

```python
def scripts_output_dir(self) -> Path:
    d = self.config.resolved_scripts_dir
    d.mkdir(parents=True, exist_ok=True)
    return d
```

### 4.2 New method: `write_standalone_script(script)`

```python
def write_standalone_script(self, script: StandaloneScript) -> Path:
    dest = self.scripts_output_dir() / script.script_name
    dest.write_text(script.code, encoding="utf-8")
    return dest
```

This is the *only* path used in `script_only` mode. The deep `sessions/{id}/{slug}/` workspace
path is **not** used for the primary script (it may still be used for execution records if needed).

### 4.3 Clean output directory layout

```
{output_dir}/
  scripts/
    count_unique_patients_a3f2c1.py    ← user-runnable script
    inspect_columns_b7d9e2.py
    outputs/
      results.json                     ← written by script (user-run or agent-run)
      stdout_a3f2c1.txt                ← Phase 2: captured stdout saved alongside results
  corpus/                              ← existing, unchanged
  manifest.json                        ← existing, unchanged
```

Phase 2 also writes a `stdout_{hash8}.txt` file next to `results.json` so the captured output is
preserved for inspection even after the session ends.

---

## 5 — Recovery / Execution Changes (`autonomy/recovery.py`)

### 5.1 `generate_python_artifact` in script_only mode

In `script_only` mode the agent **does not execute** the script. The method:

1. Calls `build_standalone_script(task, dataset_path, config, last_error)`.
2. Writes the script via `workspace.write_standalone_script(script)`.
3. Returns `ToolResult(ok=True, summary=f"Script written to {path}", data={"script_path": str(path), ...})`.

No dynamic import, no `module.run()`, no stdout capture, no retry-on-empty-output (the empty-output
retry added in the previous session is **removed** — it only makes sense in agentic mode where the
agent executes the script).

### 5.2 Repair loop in script_only mode

Retry (up to `max_repair_attempts`) still applies but only for **generation errors** (LLM returns
unparseable code, syntax error detected by `ast.parse()`). Each retry passes `last_error` to
`build_standalone_script` so the LLM can correct the code.

The retry does **not** run the script to check output.

### 5.3 Static verification

`verify_generated_tool()` is replaced for standalone scripts by `ast.parse()` + a check that the
`if __name__ == "__main__":` block and `main()` function are present.

### 5.4 Phase 2 — Script execution and output capture

When `config.autonomy.execute_for_answer` is True and `require_approval_for_shell_execution` is
False (or approval has been granted), after writing the script the agent runs it:

```python
completed = subprocess.run(
    ["python", str(script_path)],
    capture_output=True,
    text=True,
    timeout=config.autonomy.execution_timeout_seconds,
    cwd=scripts_dir,
    check=False,
)
```

**On success** (`returncode == 0`):
- `stdout` is captured.
- An `EvidenceChunk` is created with:
  - `source_type = "script_output"`
  - `content = stdout` (truncated to 4000 chars if needed)
  - `score = 2.0` — higher than any corpus chunk, so it ranks first in the responder
  - `title = f"Script output: {script_name}"`
  - `uri = str(script_path)`
- This chunk is appended to the existing evidence list alongside the script-path chunk.
- `ToolResult.data["execution_stdout"]` is set to `stdout`.
- `ToolResult.data["executed"] = True`.

**On failure** (`returncode != 0` or timeout):
- Execution error/stderr is recorded in `ToolResult.data["execution_error"]`.
- `ToolResult.data["executed"] = False`.
- The system falls back to Phase 1 behavior (script-report, no direct answer).
- If `max_repair_attempts > 0`, the error is passed as `last_error` to
  `build_standalone_script()` and a corrected script is generated and executed again.
  This is the **execute-repair loop**: generate → execute → on failure → regenerate with error
  context → execute again, up to `max_repair_attempts` times.

**Execution isolation:**
- The subprocess inherits no extra environment beyond `PYTHONPATH` pointing to the dataset path.
- `allowed_python_modules` from `AutonomyConfig` is enforced by a pre-execution `ast` import
  scan (same as static verification).

---

## 6 — Aggregator Changes (`agentic/aggregator.py`)

### 6.1 New key in aggregate dict: `"standalone_scripts"`

```python
"standalone_scripts": [
    {
        "script_path": "/absolute/path/to/script.py",
        "script_name": "count_unique_patients_a3f2c1.py",
        "description": "Counts unique patients in the PTB-XL dataset",
        "dataset_path": "/work/vajira/DATA/...",
        "output_dir": "/work/vajira/DL2026/CardioMAS/output/scripts/outputs",
        "executed": False,               # True only in Phase 2 when execute_for_answer=True
        "execution_stdout": "",          # Phase 2: captured stdout (empty if not executed)
        "execution_error": "",           # Phase 2: stderr/timeout message if failed
    },
    ...
]
```

This dict is populated from `ToolResult.data` for every `generate_python_artifact` result in
`script_only` mode. The aggregator also promotes any `script_output` evidence chunks to the front
of the evidence list (they have `score=2.0` so standard score-sorting handles this automatically).

---

## 7 — Responder Changes (`agentic/responder.py` + `inference/prompts.py`)

### 7.1 Responder dispatch logic

The responder checks conditions in this order when `standalone_scripts` is non-empty:

```
execute_for_answer=True AND any script executed successfully AND execution_stdout non-empty?
    → Phase 2 path: synthesize direct answer from script output
    → (uses Ollama responder or deterministic summary of stdout)
else
    → Phase 1 path: _compose_script_report() — list scripts, tell user how to run them
```

### 7.2 Phase 1 path: `_compose_script_report()`

When scripts were generated but not executed (or execution failed), returns:

```
Scripts have been generated to answer your query.

Query: {query}

Generated script(s):
  1. {script_name}
     Path:    {script_path}
     Purpose: {description}

How to run:
  python {script_path}

Output will be written to:
  {output_dir}/results.json

No answer is inferred by the agent. Run the script(s) to get the exact result.
```

No LLM call. This is a deterministic formatted string.

### 7.3 Phase 2 path: answer from script output

When `execution_stdout` is non-empty, inject it into the evidence and call the responder normally.
The `script_output` EvidenceChunk (score=2.0) ranks above all corpus chunks, so the Ollama
responder answers from the actual computed result rather than guessing.

If `responder_uses_ollama` is False, the deterministic responder emits the stdout verbatim as the
answer with a citation pointing to the script file.

The script-path report is still included as a secondary block below the answer so the user knows
where the script lives.

### 7.4 Ollama responder in script_only + Phase 1

If `execute_for_answer` is False (or execution failed) and `responder_uses_ollama` is True, bypass
the LLM entirely — use `_compose_script_report()`. The responder LLM is not called for Phase 1.

### 7.3 Planner system prompt update (`inference/prompts.py`)

Add to the planner system prompt:

```
STRICT RULE: If dataset_mode is "script_only" and a dataset path is configured, you MUST
select generate_python_artifact as the sole tool. Do not select retrieve_corpus,
inspect_dataset, fetch_webpage, or calculate for dataset queries. The agent does not
execute the script — it is saved for the user to run.
```

---

## 8 — CLI Output Changes (`cli/main.py`)

In script_only mode, the `query` command output should clearly show script paths.

Current final answer block:

```
Answer
The evidence provided is insufficient...
```

New block:

```
Scripts Generated
─────────────────
  count_unique_patients_a3f2c1.py
  Path: /work/vajira/.../output/scripts/count_unique_patients_a3f2c1.py

Run with:  python /work/vajira/.../output/scripts/count_unique_patients_a3f2c1.py
Output to: /work/vajira/.../output/scripts/outputs/results.json
```

The `--json` flag outputs the same data as structured JSON (unchanged).

---

## 9 — Implementation Order

### Phase 1 (write scripts, user runs them)

1. **Config** — add `dataset_mode`, `scripts_dir` fields (no behavior change yet).
2. **Script format** — add `build_standalone_script()` + `StandaloneScript` dataclass alongside
   existing `build_generated_tool_package()`. Existing agentic mode is untouched.
3. **Workspace** — add `scripts_output_dir()` and `write_standalone_script()`.
4. **Recovery** — add the `script_only` write-only branch inside `generate_python_artifact()`;
   keep existing agentic branch behind an `else`.
5. **Aggregator** — populate `standalone_scripts` key (with `executed=False`).
6. **Responder** — add `_compose_script_report()` and the Phase 1 dispatch guard.
7. **Planner** — add `_is_script_only_mode()` and update `_heuristic_plan()` +
   `_sanitize_decision()`.
8. **Prompts** — update planner system prompt rule.
9. **CLI** — update `_render_query_result()` to display script paths.

### Phase 2 (execute scripts, feed output to responder)

10. **Config** — add `execute_for_answer` and `execution_timeout_seconds` fields.
11. **Recovery** — add subprocess execution block after `write_standalone_script()`; implement
    execute-repair loop; save `stdout_{hash8}.txt` to `scripts/outputs/`.
12. **Aggregator** — pass `executed`, `execution_stdout`, `execution_error` through to aggregate
    dict; promote `script_output` EvidenceChunk (score=2.0).
13. **Responder** — add Phase 2 dispatch (stdout non-empty → synthesize answer from evidence).
14. **CLI** — when answer came from script output, display it under "Answer" with a note showing
    the script path.

---

## 10 — Files NOT Changed

- `knowledge/corpus.py`, `knowledge/loaders.py` — corpus building unchanged.
- `retrieval/` — all retrieval modes unchanged (used by non-dataset queries).
- `tools/dataset_tools.py`, `tools/retrieval_tools.py` — unchanged (available in agentic mode).
- `schemas/runtime.py`, `schemas/evidence.py` — unchanged.
- `memory/`, `safety/` — unchanged.
- `mappers/` — unchanged.
- `evaluation/` — unchanged.
- Any existing tests — they run against agentic mode (default); script_only mode needs new tests.

---

## 11 — New Tests Required

### Phase 1 tests

| Test file | What it covers |
|---|---|
| `tests/test_script_builder.py` | `build_standalone_script()` produces `ast.parse()`-valid code; script name is deterministic; `if __name__` and `main()` present; `DATASET_PATH` hardcoded correctly |
| `tests/test_script_only_planner.py` | Heuristic planner returns exactly one `generate_python_artifact` step in script_only mode; `retrieve_corpus`/`inspect_dataset` not added |
| `tests/test_script_only_responder.py` | Responder returns `_compose_script_report()` output (not an LLM answer) when `standalone_scripts` present and `executed=False` |
| `tests/test_workspace_scripts.py` | `write_standalone_script()` writes to `resolved_scripts_dir`; file is readable; path matches config |

### Phase 2 tests

| Test file | What it covers |
|---|---|
| `tests/test_script_execution.py` | Subprocess execution captures stdout; `execution_stdout` in ToolResult; `stdout_{hash}.txt` written to outputs dir; timeout respected |
| `tests/test_execute_repair_loop.py` | On non-zero returncode, `last_error` is set and a new script is generated; loop stops after `max_repair_attempts` |
| `tests/test_script_output_evidence.py` | `script_output` EvidenceChunk has score=2.0; aggregator places it first; responder synthesizes answer from it |
| `tests/test_execute_for_answer_gating.py` | When `require_approval_for_shell_execution=True`, execution is skipped and system falls back to Phase 1 report |
