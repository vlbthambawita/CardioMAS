# CardioMAS Dynamic Dataset Codegen Plan

Date: 2026-04-17  
Scope: shift the system from built-in dataset analysis helpers to query-driven dynamic code generation

## 1. Decision

CardioMAS should move toward a meta-tool architecture.

The main built-in tools should help the agent:

- generate new code,
- save it in a safe and reproducible workspace,
- verify it,
- execute it,
- capture outputs,
- and retry when generation or execution fails.

Dataset handling itself should not be implemented as a growing library of ready-made domain tools.

## 2. Hard Requirement

Do not add ready-made ECG dataset analysis tools or ECG-specific action tools.

ECG datasets must be handled through code generated on the fly from the user query and dataset context. This includes:

- reading ECG dataset files,
- parsing ECG dataset layouts,
- extracting ECG metadata,
- computing ECG statistics,
- and writing helper scripts for ECG dataset workflows.

The runtime may keep generic safety, workspace, verification, and execution utilities, but ECG logic must not be hardcoded as first-class built-in tools.

## 3. Current Mismatch

The current implementation still has several patterns that conflict with this target:

- `read_dataset_file` and `dataset_statistics` are exposed as pre-baked runtime tools
- `coding/tool_builder.py` contains fixed templates for known dataset operations
- `mappers/format_readers/` contains format-specific readers that encourage built-in handling paths
- planner heuristics directly choose these prebuilt helper tools

This plan changes that direction.

## 4. Target Architecture

The runtime should expose generic support tools, not domain tools.

```text
User query
  ->
Planner
  ->
Codegen support tools
  ->
Generated code artifact
  ->
Verification
  ->
Execution
  ->
Captured output artifacts
  ->
Grounded response
```

## 5. Built-in Tool Philosophy

### Keep as built-in

- workspace manager
- artifact registry
- code generator interface
- verifier
- executor
- retry controller
- corpus retrieval
- safety and approval tools

### Do not keep as built-in

- ECG analysis tools
- ECG reader tools
- ECG file parsers
- ECG-specific statistics tools
- ECG-specific preprocessing tools

Those should be generated from prompts and local context for each task.

## 6. Safe and Reproducible Artifact Model

Generated code should be written under a deterministic workspace such as:

```text
runtime_output/
  autonomy_workspace/
    sessions/
      <session_id>/
        <artifact_slug>/
          prompt.json
          context.json
          tool_spec.json
          tool.py
          run.json
          stdout.txt
          stderr.txt
          outputs/
```

Requirements:

- every generated artifact gets its own directory
- prompt and context must be stored
- execution inputs and outputs must be stored
- code must be reloadable later
- rerunning the same artifact should be possible from stored metadata

## 7. Proposed Repository Direction

Add or extend these modules:

```text
src/cardiomas/
  autonomy/
    workspace.py
    artifacts.py
    verifier.py
    recovery.py
    execution_policy.py
  codegen/
    prompts.py
    planner.py
    synthesizer.py
    patcher.py
    templates.py
  execution/
    runner.py
    result_capture.py
    import_guard.py
```

Reduce the role of:

- `mappers/format_readers/` as runtime-first tools
- hardcoded generated-tool names such as `dataset_statistics`
- planner rules that map directly to fixed dataset helper tools

## 8. Main Runtime Flow

### Step 1: plan the work

The planner decides whether the query needs:

- direct retrieval,
- a generated code artifact,
- a generated shell script,
- or both.

### Step 2: synthesize code

The codegen layer writes a new artifact targeted to the exact request, for example:

- read one dataset file
- inspect a folder layout
- compute class counts from a CSV
- summarize waveform header metadata
- write a script for batch scanning

### Step 3: verify

Before execution:

- check imports
- check banned calls
- run syntax validation
- run a tiny smoke test with the provided path

### Step 4: execute

Run the generated artifact in the bounded workspace and capture:

- stdout
- stderr
- exit code
- structured JSON outputs
- generated files

### Step 5: recover if needed

If the generated code fails:

- classify the failure
- patch the code
- re-verify
- retry within a bounded limit

## 9. Query-Driven Codegen Targets

The system should generate code based on the exact query, not by selecting from a predefined ECG toolbox.

Examples of allowed dynamic targets:

- “Read `/path/to/file.hea` and explain its fields.”
- “Inspect this ECG dataset directory and summarize its file structure.”
- “Compute class counts from the metadata file in this dataset.”
- “Write a shell script that scans this dataset and lists candidate record files.”
- “Analyze missing values in the tabular metadata for this dataset.”

These tasks should result in newly written code artifacts, not direct dispatch to ECG-specific built-ins.

## 10. Implementation Phases

## Phase 1: Replace Hardcoded Dataset Helper Direction

Goal:
Stop treating current generated helper names as the long-term tool model.

Tasks:

- mark `read_dataset_file` and `dataset_statistics` as transitional
- add a new generic code-artifact execution path
- shift planner language from “call helper tool” to “generate analysis artifact”

Acceptance criteria:

- the architecture no longer depends on growing fixed dataset-analysis helpers.

## Phase 2: Artifact Workspace and Provenance

Goal:
Make generated code reproducible and inspectable.

Tasks:

- create per-session artifact directories
- persist prompt, context, code, verification logs, and execution logs
- add stable artifact ids and query-derived slugs

Acceptance criteria:

- every generated code run can be inspected and replayed from disk.

## Phase 3: Generic Codegen Engine

Goal:
Build one generator that creates targeted Python tools from the query.

Tasks:

- create prompts for:
  - file reader synthesis
  - dataset inspection synthesis
  - statistics synthesis
  - shell script synthesis
- provide only generic local context:
  - dataset path
  - file list
  - sample headers
  - repo execution contract
- remove ECG-specific assumptions from prompts

Acceptance criteria:

- ECG tasks are solved by generated code, not by ECG-specific built-in logic.

## Phase 4: Verification and Execution

Goal:
Run generated code safely.

Tasks:

- add syntax checks
- add import guards
- add banned-call checks
- run smoke tests before full execution
- capture structured execution results

Acceptance criteria:

- generated code executes only after verification passes.

## Phase 5: Repair Loop

Goal:
Patch failed generated artifacts instead of failing immediately.

Tasks:

- store the failure trace
- generate a patch prompt with:
  - code
  - error
  - run context
- overwrite artifact version or create a new revision
- retry within policy limits

Acceptance criteria:

- a failed generated artifact can be repaired and rerun automatically.

## Phase 6: Planner and Response Integration

Goal:
Make codegen a first-class planning target.

Tasks:

- add planner output types such as:
  - `generate_python_artifact`
  - `generate_shell_artifact`
  - `execute_artifact`
- expose artifact traces and execution summaries in responses
- stream codegen and execution progress live

Acceptance criteria:

- live output shows planning, code generation, verification, execution, and repair steps.

## Phase 7: De-emphasize Static ECG Paths

Goal:
Prevent the runtime from quietly drifting back to ECG-specific built-ins.

Tasks:

- remove ECG-oriented references from generated tool templates
- stop exposing ECG logic as primary runtime tools
- treat `mappers/format_readers/` only as optional codegen reference material, not default execution paths

Acceptance criteria:

- ECG handling is performed through dynamic generated artifacts only.

## 11. Safety Rules

Generated code may:

- read local files within allowed paths
- write only to the autonomy workspace
- emit JSON, Markdown, and local scripts

Generated code may not by default:

- modify repo source files
- install packages
- use unrestricted subprocess calls
- fetch remote data unless enabled by policy
- run destructive shell commands

## 12. Recommended Delivery Order

1. artifact workspace and provenance model
2. generic `generate_python_artifact` path
3. verifier and execution runner
4. repair loop
5. planner integration
6. shell artifact generation
7. removal of ECG-oriented static helper paths

## 13. Definition of Done

This update is complete when CardioMAS can:

- receive a dataset-related query,
- generate a new code artifact for that exact task,
- save it in a reproducible safe location,
- verify and execute it,
- repair it if it fails,
- and answer the user using the resulting artifacts,

while keeping ECG dataset handling out of the built-in tool surface.
