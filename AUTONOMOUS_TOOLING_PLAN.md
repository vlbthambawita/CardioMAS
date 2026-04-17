# CardioMAS Autonomous Tooling Plan

Date: 2026-04-17  
Scope: add coding agents that can create, repair, and validate tools during runtime

## 1. Feasibility Decision

Yes, this is possible, but only as a guarded system.

CardioMAS can be extended so that model-driven coding agents generate small tools, statistical analysis helpers, and shell scripts on demand, then retry the failed task after validation. The important constraint is that this must not become unconstrained code execution. Generated code needs a bounded workspace, explicit policies, verification, and approval gates for risky actions.

## 2. Goal

Extend the current Agentic RAG runtime so it can:

- create dataset-reading tools when an existing tool is missing,
- generate statistical analysis code for unfamiliar datasets,
- write shell scripts for repeatable local workflows,
- diagnose tool failures,
- patch or replace failing tools,
- verify the fix,
- and retry the original step automatically.

## 3. Target Capabilities

1. On-demand dataset readers
   - generate readers for CSV, WFDB, EDF, HDF5, NumPy, and previously unseen local layouts
   - reuse `src/cardiomas/mappers/format_readers/` as seed implementations where possible

2. Statistical analysis agents
   - create analysis helpers for class counts, missingness, shape summaries, split statistics, and signal metadata
   - produce plots or reports only when explicitly requested

3. Shell script agents
   - generate small scripts for corpus build, preprocessing, batch validation, and reproducible dataset inspection
   - keep scripts under a controlled output workspace, not arbitrary repo paths

4. Self-healing execution
   - detect tool failure
   - classify cause: config issue, missing dependency, unsupported format, coding bug, or bad arguments
   - attempt bounded repair
   - verify
   - retry once the repair passes checks

## 4. Core Design Rules

1. Generated code must start in an isolated workspace.
2. Repo code should not be modified by default during runtime.
3. Code writes, shell execution, and dependency installation must have separate approval policies.
4. A failed tool should trigger diagnosis before regeneration.
5. Every repair attempt must leave a trace: prompt, files written, tests run, and retry result.

## 5. Proposed Architecture

```text
User query
  ->
Planner
  ->
Tool execution
  ->
Failure detector
  ->
Repair planner
  ->
Coding / script agent
  ->
Verification runner
  ->
Tool registry update
  ->
Retry original task
```

## 6. Repository Changes

Add these modules:

```text
src/cardiomas/
  autonomy/
    policy.py
    workspace.py
    recovery.py
    verifier.py
  coding/
    prompts.py
    tool_builder.py
    script_builder.py
    repair_agent.py
  execution/
    runner.py
    sandbox.py
    artifacts.py
```

Extend these existing modules:

- `agentic/planner.py` for “build tool” and “repair tool” decisions
- `agentic/executor.py` for failure capture and retry flow
- `tools/registry.py` for loading generated tools
- `schemas/config.py` for autonomy policy config
- `schemas/runtime.py` for repair traces and retry history
- `safety/permissions.py` and `safety/approvals.py` for code and shell gates

## 7. Config Additions

Example:

```yaml
autonomy:
  enable_code_agents: true
  allow_tool_codegen: true
  allow_script_codegen: true
  max_repair_attempts: 2
  workspace_dir: runtime_output/autonomy_workspace
  require_approval_for_repo_writes: true
  require_approval_for_shell_execution: true
  require_approval_for_installs: true
  allowed_shell_prefixes:
    - python
    - bash
    - sh
  allowed_python_modules:
    - csv
    - json
    - math
    - statistics
    - pathlib
    - numpy
    - pandas
    - ast
    - wfdb
```

## 8. Runtime Strategy

### Path A: existing tool works

- use the normal planner, executor, and responder flow

### Path B: tool missing

- planner delegates to a tool-builder agent
- agent creates a small tool package in the autonomy workspace
- verifier runs smoke tests against the target dataset
- registry loads the generated tool
- original task is retried

### Path C: tool fails

- recovery agent inspects the error and execution context
- decide whether to:
  - patch arguments,
  - patch code,
  - generate a replacement tool,
  - or stop and ask for approval
- verifier runs again before retry

## 9. Implementation Phases

## Phase 1: Failure and Repair Framework

Goal:
Add structured failure capture and repair policy before any code generation.

Tasks:

- define failure classes and retry limits
- add repair traces to runtime results
- add autonomy policy config and approval switches
- implement a repair decision object

Acceptance criteria:

- CardioMAS can distinguish recoverable tool failures from hard stops.

## Phase 2: Workspace and Generated Tool Contract

Goal:
Create a safe place and format for generated tools.

Tasks:

- add workspace management under `runtime_output/`
- define a generated tool contract:
  - `tool.py`
  - `tool_spec.json`
  - `tests/`
  - `README.md`
- load generated tools dynamically through the registry

Acceptance criteria:

- the runtime can create, load, and unload generated tools without editing `src/`.

## Phase 3: Coding Agent for Dataset Readers

Goal:
Generate dataset readers when existing tools are insufficient.

Tasks:

- create prompts for reader synthesis and patch generation
- use existing `mappers/format_readers/` as reference context
- support first targets:
  - CSV metadata readers
  - WFDB header readers
  - EDF structure readers
  - HDF5 key/shape readers
  - NumPy array inspectors
- add smoke tests that run against a real sample path

Acceptance criteria:

- a missing reader can be generated, validated, and used in the same session.

## Phase 4: Statistical Analysis Agent

Goal:
Generate analysis tools for dataset profiling.

Tasks:

- implement prompts for summary-statistics tools
- support outputs for:
  - row counts
  - class counts
  - missing values
  - numeric summaries
  - shape and duration summaries
  - split consistency checks
- emit structured JSON plus optional Markdown report

Acceptance criteria:

- CardioMAS can generate and run a new stats tool against a dataset and cite its output.

## Phase 5: Shell Script Agent

Goal:
Generate controlled scripts for repeatable local tasks.

Tasks:

- support script generation in the autonomy workspace only
- add shell script linting and dry-run validation
- restrict execution to approved command prefixes
- support first script types:
  - dataset scan
  - batch metadata extraction
  - corpus rebuild
  - report packaging

Acceptance criteria:

- scripts can be generated and inspected safely before execution.

## Phase 6: Self-Healing Loop

Goal:
Automatically repair failing tools and retry.

Tasks:

- add `agentic/recovery.py`
- route tool exceptions into diagnosis
- generate patch candidates
- run verification
- retry the original tool call with the repaired tool
- stop after bounded attempts and return a clear failure trace

Acceptance criteria:

- a recoverable bug in a generated tool can be fixed and retried automatically.

## Phase 7: Promotion and Persistence

Goal:
Promote useful generated tools from workspace artifacts into stable repo modules.

Tasks:

- define promotion rules:
  - passed tests
  - repeated reuse
  - readable code
  - no unsafe calls
- add a manual or approval-based promotion command
- copy promoted tools into a stable repo package such as `src/cardiomas/tools/generated/`

Acceptance criteria:

- high-value generated tools can become maintained first-class tools.

## Phase 8: Evaluation

Goal:
Measure whether autonomy improves the system instead of creating instability.

Tasks:

- benchmark:
  - unsupported dataset format tasks
  - broken tool repair tasks
  - statistical analysis generation tasks
  - shell script generation tasks
- track:
  - repair success rate
  - retry success rate
  - verification pass rate
  - unsafe-action rejection rate
  - latency and token cost

Acceptance criteria:

- the autonomous path improves task completion without weakening safety.

## 10. Safety Boundaries

These actions should never run without policy approval:

- package installation
- repo writes outside the autonomy workspace
- destructive shell commands
- network fetch during repair unless explicitly enabled
- arbitrary subprocess execution outside allowed prefixes

Generated code should be denied access to hidden escalation paths by default.

## 11. Recommended Delivery Order

1. autonomy policy, traces, and failure model
2. workspace and generated tool loader
3. dataset-reader coding agent
4. statistical analysis agent
5. shell script agent
6. self-healing retry loop
7. promotion flow
8. evaluation suite

## 12. Definition of Done

This plan is complete when CardioMAS can:

- detect a missing or failing tool,
- decide whether repair is safe and allowed,
- generate or patch a small tool in a controlled workspace,
- verify the new tool against the target dataset,
- retry the original task,
- and provide a full trace of what code was written, what tests ran, and why the retry succeeded or failed.
