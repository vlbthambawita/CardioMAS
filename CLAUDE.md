# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CardioMAS is an **Agentic RAG runtime** for grounded question-answering over ECG/medical datasets. Given a YAML/JSON config that declares knowledge sources, it builds a local corpus, then answers natural-language queries using a ReAct think→act→observe loop (or a simpler linear plan→execute→respond pipeline) backed by a local Ollama LLM. Distributed as the `cardiomas` PyPI package.

**Key links:** GitHub: `vlbthambawita/CardioMAS` | HF Dataset: `vlbthambawita/ECGBench` | PyPI: `cardiomas`

## Prerequisites & Setup

```bash
# Install in editable mode with dev tools (uses conda env vLLM)
conda activate vLLM
pip install -e ".[dev]"

# Ollama must be running with a model pulled
ollama pull gemma4:latest && ollama serve

# Copy and fill in tokens (HF_TOKEN, GITHUB_TOKEN optional — only for publishing)
cp .env.example .env
```

## Common Commands

```bash
# CLI — all commands require --config pointing to a YAML/JSON RuntimeConfig
cardiomas build-corpus --config examples/ollama/ptbxl_react.yaml
cardiomas build-corpus --config runtime.yaml --force        # force rebuild
cardiomas query "What is the class distribution?" --config runtime.yaml
cardiomas query "..." --config runtime.yaml --live          # stream events + LLM tokens live
cardiomas query "..." --config runtime.yaml --json          # machine-readable
cardiomas inspect-tools --config runtime.yaml               # list enabled tools
cardiomas check-ollama --config runtime.yaml                # health check

# Tests
pytest tests/                                       # full suite (73 tests, ~0.4s)
pytest tests/test_runtime.py -v                     # runtime unit tests
pytest tests/test_autonomy.py -v                    # autonomy/recovery tests
pytest tests/test_cli_fresh.py -v                   # CLI integration tests
pytest tests/test_react_agent.py -v                 # ReAct agent tests
pytest tests/test_corpus.py::test_build_bm25 -v     # single test

# Lint / format
ruff check src/ tests/
ruff format src/ tests/

# Build package (version from git tag via setuptools-scm)
python -m build
git tag v0.2.0 && git push origin v0.2.0   # triggers GitHub Actions → PyPI
```

## Architecture

All source lives under `src/cardiomas/`. Entry point is `AgenticRuntime` in `agentic/runtime.py`.

### Query Pipelines (`agentic/`)

`AgenticRuntime.query_stream(query)` dispatches to one of two pipelines based on `agent.mode`:

**ReAct mode** (`agent.mode: react`) — recommended; uses `react_agent.py`:
```
build_corpus → route → [optional upfront plan] → ReAct loop → aggregate → respond → QueryResult
```
Each ReAct iteration: orchestrator LLM picks a tool → tool executes → observation added → repeat until `action=answer`. The orchestrator call streams tokens live (visible in `--live` mode).

**Linear mode** (`agent.mode: linear`) — legacy default:
```
build_corpus → planner → executor → aggregator → responder → QueryResult
```

The `.query()` method drains the generator and returns the final `QueryResult`. Both modes yield `AgentEvent` objects for streaming.

### ReAct++ Extensions (`agent` config block)

Four optional flags in `AgentConfig` that extend the base ReAct loop:

| Flag | Default | Effect |
|---|---|---|
| `upfront_planning` | `false` | One LLM call before the loop generates an ordered tool sequence (soft guide) |
| `step_reflection` | `false` | Each orchestrator response includes a `reflection` field (`making_progress`/`stuck`/`sufficient`); `sufficient` stops the loop, 2× `stuck` injects a recovery hint |
| `scratchpad` | `true` | First line of each tool result is distilled into a "Running knowledge" block shown at every iteration |
| `tool_verification` | `true` | Pre-execution Python check validates path existence and non-empty task before calling the tool |

### Configuration System (`schemas/config.py`)

All runtime behaviour is driven by a `RuntimeConfig` loaded from YAML or JSON. Key sections:

```yaml
system_name: MySystem
output_dir: output
sources:
  - kind: dataset_dir       # also: local_dir, local_file, web_page, pdf
    path: data/
    label: my-dataset
  - kind: web_page
    url: https://physionet.org/content/ptb-xl/1.0.3/
    label: ptbxl-physionet-page
retrieval:
  mode: hybrid              # bm25 | dense | hybrid
  top_k: 5
agent:
  mode: react
  max_iterations: 10
  upfront_planning: true
  step_reflection: true
tools:
  enabled: [list_folder_structure, read_wfdb_dataset, read_dataset_website,
            retrieve_corpus, inspect_dataset, calculate, fetch_webpage]
autonomy:
  enable_code_agents: false
  dataset_mode: script_only   # script_only | agentic
  execute_for_answer: false
  max_repair_attempts: 2
```

### Knowledge & Retrieval (`knowledge/`, `retrieval/`)

`build_corpus()` loads all declared sources via `knowledge/loaders.py`, chunks them, and persists the corpus. Retrieval modes: `bm25`, `dense` (Ollama embeddings), `hybrid` (Reciprocal Rank Fusion).

### Tools (`tools/`)

Registered at runtime via `tools/registry.py:build_registry()`. All tool wrappers use `**kwargs` to normalize LLM argument spelling variants (e.g., `path`/`dataset_path`/`directory` all accepted).

| Tool name | Module | Purpose |
|---|---|---|
| `list_folder_structure` | `dataset_tools.py` | ASCII tree of a directory with file sizes and CSV headers |
| `read_wfdb_dataset` | `wfdb_tools.py` | Scan PhysioNet WFDB ECG records (`.hea`/`.dat`/`.atr`), report leads/sampling rate/duration |
| `read_dataset_website` | `web_dataset_tools.py` | Fetch and parse dataset documentation pages (PhysioNet, HuggingFace, Zenodo); resolves source labels to URLs from config |
| `retrieve_corpus` | `retrieval_tools.py` | Ranked chunk retrieval from the local corpus |
| `inspect_dataset` | `dataset_tools.py` | File structure stats, extension counts, CSV headers |
| `calculate` | `utility_tools.py` | Safe arithmetic |
| `fetch_webpage` | `research_tools.py` | Raw web fetch (requires `allow_web_fetch: true`) |
| `generate_python_artifact` | via `coding/` | LLM-generated Python script for dataset analysis |
| `generate_shell_artifact` | via `coding/` | LLM-generated shell script |

`tools/pre_exec_verifier.py:verify_tool_args()` is called before every tool execution (when `tool_verification: true`) — checks path-based tools for existence and code-gen tools for non-empty task.

### Aggregator (`agentic/aggregator.py`)

Merges tool results into ranked `EvidenceChunk` list. Tool summaries are also converted to synthetic `EvidenceChunk`s (score=0.95) so the responder and answer grader see tool output as evidence — this prevents false "hallucination" verdicts when tools produce the answer directly.

### Responder (`agentic/responder.py`)

Streams LLM tokens via `chat_client.chat_stream()`, emitting `llm_token` events. Falls back to `chat_client.chat()` if streaming returns empty. For `script_output` evidence (from executed scripts), the prompt presents the stdout prominently for synthesis rather than treating it as a citation.

### Memory (`memory/`)

- `session.py` — per-query tool call history within a single runtime session
- `persistent.py` — file-backed answer cache with token-overlap similarity search (`memory_mode: persistent`)
- `scratchpad.py` — `Scratchpad` class: collects one distilled fact per tool call, shown to orchestrator as "Running knowledge" block each iteration

### Autonomy Layer (`autonomy/`)

Optional code-generation and repair loop, enabled via `autonomy` config keys:
- `recovery.py` — `AutonomousToolManager`: wraps tool execution with retry/repair traces
- `verifier.py` — validates generated code/scripts before execution
- `workspace.py` — isolated artifact workspace for generated files
- `policy.py` — access control policies for autonomy actions

Code generation lives in `coding/tool_builder.py` (dynamic tools) and `coding/script_builder.py` (shell scripts).

### Inference (`inference/`)

`inference/ollama.py` provides `OllamaChatClient` and `OllamaEmbeddingClient`. Use `build_chat_client(config.llm)` and `build_embedding_client(config.embeddings)` — do not instantiate directly. `ChatClient.chat_stream()` has a base-class fallback that wraps `chat()` in a single chunk, so all clients support streaming.

### Schemas (`schemas/`)

Pydantic v2 throughout:
- `RuntimeConfig` — full config tree (see `schemas/config.py`)
- `AgentEvent` — streaming event with `type`, `stage`, `message`, `data`, `content` fields
- `QueryResult` — final answer + citations + tool results + `react_steps`
- `CorpusManifest` — corpus metadata persisted alongside the corpus

### Python API (`api.py`)

```python
from cardiomas import CardioMAS

api = CardioMAS(config_path="runtime.yaml")
api.build_corpus(force_rebuild=False)
result = api.query("What leads are present?")
for event in api.query_stream("..."):   # AgentEvent generator
    print(event)
```

## Safety

`safety/policy.py` + `safety/permissions.py` + `safety/approvals.py` govern what tools can do. Web fetch and action tools are **off by default** — enable via `safety.allow_web_fetch: true` / `safety.allow_action_tools: true` in the config.

## Live Streaming (`--live`)

Events emitted during a query:

| `type` | When |
|---|---|
| `status` | Stage transitions, grader verdicts, tool summaries |
| `tool_started` / `tool_finished` | Before/after each tool call |
| `llm_stream_start` / `llm_token` / `llm_stream_end` | Orchestrator LLM tokens (each ReAct iteration) and responder LLM tokens |
| `repair_trace` | Autonomy repair attempts |
| `final_result` | Contains the complete `QueryResult` |
