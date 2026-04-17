# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CardioMAS is an **Agentic RAG runtime** for grounded question-answering over ECG/medical datasets. Given a YAML/JSON config that declares knowledge sources, it builds a local corpus, then answers natural-language queries using a plan → execute → aggregate → respond pipeline backed by a local Ollama LLM. Distributed as the `cardiomas` PyPI package.

**Key links:** GitHub: `vlbthambawita/CardioMAS` | HF Dataset: `vlbthambawita/ECGBench` | PyPI: `cardiomas`

## Prerequisites & Setup

```bash
# Install in editable mode with dev tools (uses conda env vLLM)
conda activate vLLM
pip install -e ".[dev]"

# Ollama must be running with a model pulled
ollama pull gemma3:4b && ollama serve

# Copy and fill in tokens (HF_TOKEN, GITHUB_TOKEN optional — only for publishing)
cp .env.example .env
```

## Common Commands

```bash
# CLI — all commands require --config pointing to a YAML/JSON RuntimeConfig
cardiomas build-corpus --config examples/agentic_rag_demo/runtime.yaml
cardiomas build-corpus --config runtime.yaml --force        # force rebuild
cardiomas query "What is the class distribution?" --config runtime.yaml
cardiomas query "..." --config runtime.yaml --live          # stream events
cardiomas query "..." --config runtime.yaml --json          # machine-readable
cardiomas inspect-tools --config runtime.yaml               # list enabled tools
cardiomas check-ollama --config runtime.yaml                # health check

# Tests
pytest tests/                                       # full suite
pytest tests/test_runtime.py -v                     # runtime unit tests
pytest tests/test_autonomy.py -v                    # autonomy/recovery tests
pytest tests/test_cli_fresh.py -v                   # CLI integration tests
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

### Query Pipeline (`agentic/`)

`AgenticRuntime.query_stream(query)` runs four stages in sequence, yielding `AgentEvent` objects:

```
build_corpus → planner → executor → aggregator → responder → QueryResult
```

- **planner** (`planner.py`) — given the query + tool specs, selects which tools to call and with what args (heuristic or Ollama-based)
- **executor** (`executor.py`) — calls selected tools, captures results and errors; yields per-tool events
- **aggregator** (`aggregator.py`) — merges tool results into ranked evidence
- **responder** (`responder.py`) — generates a grounded answer with citations from the aggregated evidence

The non-streaming `.query()` method drains the generator and returns the final `QueryResult`.

### Configuration System (`schemas/config.py`)

All runtime behaviour is driven by a `RuntimeConfig` loaded from YAML or JSON:

```yaml
system_name: MySystem
output_dir: output          # corpus + manifest written here
sources:
  - kind: dataset_dir       # also: local_dir, local_file, web_page, pdf
    path: data/
    label: my-dataset
retrieval:
  mode: hybrid              # bm25 | dense | hybrid
  top_k: 5
  chunk_size: 700
tools:
  enabled: [retrieve_corpus, inspect_dataset, calculate, fetch_webpage]
safety:
  allow_web_fetch: false
  allow_action_tools: false
autonomy:
  enable_code_agents: false
  allow_tool_codegen: false
  max_repair_attempts: 2
```

### Knowledge & Retrieval (`knowledge/`, `retrieval/`)

`build_corpus()` loads all declared sources via `knowledge/loaders.py`, chunks them (`knowledge/chunking.py`), and persists the corpus. Retrieval supports three modes:
- `bm25` — keyword retrieval (`retrieval/bm25.py`)
- `dense` — embedding-based (`retrieval/dense.py`) via Ollama embeddings
- `hybrid` — Reciprocal Rank Fusion of BM25 + dense (`retrieval/hybrid.py`)

### Tools (`tools/`)

Tools are registered at runtime via `tools/registry.py:build_registry()`. Built-in tools:

| Tool name | Module | Purpose |
|---|---|---|
| `retrieve_corpus` | `retrieval_tools.py` | Ranked chunk retrieval |
| `inspect_dataset` | `dataset_tools.py` | Dataset file structure/stats |
| `calculate` | `utility_tools.py` | Safe arithmetic |
| `fetch_webpage` | `research_tools.py` | Web fetch (requires `allow_web_fetch: true`) |

### Autonomy Layer (`autonomy/`)

Optional code-generation and repair loop, enabled via `autonomy` config keys:
- `recovery.py` — `AutonomousToolManager`: wraps tool execution with retry/repair traces
- `verifier.py` — validates generated code/scripts before execution
- `workspace.py` — isolated artifact workspace for generated files
- `policy.py` — access control policies for autonomy actions

Code generation lives in `coding/tool_builder.py` (dynamic tools) and `coding/script_builder.py` (shell scripts).

### Inference (`inference/`)

`inference/ollama.py` provides `OllamaChatClient` and `OllamaEmbeddingClient`, both with `.health_check()`. `llm_factory` is replaced — use `build_chat_client(config.llm)` and `build_embedding_client(config.embeddings)` directly.

### Schemas (`schemas/`)

Pydantic v2 throughout:
- `RuntimeConfig` — full config tree
- `AgentEvent` — streaming event (type, stage, message, data)
- `QueryResult` — final answer + citations + tool results
- `CorpusManifest` — corpus metadata written alongside the corpus

### Python API (`api.py`)

```python
from cardiomas import CardioMAS

api = CardioMAS(config_path="runtime.yaml")
api.build_corpus(force_rebuild=False)
result = api.query("What leads are present?")
for event in api.query_stream("..."):   # AgentEvent generator
    print(event)
api.inspect_tools()
api.check_ollama()
```

### Format Readers (`mappers/format_readers/`)

Readers for ECG-specific formats: WFDB, CSV, EDF, HDF5, NumPy — used by `inspect_dataset` tool.

## Safety

`safety/policy.py` + `safety/permissions.py` + `safety/approvals.py` govern what tools can do. Web fetch and action tools are **off by default** — enable via `safety.allow_web_fetch: true` / `safety.allow_action_tools: true` in the config.
