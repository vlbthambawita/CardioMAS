# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CardioMAS is a locally-runnable multi-agent system that analyzes ECG datasets, generates reproducible train/val/test splits, and publishes split manifests (record IDs only — no raw data) to `vlbthambawita/ECGBench` on HuggingFace. It is distributed as the `cardiomas` PyPI package.

**Key links:** GitHub: `vlbthambawita/CardioMAS` | HF Dataset: `vlbthambawita/ECGBench` | PyPI: `cardiomas`

## Prerequisites & Setup

```bash
# Install in editable mode with dev tools
pip install -e ".[dev]"

# Ollama must be running with a model pulled (default: llama3.1:8b)
ollama pull llama3.1:8b && ollama serve

# Copy and fill in tokens
cp .env.example .env
```

## Common Commands

```bash
# Run the CLI
cardiomas analyze https://physionet.org/content/ptb-xl/1.0.3/
cardiomas analyze /local/path --local-path /local/path --dry-run
cardiomas status ptb-xl
cardiomas list
cardiomas verify ptb-xl
cardiomas version

# Tests
pytest tests/                          # all 26 tests
pytest tests/test_splitters.py -v      # splitter unit tests
pytest tests/test_security.py -v       # security/PII tests
pytest tests/test_cli.py -v            # CLI integration tests

# Lint / format
ruff check src/ tests/
ruff format src/ tests/

# Build package (version comes from git tag via setuptools-scm)
python -m build

# Publish a release
git tag v0.1.0 && git push origin v0.1.0   # triggers GitHub Actions → PyPI
```

## Architecture

The package lives entirely under `src/cardiomas/`. The old top-level `agents/`, `graph/`, `main.py`, `orchestrator.py` have been removed.

### LangGraph Pipeline (`graph/workflow.py`)

`run_pipeline(source, options)` builds a `StateGraph(dict)` and runs it. The seven nodes are LangGraph-wrapped calls to agent functions in `agents/`. The shared state carrier is `GraphState` (Pydantic, serialized to/from dict for LangGraph).

**Node sequence:**

```
orchestrator → [HF cache hit?] → return_existing → END
                   ↓ (no hit)
             discovery → paper → analysis → splitter → security
                                                           ↓
                                              [audit passed?] → publisher → END
                                              [audit failed] → end_with_error → END
                                              [dry-run]      → end_dry_run → END
```

### Agents (`agents/`)

Each agent function takes and returns `GraphState`. They use `agents/base.py:run_agent()` which loads an `.md` skill file from `skills/` as the LLM system prompt, then invokes the LLM.

| Agent | Key responsibility |
|---|---|
| `orchestrator` | Check HF cache; short-circuit if already published |
| `discovery` | Identify dataset from URL/path; populate `DatasetInfo` |
| `paper` | Find & parse the dataset paper; extract split methodology |
| `analysis` | Scan files, parse CSV metadata, compute statistics |
| `splitter` | Generate deterministic splits; build `SplitManifest` |
| `security` | PII scan, raw-data check, patient-leakage check |
| `publisher` | Push to `vlbthambawita/ECGBench`; update GitHub README |

### Reproducibility Guarantee

`splitters/strategies.py:deterministic_split()` sorts record IDs, computes SHA-256(sorted_ids + seed + strategy) as the RNG seed, then shuffles+slices. Same inputs always yield identical splits.

### Schemas (`schemas/`)

All state is Pydantic v2:
- `GraphState` — full pipeline state (dataset_info, proposed_splits, security_audit, etc.)
- `DatasetInfo` — dataset metadata with `DatasetSource` enum
- `SplitManifest` — split output + `ReproducibilityConfig`
- `SecurityAudit` — audit results with blocking issues

### Tools (`tools/`)

LangChain `@tool`-decorated functions called directly by agents. Five modules: `data_tools`, `research_tools`, `split_tools`, `publishing_tools`, `security_tools`. All tools return dicts (never raise — errors land in `"error"` key).

### Dataset Registry (`datasets/registry.yaml`)

YAML catalog of 6 known ECG datasets (PTB-XL, MIMIC-IV-ECG, CPSC-2018, Georgia, Chapman-Shaoxing, CODE-15). The `DatasetRegistry` singleton loads it at import time. New datasets can be registered programmatically via `get_registry().register(DatasetInfo(...))`.

### LLM (`llm_factory.py`)

`get_llm(prefer_cloud=False)` returns a LangChain `BaseChatModel`. Default: `ChatOllama` (local). Optional cloud fallback via `CLOUD_LLM_PROVIDER=openai|anthropic` in `.env`. Ollama health-check runs on every call to `get_local_llm()`.

### CLI (`cli/main.py`) and Python API (`api.py`)

CLI uses `typer`. All commands support `--json` for machine-readable output. The `CardioMAS` class in `api.py` is the public Python library API (`from cardiomas import CardioMAS`).

## Publishing a New Version

```bash
git tag v0.1.0
git push origin v0.1.0
# GitHub Actions (.github/workflows/publish.yml) builds and publishes to PyPI via
# OIDC trusted publishing (no PYPI_API_TOKEN needed after one-time PyPI setup).
```

## Environment Variables

See `.env.example` for the full list. Key ones:

| Variable | Purpose |
|---|---|
| `OLLAMA_MODEL` | Local model name (default `llama3.1:8b`) |
| `HF_TOKEN` | Required for pushing to `vlbthambawita/ECGBench` |
| `GITHUB_TOKEN` | Required for updating `vlbthambawita/CardioMAS` README |
| `CARDIOMAS_SEED` | Global reproducibility seed (default `42`) |
