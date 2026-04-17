# CardioMAS

CardioMAS is a local-first Agentic RAG runtime for dataset understanding and grounded question answering. The current rebuild supports a deterministic baseline and an Ollama-backed path for local embeddings, grounded response generation, and optional model-based planning.

## Architecture

The active runtime is organized around:

- `knowledge/` for source loading, chunking, and corpus build
- `retrieval/` for BM25, dense, and hybrid search
- `tools/` for retrieval, dataset inspection, web fetch, and utility tools
- `agentic/` for planning, execution, aggregation, and answering
- `inference/` for Ollama chat and embedding clients
- `memory/` and `safety/` for session state and execution policy

## Install

```bash
pip install -e ".[dev]"
```

## Ollama Setup

Use the local deterministic path if you do not want model inference. To enable Ollama, start the server and pull at least one chat model and one embedding model.

Example:

```bash
ollama pull llama3.2
ollama pull embeddinggemma
```

CardioMAS expects Ollama at `http://localhost:11434` unless you override `base_url` in the config.

## Runtime Config

CardioMAS runs from a YAML or JSON config file.

Deterministic example:

```yaml
system_name: CardioMAS
output_dir: runtime_output/demo
sources:
  - kind: dataset_dir
    path: examples/agentic_rag_demo
    label: demo-dataset
  - kind: local_file
    path: examples/agentic_rag_demo/notes.md
    label: demo-notes
retrieval:
  mode: hybrid
  top_k: 4
tools:
  enabled:
    - retrieve_corpus
    - inspect_dataset
    - calculate
safety:
  allow_web_fetch: false
```

Ollama example:

```yaml
system_name: CardioMAS
output_dir: runtime_output/ollama-local
sources:
  - kind: dataset_dir
    path: examples/agentic_rag_demo/data
    label: demo-dataset
  - kind: local_file
    path: examples/agentic_rag_demo/notes.md
    label: demo-notes
retrieval:
  mode: hybrid
  top_k: 5
llm:
  provider: ollama
  base_url: http://localhost:11434
  planner_mode: ollama
  model: llama3.2
  temperature: 0.1
  max_tokens: 800
embeddings:
  provider: ollama
  base_url: http://localhost:11434
  model: embeddinggemma
  batch_size: 16
tools:
  enabled:
    - retrieve_corpus
    - inspect_dataset
    - calculate
safety:
  allow_web_fetch: false
```

Ready-to-run examples:

- `examples/agentic_rag_demo/runtime.yaml`
- `examples/ollama/runtime_local.yaml`
- `examples/ollama/runtime_ptbxl.yaml`

## Commands

Build the corpus:

```bash
cardiomas build-corpus --config examples/ollama/runtime_local.yaml
```

Check Ollama connectivity and visible local models:

```bash
cardiomas check-ollama --config examples/ollama/runtime_local.yaml
```

Run a grounded query:

```bash
cardiomas query "What labels are present in the dataset?" \
  --config examples/ollama/runtime_local.yaml
```

Inspect enabled tools:

```bash
cardiomas inspect-tools --config examples/ollama/runtime_local.yaml
```

Print the resolved config:

```bash
cardiomas show-config --config examples/ollama/runtime_local.yaml
```

## Behavior

- If `embeddings:` is configured and Ollama is reachable, corpus chunks are stored with embedding vectors and dense or hybrid retrieval uses them.
- If `llm:` is configured, the responder uses Ollama for grounded answer synthesis.
- If `llm.planner_mode: ollama`, the planner asks Ollama for a structured tool plan and falls back to the heuristic planner if the response is invalid.
- If Ollama is unavailable, query-time planning and response generation fall back to deterministic behavior, and corpus build falls back to lexical-only retrieval with a warning.

## PTB-XL Example

The repo includes an Ollama-ready PTB-XL config at `examples/ollama/runtime_ptbxl.yaml`. Build and query it with:

```bash
cardiomas check-ollama --config examples/ollama/runtime_ptbxl.yaml
cardiomas build-corpus --config examples/ollama/runtime_ptbxl.yaml
cardiomas query "Summarize the PTB-XL dataset structure and metadata." \
  --config examples/ollama/runtime_ptbxl.yaml
```

## Python API

```python
from cardiomas import CardioMAS

cm = CardioMAS(config_path="examples/ollama/runtime_local.yaml")
print(cm.check_ollama())
cm.build_corpus()
result = cm.query("What labels are present in the dataset?")
print(result["answer"])
```
