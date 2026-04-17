# CardioMAS

CardioMAS is a local-first Agentic RAG runtime for dataset understanding and grounded question answering. The current runtime supports deterministic execution, Ollama-backed planning and response generation, and a guarded autonomy layer that can generate small local tools and shell scripts on demand.

## Architecture

The active runtime is organized around:

- `knowledge/` for source loading, chunking, and corpus build
- `retrieval/` for BM25, dense, and hybrid search
- `tools/` for retrieval, dataset inspection, research, and utility tools
- `agentic/` for planning, execution, aggregation, and answering
- `inference/` for Ollama chat and embedding clients
- `autonomy/` and `coding/` for generated-tool workspaces, verification, and bounded repair loops

## Install

```bash
pip install -e ".[dev]"
```

## Ollama Setup

Use the deterministic path if you do not want model inference. To enable Ollama, start the server and pull at least one chat model and one embedding model.

```bash
ollama pull llama3.2
ollama pull embeddinggemma
```

CardioMAS expects Ollama at `http://localhost:11434` unless you override `base_url` in the config.

## Runtime Config

CardioMAS runs from a YAML or JSON config file.

Minimal deterministic example:

```yaml
system_name: CardioMAS
output_dir: runtime_output/demo
sources:
  - kind: dataset_dir
    path: examples/agentic_rag_demo
    label: demo-dataset
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

Ollama + autonomy example:

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
embeddings:
  provider: ollama
  base_url: http://localhost:11434
  model: embeddinggemma
autonomy:
  enable_code_agents: true
  allow_tool_codegen: true
  allow_script_codegen: true
  max_repair_attempts: 2
  workspace_dir: runtime_output/autonomy_workspace
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

Check Ollama connectivity:

```bash
cardiomas check-ollama --config examples/ollama/runtime_local.yaml
```

Query the dataset:

```bash
cardiomas query "What labels are present in the dataset?" \
  --config examples/ollama/runtime_local.yaml
```

Stream step-level events and live Ollama tokens:

```bash
cardiomas query "What labels are present in the dataset?" \
  --config examples/ollama/runtime_local.yaml \
  --live
```

Ask for generated statistics:

```bash
cardiomas query "Give me summary statistics for this dataset." \
  --config examples/ollama/runtime_local.yaml
```

Ask for a generated shell script:

```bash
cardiomas query "Write a shell script to inspect this dataset." \
  --config examples/ollama/runtime_local.yaml
```

Inspect enabled tools:

```bash
cardiomas inspect-tools --config examples/ollama/runtime_local.yaml
```

## Behavior

- If `embeddings:` is configured and Ollama is reachable, corpus chunks are stored with embedding vectors and dense or hybrid retrieval uses them.
- If `llm:` is configured, the responder uses Ollama for grounded answer synthesis.
- If `llm.planner_mode: ollama`, the planner can choose autonomous tools such as `dataset_statistics`, `read_dataset_file`, and `generate_shell_script`.
- If `autonomy:` is enabled, CardioMAS writes generated tool packages and scripts under the autonomy workspace, verifies them, and records repair traces in query results.
- Generated Python tools are limited to a safe import set and are re-generated when verification fails.
- Shell scripts are written but not executed automatically.
- `cardiomas query --live` streams step events such as planning, tool start/finish, repair traces, and raw LLM token chunks. Planner and responder token streams are structured JSON because those stages currently use JSON-constrained prompts.

## PTB-XL Example

The repo includes an Ollama-ready PTB-XL config at `examples/ollama/runtime_ptbxl.yaml`.

```bash
cardiomas check-ollama --config examples/ollama/runtime_ptbxl.yaml
cardiomas build-corpus --force --config examples/ollama/runtime_ptbxl.yaml
cardiomas query "Summarize the PTB-XL dataset structure and metadata columns." \
  --config examples/ollama/runtime_ptbxl.yaml
cardiomas query "Give me summary statistics for this dataset." \
  --config examples/ollama/runtime_ptbxl.yaml
```

## Python API

```python
from cardiomas import CardioMAS

cm = CardioMAS(config_path="examples/ollama/runtime_local.yaml")
cm.build_corpus()
result = cm.query("Give me summary statistics for this dataset.")
print(result["answer"])
print(result["repair_traces"])

for event in cm.query_stream("What labels are present in the dataset?"):
    print(event["type"], event["stage"], event.get("message", ""), event.get("content", ""))
```
