# CardioMAS

CardioMAS is a local-first Agentic RAG runtime for dataset understanding and grounded question answering. The current runtime supports deterministic execution, Ollama-backed planning and response generation, and a guarded autonomy layer that generates query-specific Python and shell artifacts on demand.

## Architecture

The active runtime is organized around:

- `knowledge/` for source loading, chunking, and corpus build
- `retrieval/` for BM25, dense, and hybrid search
- `tools/` for retrieval, dataset inspection, research, and utility tools
- `agentic/` for planning, execution, aggregation, and answering
- `inference/` for Ollama chat and embedding clients
- `autonomy/` and `coding/` for generated artifact workspaces, verification, and bounded repair loops

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

Script-first dataset query example (Phase 1 — write script, user runs it):

```yaml
system_name: CardioMAS
output_dir: runtime_output/script-mode
sources:
  - kind: dataset_dir
    path: /path/to/your/dataset
    label: my-dataset
scripts_dir: runtime_output/scripts        # flat, user-accessible script location
autonomy:
  enable_code_agents: true
  allow_tool_codegen: true
  dataset_mode: script_only                # route all dataset queries to script generation
  max_repair_attempts: 2
tools:
  enabled:
    - generate_python_artifact
```

Script-first with auto-execution (Phase 2 — run script, synthesize answer from output):

```yaml
autonomy:
  enable_code_agents: true
  allow_tool_codegen: true
  dataset_mode: script_only
  execute_for_answer: true                 # execute the script and feed stdout to the responder
  execution_timeout_seconds: 60            # subprocess timeout in seconds
  require_approval_for_shell_execution: false   # must be false for auto-execution
  max_repair_attempts: 2
scripts_dir: runtime_output/scripts
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

## dataset_mode: script_only

When `autonomy.dataset_mode` is set to `script_only`, every query that involves a configured dataset generates a standalone Python script instead of attempting to answer directly from corpus retrieval or inspection tools.

**Phase 1 (default)** — the script is written to `scripts_dir` and the agent tells you where it is and how to run it. No execution happens inside the agent.

```
runtime_output/scripts/
  count-unique-patients_82c4092a.py   ← run with: python count-unique-patients_82c4092a.py
  outputs/
    results.json                      ← written when you run the script
```

**Phase 2** — set `execute_for_answer: true` (and `require_approval_for_shell_execution: false`) to have the agent run the script automatically, capture its stdout, and synthesize a direct answer from the computed output. The script is still written to `scripts_dir` so you can inspect or re-run it. If execution fails, the agent regenerates the script up to `max_repair_attempts` times using the error output as context.

### LLM-driven code synthesis

When `llm:` is configured, each query generates a **different, purpose-built script** using a three-step pipeline:

1. **Dataset discovery** — scans the dataset directory and reads actual column names from CSV/TSV files (no LLM)
2. **Computation planning** — LLM reasons about which files and columns to use for the specific query
3. **Code synthesis** — LLM writes the complete Python script, grounded in real column names
4. **Repair** — on execution failure, LLM sees the broken code and the real error message and fixes it

```yaml
llm:
  provider: ollama
  base_url: http://localhost:11434
  planner_mode: heuristic       # planner stays heuristic; only code synthesis calls Ollama
  model: gemma3:4b
  code_max_tokens: 4000         # scripts need more tokens than 800-token JSON responses
  code_temperature: 0.2
```

> `code_max_tokens` defaults to 4000. The standard `max_tokens: 800` is for the JSON-constrained planner and responder — it is too small for a full Python script.

When `llm:` is **not** configured, a generic template is used as fallback (same code structure for every query).

Generated scripts:

- Are standalone (`python script.py` works with no arguments).
- Hardcode `DATASET_PATH` and `OUTPUT_DIR` from the config.
- Write `results.json` to the `outputs/` subdirectory.
- When LLM-generated: read the specific files and columns required by the query.
- Are verified with `ast.parse()` before being saved.

## Behavior

- If `embeddings:` is configured and Ollama is reachable, corpus chunks are stored with embedding vectors and dense or hybrid retrieval uses them.
- If `llm:` is configured, the responder uses Ollama for grounded answer synthesis.
- If `llm.planner_mode: ollama`, the planner can choose autonomous tools such as `generate_python_artifact` and `generate_shell_artifact`.
- If `autonomy.dataset_mode: script_only`, the planner routes all dataset queries exclusively to `generate_python_artifact`; corpus retrieval and inspection tools are not used for those queries.
- If `autonomy:` is enabled (agentic mode), CardioMAS writes each generated artifact under `autonomy_workspace/sessions/<session_id>/<artifact_slug>/`, stores `prompt.json` and `context.json`, verifies the code, records per-attempt run logs, and reports repair traces in query results.
- `generate_python_artifact` is the main dynamic path for dataset file reading, metadata extraction, and statistical analysis; the runtime does not expose ECG-specific built-in analysis tools.
- Generated Python artifacts are limited to a safe import set and are re-generated when verification fails.
- Shell artifacts are saved in the same workspace and execute only when policy allows it.
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

With `dataset_mode: script_only` the agent generates a ready-to-run script instead of guessing the answer:

```bash
cardiomas query "How many unique patients are in this dataset?" \
  --config examples/ollama/runtime_ptbxl.yaml
# → Scripts Generated
# →   count-unique-patients_82c4092a.py
# →   Path: runtime_output/scripts/count-unique-patients_82c4092a.py
# →   Run:  python runtime_output/scripts/count-unique-patients_82c4092a.py
python runtime_output/scripts/count-unique-patients_82c4092a.py
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
