# CardioMAS

CardioMAS is a fresh Agentic RAG runtime for dataset understanding and grounded question answering.

The rebuild follows [AGENTIC_RAG_ALIGNMENT_PLAN.md](AGENTIC_RAG_ALIGNMENT_PLAN.md) and replaces the old workflow-specific architecture with a smaller core:

- `knowledge/` for ingestion and corpus build
- `retrieval/` for BM25, dense, and hybrid ranking
- `tools/` for retrieval, dataset inspection, research, and utility tools
- `agentic/` for planning, execution, aggregation, and grounded responses
- `memory/` and `safety/` for session state and permissions

## Install

```bash
pip install -e ".[dev]"
```

## Runtime Config

CardioMAS runs from a YAML or JSON config file.

Example:

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

## Commands

Build the corpus:

```bash
cardiomas build-corpus --config examples/agentic_rag_demo/runtime.yaml
```

Run a grounded query:

```bash
cardiomas query "What labels are present in the demo dataset?" \
  --config examples/agentic_rag_demo/runtime.yaml
```

Inspect enabled tools:

```bash
cardiomas inspect-tools --config examples/agentic_rag_demo/runtime.yaml
```

## Python API

```python
from cardiomas import CardioMAS

cm = CardioMAS(config_path="examples/agentic_rag_demo/runtime.yaml")
cm.build_corpus()
result = cm.query("How many files are in the dataset?")
print(result["answer"])
```
