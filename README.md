# CardioMAS — Cardio Multi-Agent System

[![CI](https://github.com/vlbthambawita/CardioMAS/actions/workflows/ci.yml/badge.svg)](https://github.com/vlbthambawita/CardioMAS/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cardiomas)](https://pypi.org/project/cardiomas/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HF-vlbthambawita%2FECGBench-yellow)](https://huggingface.co/datasets/vlbthambawita/ECGBench)

A locally-runnable multi-agent system that analyzes ECG datasets and generates reproducible train/validation/test splits. Outputs are saved locally by default. Publishing to [vlbthambawita/ECGBench](https://huggingface.co/datasets/vlbthambawita/ECGBench) on HuggingFace is an explicit opt-in step that requires write access.

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.com/) running locally with a model pulled (default: `llama3.1:8b`)

```bash
ollama pull llama3.1:8b
ollama serve
pip install cardiomas
```

## Quick Start

```bash
# Analyze a dataset — results saved to ./output/ptb-xl/
cardiomas analyze https://physionet.org/content/ptb-xl/1.0.3/

# Use a local directory
cardiomas analyze /data/ptb-xl/

# Analyze and push to HuggingFace in one step (requires HF_TOKEN)
cardiomas analyze /data/ptb-xl/ --push
```

After `analyze`, the following files are written locally:

```
output/
└── ptb-xl/
    ├── splits.json           # train/val/test record IDs + reproducibility config
    ├── split_metadata.json   # seed, strategy, version, timestamp
    └── analysis_report.md    # LLM-generated dataset analysis
```

## CLI Reference

### `cardiomas analyze`

Analyze a dataset and save splits locally.

```
cardiomas analyze DATASET_SOURCE [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--local-path PATH` | | Explicit local data path (skips download) |
| `--output-dir PATH` | `output` | Where to save results |
| `--seed INT` | `42` | Reproducibility seed |
| `--custom-split SPEC` | | e.g. `train:0.7,val:0.15,test:0.15` |
| `--stratify-by FIELD` | | Metadata field to stratify splits by |
| `--ignore-official` | | Ignore official splits, generate fresh |
| `--push` | | Also push to HuggingFace (requires `HF_TOKEN`) |
| `--force-reanalysis` | | Re-run even if already analyzed |
| `--use-cloud-llm` | | Use cloud LLM instead of local Ollama |
| `--verbose` / `-v` | | Stream agent reasoning live |
| `--json` | | Machine-readable JSON output |

### `cardiomas push`

Push previously saved local splits to HuggingFace. Requires `HF_TOKEN`.

```bash
cardiomas push ptb-xl
cardiomas push ptb-xl --output-dir /my/results
```

Runs a security audit (PII check, raw-data check, patient leakage check) before uploading. Refuses to push if any check fails.

### `cardiomas status`

Check if a dataset has published splits on HuggingFace.

```bash
cardiomas status ptb-xl
```

### `cardiomas list`

```bash
cardiomas list              # show known datasets (registry)
cardiomas list --remote     # show datasets published on HuggingFace
cardiomas list --local      # show locally cached datasets
```

### `cardiomas verify`

Re-check reproducibility metadata of published splits.

```bash
cardiomas verify ptb-xl --seed 42
```

### `cardiomas contribute`

Submit community splits to `vlbthambawita/ECGBench`.

```bash
cardiomas contribute ptb-xl --split-file my_splits.json
```

### `cardiomas config`

```bash
cardiomas config --show
cardiomas config --set OLLAMA_MODEL=mistral
```

## HuggingFace Publishing (opt-in)

Publishing requires write access to `vlbthambawita/ECGBench`. Set your token before pushing:

```bash
export HF_TOKEN=hf_...
cardiomas push ptb-xl
```

Only record identifiers are ever published — no raw ECG signals, no patient data.

## Python API

```python
from cardiomas import CardioMAS

mas = CardioMAS(ollama_model="llama3.1:8b", seed=42)

# Analyze and save locally
result = mas.analyze("/data/ptb-xl/")
print(result["local_output_dir"])   # output/ptb-xl

# Analyze and push to HuggingFace
mas.analyze("/data/ptb-xl/", push_to_hf=True)

# Read back published splits
splits = mas.get_splits("ptb-xl")
train_ids = splits["train"]

# Custom splits
mas.analyze(
    "/data/ptb-xl/",
    custom_split={"train": 0.7, "val": 0.15, "test": 0.15},
    stratify_by="scp_codes",
    seed=123,
)
```

## Environment Variables

Copy `.env.example` to `.env` and fill in as needed.

| Variable | Required for | Default |
|---|---|---|
| `OLLAMA_MODEL` | local LLM | `llama3.1:8b` |
| `OLLAMA_BASE_URL` | local LLM | `http://localhost:11434` |
| `HF_TOKEN` | `--push` / `cardiomas push` | — |
| `GITHUB_TOKEN` | GitHub README auto-update | — |
| `CARDIOMAS_SEED` | reproducibility | `42` |
| `CLOUD_LLM_PROVIDER` | `--use-cloud-llm` | `none` |

## Links

- **HuggingFace Dataset**: [vlbthambawita/ECGBench](https://huggingface.co/datasets/vlbthambawita/ECGBench)
- **PyPI**: [cardiomas](https://pypi.org/project/cardiomas/)
- **GitHub**: [vlbthambawita/CardioMAS](https://github.com/vlbthambawita/CardioMAS)

<!-- DATASETS TABLE -->
