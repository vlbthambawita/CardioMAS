# CardioMAS — Cardio Multi-Agent System

[![CI](https://github.com/vlbthambawita/CardioMAS/actions/workflows/ci.yml/badge.svg)](https://github.com/vlbthambawita/CardioMAS/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cardiomas)](https://pypi.org/project/cardiomas/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HF-vlbthambawita%2FECGBench-yellow)](https://huggingface.co/datasets/vlbthambawita/ECGBench)

A locally-runnable multi-agent system that analyzes ECG datasets and generates reproducible train/validation/test splits. Outputs are saved locally by default. Publishing to [vlbthambawita/ECGBench](https://huggingface.co/datasets/vlbthambawita/ECGBench) on HuggingFace is an explicit opt-in step that requires write access.

## Architecture

```mermaid
flowchart TD
    CLI(["`**CLI / Python API**
    cardiomas analyze`"])
    CLI --> ORCH

    subgraph PIPELINE ["LangGraph Pipeline — V2 Hub-and-Spoke"]
        ORCH(["🎯 **Orchestrator**
        Dynamic routing hub
        sets next_agent each turn"])

        %% Worker agents (spokes)
        NL["📝 NL Requirement
        Parse natural-language
        user requirements"]

        DISC["🔍 Discovery
        Identify dataset type,
        source & metadata"]

        PAPER["📄 Paper
        Find & parse paper
        Extract split methodology"]

        ANALYSIS["📊 Analysis
        Scan files, parse CSV
        Compute statistics"]

        SPLIT["✂️ Splitter
        SHA-256 seeded
        deterministic splits"]

        SEC["🔒 Security
        PII scan · raw-data
        Patient leakage check"]

        CODER["💻 Coder
        Generate custom
        split code"]

        PUB["☁️ Publisher
        Push to HF
        Update GitHub README"]

        %% Orchestrator → workers (dispatch)
        ORCH -->|dispatch| NL
        ORCH -->|dispatch| DISC
        ORCH -->|dispatch| PAPER
        ORCH -->|dispatch| ANALYSIS
        ORCH -->|dispatch| SPLIT
        ORCH -->|dispatch| SEC
        ORCH -->|dispatch| CODER
        ORCH -->|dispatch| PUB

        %% Workers → Orchestrator (return)
        NL -->|done| ORCH
        DISC -->|done| ORCH
        PAPER -->|done| ORCH
        ANALYSIS -->|done| ORCH
        SPLIT -->|done| ORCH
        SEC -->|done| ORCH
        CODER -->|done| ORCH
        PUB -->|done| ORCH

        %% Terminal routes from orchestrator
        ORCH -->|cache hit| EXISTING["↩ Return existing\nsplits from HF"]
        ORCH -->|audit failed| ERR["❌ End with error\nBlocked — not saved"]
        ORCH -->|pipeline done| SAVED["💾 End — saved locally"]
    end

    PUB -.->|if --push| HF[("HuggingFace
    vlbthambawita/ECGBench")]
    SAVED --> OUT

    OUT["`**output/&lt;dataset&gt;/**
    splits.json
    split_metadata.json
    analysis_report.md`"]

    style PIPELINE fill:#1a1a2e,stroke:#4a9eff,color:#fff
    style ORCH fill:#0f3460,stroke:#4a9eff,color:#fff,shape:stadium
    style NL fill:#2d1b4e,stroke:#9b59b6,color:#fff
    style DISC fill:#1b4332,stroke:#40916c,color:#fff
    style PAPER fill:#3b1f5e,stroke:#9b59b6,color:#fff
    style ANALYSIS fill:#7b4220,stroke:#e67e22,color:#fff
    style SPLIT fill:#1a5276,stroke:#2e86c1,color:#fff
    style SEC fill:#641e16,stroke:#e74c3c,color:#fff
    style CODER fill:#1a3a4a,stroke:#00bcd4,color:#fff
    style PUB fill:#145a32,stroke:#27ae60,color:#fff
    style SAVED fill:#1a3a1a,stroke:#27ae60,color:#fff
    style ERR fill:#3b0f0f,stroke:#e74c3c,color:#fff
    style HF fill:#2d2d2d,stroke:#f5a623,color:#fff
    style OUT fill:#2d2d2d,stroke:#4a9eff,color:#fff
    style EXISTING fill:#2d2d2d,stroke:#4a9eff,color:#fff
    style CLI fill:#0d0d0d,stroke:#4a9eff,color:#fff
```

Each node is a dedicated LLM-backed agent. The **orchestrator** is the central hub — it dynamically decides which agent to invoke next after each agent completes (hub-and-spoke pattern). Every worker agent returns to the orchestrator after finishing. Agents communicate only through the shared `GraphState`. The security agent is a hard gate — publishing is blocked if any check fails.

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.com/) running locally with a model pulled (default: `gemma4:e2b`)

```bash
ollama pull gemma4:e2b
ollama serve
pip install cardiomas
```

## Quick Start

```bash
# Analyze a dataset — results saved to ./output/ptb-xl/
cardiomas analyze https://physionet.org/content/ptb-xl/1.0.3/

# Use a local directory
cardiomas analyze /data/ptb-xl/

# Stream all agent reasoning live
cardiomas analyze /data/ptb-xl/ --verbose

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
| `--verbose` / `-v` | | Stream agent reasoning and LLM calls live |
| `--json` | | Machine-readable JSON output |

#### Verbose output

`--verbose` (`-v`) prints every agent step and LLM prompt/response in real time instead of showing a spinner:

```bash
cardiomas analyze /data/ptb-xl/ -v
```

```
Verbose mode on — streaming agent output below.

  [orchestrator] pipeline start — source: /data/ptb-xl/
  [orchestrator] no cache hit — running full pipeline
  [discovery]    registry hit → ptb-xl
  [paper]        searching arXiv: 'ptb-xl ECG dataset electrocardiogram'
  [paper]        found 3 result(s)
  [paper]        calling LLM (ChatOllama)…
──────────────── paper — LLM call ────────────────
╭─ prompt ────────────────────────────────────────╮
│ Analyze this ECG dataset paper and extract: …  │
╰─────────────────────────────────────────────────╯
╭─ response ──────────────────────────────────────╮
│ 1. Official splits: Yes (Section 2.3, page 4)  │
╰─────────────────────────────────────────────────╯
  [analysis]     found 42 files
  [splitter]     saved splits → output/ptb-xl/splits.json
  [security]     audit PASSED — no PII, no raw data, no leakage
```

Each agent is color-coded. Without `--verbose`, only a spinner runs during the pipeline and a summary table is shown at the end.

### `cardiomas push`

Push previously saved local splits to HuggingFace. Requires `HF_TOKEN`.

```bash
cardiomas push ptb-xl
cardiomas push ptb-xl --output-dir /my/results
```

Runs a security audit before uploading. Refuses to push if any check fails.

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

mas = CardioMAS(ollama_model="gemma4:e2b", seed=42)

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

## Using Different Local Models

Any model available in Ollama works. Pull a model, then point CardioMAS at it:

```bash
# Default (recommended for full pipeline)
ollama pull gemma4:e2b

# Larger Gemma 4 variants for heavier reasoning tasks
ollama pull gemma4:e4b
ollama pull gemma4:27b

# DeepSeek Coder — best for the coding agent
ollama pull deepseek-coder:6.7b

# Use a specific model for the whole pipeline
OLLAMA_MODEL=gemma4:e2b cardiomas analyze /data/ptb-xl/

# Or set it permanently in .env
echo "OLLAMA_MODEL=gemma4:e2b" >> .env
```

## Per-Agent LLM Configuration

> **Available in v0.2.0 (dev/v2-dynamic-orchestrator)**

Each agent can use a different LLM. This is useful when you want a fast, lightweight model for simple tasks (discovery, security scan) and a more capable model for reasoning-heavy tasks (analysis, coding).

### Via environment variables

```bash
# Fallback for all agents
OLLAMA_MODEL=gemma4:e2b

# Per-agent overrides (all optional)
AGENT_LLM_ORCHESTRATOR=gemma4:e2b
AGENT_LLM_NL_REQUIREMENT=gemma4:e2b
AGENT_LLM_DISCOVERY=gemma4:e2b
AGENT_LLM_PAPER=gemma4:e2b
AGENT_LLM_ANALYSIS=gemma4:e2b
AGENT_LLM_SPLITTER=gemma4:e2b
AGENT_LLM_SECURITY=gemma4:e2b
AGENT_LLM_CODER=deepseek-coder:6.7b
AGENT_LLM_PUBLISHER=gemma4:e2b
```

Set these in `.env` or export them before running `cardiomas`.

### Via CLI flags

```bash
cardiomas analyze /data/ptb-xl/ \
  --llm-coder deepseek-coder:6.7b \
  --llm-analysis gemma4:e4b \
  --llm-discovery gemma4:e2b
```

### Via Python API

```python
from cardiomas import CardioMAS

mas = CardioMAS(
    agent_llms={
        "coder":    "deepseek-coder:6.7b",
        "analysis": "gemma4:e4b",
        "default":  "gemma4:e2b",   # fallback for all other agents
    }
)
mas.analyze("/data/ptb-xl/")
```

### Model recommendations

| Agent | Recommended model | Why |
|---|---|---|
| `orchestrator` | `gemma4:e2b` | Default — dynamic routing decisions |
| `nl_requirement` | `gemma4:e2b` | Simple parsing task |
| `discovery` | `gemma4:e2b` | Lookup + classification |
| `paper` | `gemma4:e2b` or `gemma4:e4b` | Needs to read and summarise papers |
| `analysis` | `gemma4:e2b` or `gemma4:e4b` | Statistical reasoning |
| `splitter` | `gemma4:e2b` | Deterministic — LLM role is minimal |
| `security` | `gemma4:e2b` | Pattern matching |
| `coder` | `deepseek-coder:6.7b` | Code generation |
| `publisher` | `gemma4:e2b` | Structured output |

### Verbose LLM name display

With `--verbose`, each LLM call shows the model name and backend:

```
──────────── paper — LLM call [gemma4:e2b @ ollama] ────────────
```

## Environment Variables

Copy `.env.example` to `.env` and fill in as needed.

| Variable | Required for | Default |
|---|---|---|
| `OLLAMA_MODEL` | local LLM (default for all agents) | `gemma4:e2b` |
| `OLLAMA_BASE_URL` | local LLM | `http://localhost:11434` |
| `AGENT_LLM_<AGENT>` | per-agent model override | *(falls back to `OLLAMA_MODEL`)* |
| `HF_TOKEN` | `--push` / `cardiomas push` | — |
| `GITHUB_TOKEN` | GitHub README auto-update | — |
| `CARDIOMAS_SEED` | reproducibility | `42` |
| `CLOUD_LLM_PROVIDER` | `--use-cloud-llm` | `none` |

## Links

- **HuggingFace Dataset**: [vlbthambawita/ECGBench](https://huggingface.co/datasets/vlbthambawita/ECGBench)
- **PyPI**: [cardiomas](https://pypi.org/project/cardiomas/)
- **GitHub**: [vlbthambawita/CardioMAS](https://github.com/vlbthambawita/CardioMAS)

<!-- DATASETS TABLE -->
