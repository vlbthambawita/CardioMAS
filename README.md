# CardioMAS — Cardio Multi-Agent System

[![CI](https://github.com/vlbthambawita/CardioMAS/actions/workflows/ci.yml/badge.svg)](https://github.com/vlbthambawita/CardioMAS/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cardiomas)](https://pypi.org/project/cardiomas/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HF-vlbthambawita%2FECGBench-yellow)](https://huggingface.co/datasets/vlbthambawita/ECGBench)

A locally-runnable multi-agent system that analyzes ECG datasets, generates reproducible train/validation/test splits, and publishes split manifests (record IDs only) to [vlbthambawita/ECGBench](https://huggingface.co/datasets/vlbthambawita/ECGBench) on HuggingFace.

## Quick Start

```bash
pip install cardiomas

# Analyze a dataset and publish splits
cardiomas analyze https://physionet.org/content/ptb-xl/1.0.3/

# Check existing splits
cardiomas status ptb-xl

# List all analyzed datasets
cardiomas list --remote
```

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.com/) running locally with a model pulled (default: `llama3.1:8b`)

```bash
ollama pull llama3.1:8b
ollama serve
```

## Python API

```python
from cardiomas import CardioMAS

mas = CardioMAS(ollama_model="llama3.1:8b", seed=42)
result = mas.analyze("https://physionet.org/content/ptb-xl/1.0.3/")

splits = mas.get_splits("ptb-xl")
train_ids = splits["train"]
```

## Links

- **HuggingFace Dataset**: [vlbthambawita/ECGBench](https://huggingface.co/datasets/vlbthambawita/ECGBench)
- **PyPI**: [cardiomas](https://pypi.org/project/cardiomas/)
- **GitHub**: [vlbthambawita/CardioMAS](https://github.com/vlbthambawita/CardioMAS)
