# Changelog

All notable changes to CardioMAS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Complete rewrite as a full multi-agent ECG dataset split system
- `cardiomas` CLI with analyze/status/list/config/contribute/verify/version commands
- LangGraph-based 7-agent pipeline
- Reproducible, deterministic train/val/test split generation
- HuggingFace `vlbthambawita/ECGBench` publishing
- Security audit agent to prevent raw data or PII leakage
- Dataset registry for known ECG datasets (PTB-XL, MIMIC-IV-ECG, etc.)
- Python library API (`CardioMAS` class)
- pyproject.toml packaging with setuptools-scm auto-versioning
- GitHub Actions CI and auto-publish to PyPI on tag push
