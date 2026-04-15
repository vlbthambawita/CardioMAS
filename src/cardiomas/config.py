from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_path(key: str, default: str) -> Path:
    return Path(_env(key, default)).expanduser()


# ── Local LLM ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "llama3.1:8b")

# ── Cloud LLM (optional) ───────────────────────────────────────────────────
CLOUD_LLM_PROVIDER: str = _env("CLOUD_LLM_PROVIDER", "none")  # none | openai | anthropic
CLOUD_LLM_MODEL: str = _env("CLOUD_LLM_MODEL", "")

# ── HuggingFace ────────────────────────────────────────────────────────────
HF_TOKEN: str = _env("HF_TOKEN", "")
HF_REPO_ID: str = _env("HF_REPO_ID", "vlbthambawita/ECGBench")

# ── GitHub ─────────────────────────────────────────────────────────────────
GITHUB_TOKEN: str = _env("GITHUB_TOKEN", "")
GITHUB_REPO: str = _env("GITHUB_REPO", "vlbthambawita/CardioMAS")

# ── Storage ────────────────────────────────────────────────────────────────
DATA_DIR: Path = _env_path("CARDIOMAS_DATA_DIR", "~/.cardiomas/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────
SEED: int = int(_env("CARDIOMAS_SEED", "42"))
