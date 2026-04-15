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

# ── Per-agent LLM overrides ────────────────────────────────────────────────
# Format: AGENT_LLM_<AGENT>=model  e.g. AGENT_LLM_CODER=deepseek-coder:6.7b
# Falls back to OLLAMA_MODEL when not set.
_AGENT_LLM_OVERRIDES: dict[str, str] = {}
for _agent in (
    "orchestrator", "nl_requirement", "discovery", "paper",
    "analysis", "splitter", "security", "coder", "publisher",
):
    _val = _env(f"AGENT_LLM_{_agent.upper()}", "")
    if _val:
        _AGENT_LLM_OVERRIDES[_agent] = _val


def get_agent_llm(agent_name: str) -> str:
    """Return the Ollama model name for a given agent (falls back to OLLAMA_MODEL)."""
    return _AGENT_LLM_OVERRIDES.get(agent_name, OLLAMA_MODEL)


def set_agent_llm(agent_name: str, model: str) -> None:
    """Override the model for a specific agent at runtime."""
    _AGENT_LLM_OVERRIDES[agent_name] = model


# ── Context compression ────────────────────────────────────────────────────
CONTEXT_COMPRESS_THRESHOLD: int = int(_env("CONTEXT_COMPRESS_THRESHOLD", "6000"))
CONTEXT_COMPRESS_MODEL: str = _env("CONTEXT_COMPRESS_MODEL", "gemma3:4b")

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
