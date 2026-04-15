from __future__ import annotations

import logging

import requests
from langchain_core.language_models import BaseChatModel

import cardiomas.config as cfg

logger = logging.getLogger(__name__)


def _check_ollama() -> bool:
    try:
        r = requests.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        model_base = cfg.OLLAMA_MODEL.split(":")[0]
        return any(m.startswith(model_base) for m in models)
    except Exception:
        return False


def get_local_llm(temperature: float = 0.1) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    if not _check_ollama():
        raise RuntimeError(
            f"Ollama is not running or model '{cfg.OLLAMA_MODEL}' is not available. "
            f"Run: ollama pull {cfg.OLLAMA_MODEL} && ollama serve"
        )
    return ChatOllama(
        model=cfg.OLLAMA_MODEL,
        base_url=cfg.OLLAMA_BASE_URL,
        temperature=temperature,
    )


def get_cloud_llm(temperature: float = 0.1) -> BaseChatModel:
    provider = cfg.CLOUD_LLM_PROVIDER.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=cfg.CLOUD_LLM_MODEL or "gpt-4o", temperature=temperature)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=cfg.CLOUD_LLM_MODEL or "claude-3-5-sonnet-20241022", temperature=temperature)  # type: ignore[call-arg]
    else:
        raise ValueError(f"Unsupported cloud LLM provider: {provider}")


def get_llm(prefer_cloud: bool = False, temperature: float = 0.1) -> BaseChatModel:
    if prefer_cloud and cfg.CLOUD_LLM_PROVIDER != "none":
        try:
            return get_cloud_llm(temperature)
        except Exception as e:
            logger.warning(f"Cloud LLM failed ({e}), falling back to local Ollama.")
    return get_local_llm(temperature)
