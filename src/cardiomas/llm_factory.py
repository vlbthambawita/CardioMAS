from __future__ import annotations

import logging

import requests
from langchain_core.language_models import BaseChatModel

import cardiomas.config as cfg

logger = logging.getLogger(__name__)


def _check_ollama(model: str | None = None) -> bool:
    try:
        r = requests.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        check_model = model or cfg.OLLAMA_MODEL
        models = [m["name"] for m in r.json().get("models", [])]
        model_base = check_model.split(":")[0]
        return any(m.startswith(model_base) for m in models)
    except Exception:
        return False


def get_local_llm(temperature: float = 0.1, model: str | None = None) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    model_name = model or cfg.OLLAMA_MODEL
    if not _check_ollama(model_name):
        raise RuntimeError(
            f"Ollama is not running or model '{model_name}' is not available. "
            f"Run: ollama pull {model_name} && ollama serve"
        )
    return ChatOllama(
        model=model_name,
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


def get_llm_for_agent(
    agent_name: str,
    prefer_cloud: bool = False,
    temperature: float = 0.1,
    agent_llm_map: dict[str, str] | None = None,
) -> BaseChatModel:
    """Return the LLM configured for a specific agent.

    Priority:
    1. agent_llm_map (runtime override from UserOptions / Python API)
    2. AGENT_LLM_<AGENT> env var
    3. OLLAMA_MODEL (global default)
    4. Cloud LLM if prefer_cloud and CLOUD_LLM_PROVIDER != none
    """
    if prefer_cloud and cfg.CLOUD_LLM_PROVIDER != "none":
        try:
            return get_cloud_llm(temperature)
        except Exception as e:
            logger.warning(f"Cloud LLM failed ({e}), falling back to local Ollama.")

    # Resolve model: runtime map wins over env var
    model_name: str
    if agent_llm_map and agent_name in agent_llm_map:
        model_name = agent_llm_map[agent_name]
    elif agent_llm_map and "default" in agent_llm_map:
        model_name = agent_llm_map["default"]
    else:
        model_name = cfg.get_agent_llm(agent_name)

    return get_local_llm(temperature, model=model_name)
