from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from cardiomas.agentic.runtime import AgenticRuntime
from cardiomas.schemas.config import RuntimeConfig


class CardioMAS:
    """Fresh Agentic RAG API for CardioMAS."""

    def __init__(self, config_path: str | None = None, config: RuntimeConfig | None = None) -> None:
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = RuntimeConfig.from_file(config_path)
        else:
            self.config = RuntimeConfig()
        self.runtime = AgenticRuntime(self.config)

    def build_corpus(self, force_rebuild: bool = False, agent_name: str | None = None) -> dict[str, Any]:
        return self.runtime.build_corpus(force_rebuild=force_rebuild, agent_name=agent_name).model_dump(mode="json")

    def query(self, query: str, force_rebuild: bool = False, agent_name: str | None = None) -> dict[str, Any]:
        return self.runtime.query(query, force_rebuild=force_rebuild, agent_name=agent_name).model_dump(mode="json")

    def query_stream(self, query: str, force_rebuild: bool = False, agent_name: str | None = None) -> Iterator[dict[str, Any]]:
        for event in self.runtime.query_stream(query, force_rebuild=force_rebuild, agent_name=agent_name):
            yield event.model_dump(mode="json")

    def list_agents(self) -> list[dict[str, Any]]:
        """Return a list of configured named agents with corpus-built status."""
        result = []
        for agent_cfg in self.config.named_agents:
            corpus_path = self.config.agent_corpus_path(agent_cfg.name)
            result.append({
                "name": agent_cfg.name,
                "description": agent_cfg.description,
                "knowledge_enabled": agent_cfg.knowledge.enabled,
                "source_count": len(agent_cfg.knowledge.sources),
                "corpus_built": corpus_path.exists(),
                "corpus_path": str(corpus_path),
            })
        return result

    def inspect_tools(self) -> list[dict[str, Any]]:
        return [spec.model_dump(mode="json") for spec in self.runtime.inspect_tools()]

    def check_ollama(self) -> dict[str, Any]:
        return self.runtime.check_ollama()
