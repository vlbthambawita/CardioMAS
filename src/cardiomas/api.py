from __future__ import annotations

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

    def build_corpus(self, force_rebuild: bool = False) -> dict[str, Any]:
        return self.runtime.build_corpus(force_rebuild=force_rebuild).model_dump(mode="json")

    def query(self, query: str, force_rebuild: bool = False) -> dict[str, Any]:
        return self.runtime.query(query, force_rebuild=force_rebuild).model_dump(mode="json")

    def inspect_tools(self) -> list[dict[str, Any]]:
        return [spec.model_dump(mode="json") for spec in self.runtime.inspect_tools()]
