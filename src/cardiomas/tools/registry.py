from __future__ import annotations

from collections.abc import Callable

from cardiomas.autonomy.recovery import AutonomousToolManager
from cardiomas.inference.base import EmbeddingClient
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.tools import ToolResult, ToolSpec
from cardiomas.tools.dataset_tools import inspect_dataset
from cardiomas.tools.research_tools import fetch_webpage
from cardiomas.tools.retrieval_tools import retrieve_corpus
from cardiomas.tools.utility_tools import calculate_expression


ToolHandler = Callable[..., ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    def execute(self, name: str, **kwargs) -> ToolResult:
        if name not in self._handlers:
            raise KeyError(f"Unknown tool: {name}")
        return self._handlers[name](**kwargs)

    def has(self, name: str) -> bool:
        return name in self._handlers

    def agent_tool_list(self) -> str:
        """Human-readable tool list for LLM prompts."""
        lines = []
        for spec in self.specs():
            lines.append(f"- {spec.name}: {spec.description}")
        return "\n".join(lines) or "(no tools available)"


def build_registry(
    config: RuntimeConfig,
    chunks,
    embedding_client: EmbeddingClient | None = None,
    autonomy_manager: AutonomousToolManager | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    if "retrieve_corpus" in config.tools.enabled:
        registry.register(
            ToolSpec(
                name="retrieve_corpus",
                description="Retrieve ranked evidence chunks from the built local corpus.",
                category="retrieval",
            ),
            lambda query, top_k=None: retrieve_corpus(
                chunks=chunks,
                query=query,
                config=config,
                embedding_client=embedding_client,
                top_k=top_k,
            ),
        )

    if "inspect_dataset" in config.tools.enabled:
        registry.register(
            ToolSpec(
                name="inspect_dataset",
                description="Inspect a local dataset directory and summarize files and CSV headers.",
                category="dataset",
            ),
            inspect_dataset,
        )

    if "calculate" in config.tools.enabled:
        registry.register(
            ToolSpec(
                name="calculate",
                description="Safely evaluate a simple arithmetic expression.",
                category="utility",
            ),
            calculate_expression,
        )

    if "fetch_webpage" in config.tools.enabled:
        registry.register(
            ToolSpec(
                name="fetch_webpage",
                description="Fetch a webpage directly by URL when web access is allowed by config.",
                category="research",
            ),
            fetch_webpage,
        )

    if autonomy_manager is not None and autonomy_manager.enabled:
        for spec in autonomy_manager.tool_specs():
            if registry.has(spec.name):
                continue
            if spec.name == "generate_python_artifact":
                registry.register(spec, autonomy_manager.generate_python_artifact)
            elif spec.name == "generate_shell_artifact":
                registry.register(spec, autonomy_manager.generate_shell_artifact)

    return registry
