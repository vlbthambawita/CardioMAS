from __future__ import annotations

from collections.abc import Callable

from cardiomas.autonomy.recovery import AutonomousToolManager
from cardiomas.inference.base import EmbeddingClient
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.tools import ToolResult, ToolSpec
from cardiomas.tools.dataset_tools import inspect_dataset, list_folder_structure
from cardiomas.tools.research_tools import fetch_webpage
from cardiomas.tools.retrieval_tools import retrieve_corpus
from cardiomas.tools.utility_tools import calculate_expression
from cardiomas.tools.web_dataset_tools import _configured_web_urls, read_dataset_website
from cardiomas.tools.wfdb_tools import read_wfdb_dataset


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

    if "list_folder_structure" in config.tools.enabled:
        registry.register(
            ToolSpec(
                name="list_folder_structure",
                description=(
                    "Return a tree-formatted listing of every file and sub-directory inside "
                    "a given dataset path, including file sizes and CSV/TSV column headers. "
                    "Use this first when you need to understand which files are available "
                    "and how the data is organised before deciding what to read or compute."
                ),
                category="dataset",
            ),
            list_folder_structure,
        )

    if "read_wfdb_dataset" in config.tools.enabled:
        registry.register(
            ToolSpec(
                name="read_wfdb_dataset",
                description=(
                    "Inspect a PhysioNet WFDB-format ECG dataset directory and return "
                    "structured metadata without loading raw signal data. Scans for WFDB "
                    "header files (.hea), parses signal names (leads), sampling frequencies, "
                    "recording durations, ADC units, and detects annotation file types "
                    "(.atr, .ecg, .ann, etc.) and signal file formats (.dat, .edf, .mat). "
                    "Use this tool when the dataset contains .hea files and you need to know "
                    "which ECG leads are recorded, at what sampling rate, how long each "
                    "recording is, or what annotations are available before writing analysis "
                    "code. Works with or without the wfdb Python library installed."
                ),
                category="dataset",
            ),
            read_wfdb_dataset,
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

    if "read_dataset_website" in config.tools.enabled:
        web_sources = _configured_web_urls(config)
        if web_sources:
            url_hints = "; ".join(f"{label} → {url}" for label, url in web_sources)
            configured_note = (
                f" The following dataset websites are already configured in this "
                f"session and are the primary targets for this tool: {url_hints}."
            )
        else:
            configured_note = (
                " No dataset website URLs are configured yet — the user can add "
                "them under 'sources' with 'kind: web_page' in the YAML config."
            )
        registry.register(
            ToolSpec(
                name="read_dataset_website",
                description=(
                    "Fetch a dataset documentation website (PhysioNet, HuggingFace, "
                    "Zenodo, Kaggle, or any research data portal) and extract structured "
                    "metadata: description, file listing, signal/variable names, sampling "
                    "frequency, record count, annotation types, license, DOI, and download "
                    "links. Use this before writing analysis code when you need to understand "
                    "what data the dataset contains, how it is organised, or what the "
                    "columns/signals represent. Takes a single 'url' argument."
                    + configured_note
                ),
                category="research",
            ),
            lambda url: read_dataset_website(url=url, config=config),
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
