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


def _path_kwarg(**kwargs: object) -> str:
    """Normalise LLM arg spelling: 'path', 'dataset_path', 'directory' → str."""
    return str(
        kwargs.get("path")
        or kwargs.get("dataset_path")
        or kwargs.get("directory")
        or ""
    )


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
                    "a given dataset directory, including file sizes and CSV/TSV column headers. "
                    "Required argument: 'path' (string) — absolute path to the directory. "
                    "Optional: 'max_depth' (int, default 4). "
                    "Use this first when you need to understand which files are available "
                    "and how the data is organised before deciding what to read or compute."
                ),
                category="dataset",
            ),
            lambda **kw: list_folder_structure(
                path=_path_kwarg(**kw),
                max_depth=int(kw.get("max_depth", 4)),
            ),
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
                    "Required argument: 'path' (string) — absolute path to the WFDB dataset root. "
                    "Optional: 'max_records' (int, default 20). "
                    "Use this when the dataset contains .hea files and you need to know "
                    "which ECG leads are recorded, at what sampling rate, how long each "
                    "recording is, or what annotations are available."
                ),
                category="dataset",
            ),
            lambda **kw: read_wfdb_dataset(
                path=_path_kwarg(**kw),
                max_records=int(kw.get("max_records", 20)),
            ),
        )

    if "inspect_dataset" in config.tools.enabled:
        registry.register(
            ToolSpec(
                name="inspect_dataset",
                description=(
                    "Inspect a local dataset directory and summarize files and CSV headers. "
                    "Required argument: 'path' (string) — absolute path to the directory."
                ),
                category="dataset",
            ),
            lambda **kw: inspect_dataset(path=_path_kwarg(**kw)),
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
        # Build a label→URL lookup so the agent can call the tool with a label
        label_to_url: dict[str, str] = {label: url for label, url in web_sources}

        if web_sources:
            url_list = "; ".join(f"'{label}' ({url})" for label, url in web_sources)
            configured_note = (
                f" Configured dataset websites (pass the full URL or the label as 'url'): "
                f"{url_list}."
            )
        else:
            configured_note = (
                " No dataset website URLs configured yet — add them under 'sources' "
                "with 'kind: web_page' in the YAML config."
            )

        def _resolve_and_fetch(**kw: object) -> ToolResult:
            raw = str(kw.get("url", "")).strip()
            # If the agent passed a label instead of a URL, resolve it
            resolved = label_to_url.get(raw, raw)
            return read_dataset_website(url=resolved, config=config)

        registry.register(
            ToolSpec(
                name="read_dataset_website",
                description=(
                    "Fetch a dataset documentation website (PhysioNet, HuggingFace, "
                    "Zenodo, Kaggle, or any research data portal) and extract structured "
                    "metadata: description, file listing, signal/variable names, sampling "
                    "frequency, record count, annotation types, license, DOI, and download "
                    "links. Required argument: 'url' (string) — full URL starting with "
                    "http:// or https://, or a configured label (see below). "
                    "Use this before writing analysis code to understand what the dataset "
                    "contains, how it is organised, or what the columns/signals represent."
                    + configured_note
                ),
                category="research",
            ),
            _resolve_and_fetch,
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
                # Normalise LLM argument spelling: 'code'/'description' → 'task',
                # 'dataset_path'/'directory' → 'dataset_path'
                def _gen_python(**kw: object) -> ToolResult:
                    return autonomy_manager.generate_python_artifact(
                        task=str(
                            kw.get("task")
                            or kw.get("code")
                            or kw.get("description")
                            or kw.get("prompt")
                            or ""
                        ),
                        dataset_path=str(
                            kw.get("dataset_path")
                            or kw.get("path")
                            or kw.get("directory")
                            or ""
                        ),
                        target_path=str(kw.get("target_path", "")),
                        artifact_name=str(kw.get("artifact_name", "")),
                    )
                registry.register(spec, _gen_python)
            elif spec.name == "generate_shell_artifact":
                def _gen_shell(**kw: object) -> ToolResult:
                    return autonomy_manager.generate_shell_artifact(
                        task=str(
                            kw.get("task")
                            or kw.get("code")
                            or kw.get("description")
                            or kw.get("prompt")
                            or ""
                        ),
                        dataset_path=str(
                            kw.get("dataset_path")
                            or kw.get("path")
                            or kw.get("directory")
                            or ""
                        ),
                    )
                registry.register(spec, _gen_shell)

    return registry
