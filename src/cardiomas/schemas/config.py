from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, model_validator


class KnowledgeSource(BaseModel):
    kind: Literal["local_dir", "local_file", "dataset_dir", "web_page", "pdf"] = Field(
        validation_alias=AliasChoices("kind", "type")
    )
    id: str = ""
    label: str = ""
    path: str | None = None
    url: str | None = None
    recursive: bool = True
    include_extensions: list[str] = Field(default_factory=list)
    trust: Literal["high", "medium", "low"] = "medium"
    visibility: Literal["private", "internal", "public"] = "private"
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_locator(self) -> "KnowledgeSource":
        if not (self.path or self.url):
            raise ValueError("KnowledgeSource requires either 'path' or 'url'.")
        if not self.id:
            locator = self.path or self.url or self.kind
            self.id = _slug(locator)
        if not self.label:
            self.label = Path(self.path).name if self.path else (self.url or self.id)
        return self


class RetrievalConfig(BaseModel):
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = 5
    chunk_size: int = 700
    chunk_overlap: int = 120
    min_score: float = 0.0
    rrf_k: int = 60


class ToolPolicyConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: [
            "retrieve_corpus",
            "list_folder_structure",
            "read_wfdb_dataset",
            "read_dataset_website",
            "inspect_dataset",
            "calculate",
            "fetch_webpage",
        ]
    )


class MemoryPolicy(BaseModel):
    mode: Literal["none", "session"] = "session"
    max_turns: int = 8


class SafetyConfig(BaseModel):
    allow_web_fetch: bool = False
    allow_action_tools: bool = False
    require_approval_for_actions: bool = True


class ResponseConfig(BaseModel):
    include_citations: bool = True
    max_citations: int = 5


class AgentConfig(BaseModel):
    mode: Literal["react", "linear"] = "linear"
    max_iterations: int = 5
    query_decomposition: bool = False
    self_reflection: bool = False
    retrieval_grading: bool = True
    memory_mode: Literal["session", "persistent", "none"] = "session"
    persistent_memory_max: int = 200
    # ReAct++ extensions
    upfront_planning: bool = False   # one LLM call before loop to produce ordered tool plan
    step_reflection: bool = False    # agent reflects on progress after each observation
    scratchpad: bool = True          # maintain distilled key-facts across iterations
    tool_verification: bool = True   # validate tool args before execution


class AutonomyConfig(BaseModel):
    enable_code_agents: bool = False
    allow_tool_codegen: bool = False
    allow_script_codegen: bool = False
    dataset_mode: Literal["script_only", "agentic"] = "agentic"
    execute_for_answer: bool = False
    execution_timeout_seconds: int = 60
    max_repair_attempts: int = 2
    workspace_dir: str = ""
    require_approval_for_repo_writes: bool = True
    require_approval_for_shell_execution: bool = True
    require_approval_for_installs: bool = True
    allowed_shell_prefixes: list[str] = Field(default_factory=lambda: ["python", "bash", "sh", "cardiomas"])
    allowed_python_modules: list[str] = Field(
        default_factory=lambda: [
            "csv",
            "json",
            "math",
            "statistics",
            "pathlib",
            "collections",
            "datetime",
            "numpy",
            "pandas",
            "cardiomas",
        ]
    )

    @property
    def enabled(self) -> bool:
        return self.enable_code_agents


class LLMConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    planner_mode: Literal["heuristic", "ollama"] = "heuristic"
    model: str = ""
    planner_model: str = ""
    responder_model: str = ""
    code_model: str = ""
    router_model: str = ""
    temperature: float = 0.1
    max_tokens: int = 800
    code_max_tokens: int = 4000
    code_temperature: float = 0.2
    router_max_tokens: int = 300
    timeout_seconds: float = 60.0
    keep_alive: str = "5m"

    @property
    def resolved_planner_model(self) -> str:
        return self.planner_model or self.model

    @property
    def resolved_responder_model(self) -> str:
        return self.responder_model or self.model

    @property
    def resolved_code_model(self) -> str:
        return self.code_model or self.model

    @property
    def resolved_router_model(self) -> str:
        return self.router_model or self.model

    @property
    def planner_enabled(self) -> bool:
        return self.planner_mode == "ollama" and bool(self.resolved_planner_model)

    @property
    def responder_enabled(self) -> bool:
        return bool(self.resolved_responder_model)


class EmbeddingConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str
    batch_size: int = 32
    timeout_seconds: float = 60.0
    keep_alive: str = "5m"


class RuntimeConfig(BaseModel):
    system_name: str = "CardioMAS"
    output_dir: str = "runtime_output"
    sources: list[KnowledgeSource] = Field(default_factory=list)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tools: ToolPolicyConfig = Field(default_factory=ToolPolicyConfig)
    memory: MemoryPolicy = Field(default_factory=MemoryPolicy)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    autonomy: AutonomyConfig = Field(default_factory=AutonomyConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig | None = None
    embeddings: EmbeddingConfig | None = None
    scripts_dir: str = ""

    @property
    def corpus_path(self) -> Path:
        return Path(self.output_dir) / "corpus.jsonl"

    @property
    def manifest_path(self) -> Path:
        return Path(self.output_dir) / "corpus_manifest.json"

    @property
    def autonomy_workspace_path(self) -> Path:
        if self.autonomy.workspace_dir:
            return Path(self.autonomy.workspace_dir)
        return Path(self.output_dir) / "autonomy_workspace"

    @property
    def resolved_scripts_dir(self) -> Path:
        if self.scripts_dir:
            return Path(self.scripts_dir)
        return Path(self.output_dir) / "scripts"

    @property
    def planner_uses_ollama(self) -> bool:
        return bool(self.llm and self.llm.planner_enabled)

    @property
    def responder_uses_ollama(self) -> bool:
        return bool(self.llm and self.llm.responder_enabled)

    @property
    def embeddings_enabled(self) -> bool:
        return self.embeddings is not None

    @classmethod
    def from_file(cls, path: str) -> "RuntimeConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}."
                ) from exc
        else:
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                mark = getattr(exc, "problem_mark", None)
                location = ""
                if mark is not None:
                    location = f" at line {mark.line + 1}, column {mark.column + 1}"
                raise ValueError(
                    f"Invalid YAML in {path}{location}. Check indentation and ':' placement."
                ) from exc

        if not isinstance(data, dict):
            raise ValueError("Runtime config must contain a mapping/object at the top level.")

        resolved = _resolve_relative_paths(data, config_path.parent)
        return cls.model_validate(resolved)


def _resolve_relative_paths(data: dict, base_dir: Path) -> dict:
    resolved = dict(data)
    output_dir = resolved.get("output_dir")
    if isinstance(output_dir, str) and not Path(output_dir).is_absolute():
        resolved["output_dir"] = str((base_dir / output_dir).resolve())

    sources = []
    for item in resolved.get("sources", []):
        if not isinstance(item, dict):
            sources.append(item)
            continue
        source = dict(item)
        path_value = source.get("path")
        if isinstance(path_value, str) and not Path(path_value).is_absolute():
            source["path"] = str((base_dir / path_value).resolve())
        sources.append(source)
    resolved["sources"] = sources
    autonomy = resolved.get("autonomy")
    if isinstance(autonomy, dict):
        autonomy_resolved = dict(autonomy)
        workspace_dir = autonomy_resolved.get("workspace_dir")
        if isinstance(workspace_dir, str) and workspace_dir and not Path(workspace_dir).is_absolute():
            autonomy_resolved["workspace_dir"] = str((base_dir / workspace_dir).resolve())
        resolved["autonomy"] = autonomy_resolved
    scripts_dir = resolved.get("scripts_dir")
    if isinstance(scripts_dir, str) and scripts_dir and not Path(scripts_dir).is_absolute():
        resolved["scripts_dir"] = str((base_dir / scripts_dir).resolve())
    return resolved


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-") or "source"
