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


class ToolPolicyConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: [
            "retrieve_corpus",
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


class RuntimeConfig(BaseModel):
    system_name: str = "CardioMAS"
    output_dir: str = "runtime_output"
    sources: list[KnowledgeSource] = Field(default_factory=list)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tools: ToolPolicyConfig = Field(default_factory=ToolPolicyConfig)
    memory: MemoryPolicy = Field(default_factory=MemoryPolicy)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)

    @property
    def corpus_path(self) -> Path:
        return Path(self.output_dir) / "corpus.jsonl"

    @property
    def manifest_path(self) -> Path:
        return Path(self.output_dir) / "corpus_manifest.json"

    @classmethod
    def from_file(cls, path: str) -> "RuntimeConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)

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
    return resolved


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-") or "source"
