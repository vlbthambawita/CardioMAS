from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import AliasChoices, BaseModel, Field, model_validator


DEFAULT_ORGANIZATION_GOAL = "Build reusable dataset knowledge and analysis artifacts"
DEFAULT_ORGANIZATION_OUTPUT_DIR = "organization_output"


class OrganizationConfig(BaseModel):
    dataset_name: str | None = None
    dataset_dir: str | None = None
    local_data_path: str | None = Field(default=None, validation_alias=AliasChoices("local_data_path", "data_path"))
    knowledge_urls: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("knowledge_urls", "knowledge_links"),
    )
    goal: str = DEFAULT_ORGANIZATION_GOAL
    output_dir: str = DEFAULT_ORGANIZATION_OUTPUT_DIR
    approve: bool = False

    @model_validator(mode="after")
    def _ensure_dataset_path(self) -> "OrganizationConfig":
        if not (self.local_data_path or self.dataset_dir):
            raise ValueError("Config must define either 'local_data_path' or 'dataset_dir'.")
        return self

    @property
    def resolved_dataset_dir(self) -> str:
        return self.local_data_path or self.dataset_dir or ""

    @classmethod
    def from_file(cls, path: str) -> "OrganizationConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = config_path.read_text(encoding="utf-8")
        suffix = config_path.suffix.lower()

        if suffix == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)

        if not isinstance(data, dict):
            raise ValueError("Organization config must contain a mapping/object at the top level.")

        data = _resolve_config_relative_paths(data, config_path.parent)
        return cls.model_validate(data)


def resolve_organization_config(
    *,
    config_path: str | None = None,
    dataset_dir: str | None = None,
    dataset_name: str | None = None,
    knowledge_urls: list[str] | None = None,
    goal: str | None = None,
    output_dir: str | None = None,
    approve: bool | None = None,
) -> OrganizationConfig:
    base = OrganizationConfig.from_file(config_path) if config_path else OrganizationConfig(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        knowledge_urls=knowledge_urls or [],
        goal=goal or DEFAULT_ORGANIZATION_GOAL,
        output_dir=output_dir or DEFAULT_ORGANIZATION_OUTPUT_DIR,
        approve=bool(approve),
    )

    merged = base.model_copy(update={
        "dataset_dir": dataset_dir or base.dataset_dir,
        "local_data_path": dataset_dir or base.local_data_path,
        "dataset_name": dataset_name or base.dataset_name,
        "knowledge_urls": knowledge_urls if knowledge_urls is not None else base.knowledge_urls,
        "goal": goal or base.goal,
        "output_dir": output_dir or base.output_dir,
        "approve": approve if approve is not None else base.approve,
    })

    if not merged.dataset_name:
        merged = merged.model_copy(update={"dataset_name": Path(merged.resolved_dataset_dir).name})

    return merged


def _resolve_config_relative_paths(data: dict, base_dir: Path) -> dict:
    resolved = dict(data)
    for key in ("dataset_dir", "local_data_path", "data_path", "output_dir"):
        value = resolved.get(key)
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                resolved[key] = str((base_dir / path).resolve())
    return resolved
