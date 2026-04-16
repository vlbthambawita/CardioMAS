from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ToolContract(BaseModel):
    name: str
    description: str
    cli_command: str
    outputs: list[str] = Field(default_factory=list)


class DatasetInventorySummary(BaseModel):
    dataset_name: str
    dataset_dir: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_files: int
    total_directories: int
    total_size_bytes: int
    extension_counts: dict[str, int] = Field(default_factory=dict)
    sample_files: list[str] = Field(default_factory=list)
    csv_schemas: dict[str, list[str]] = Field(default_factory=dict)


def default_tool_contracts() -> list[ToolContract]:
    return [
        ToolContract(
            name="dataset_inventory",
            description="Inspect the dataset directory and produce reusable inventory outputs.",
            cli_command="cardiomas organize /path/to/dataset --approve",
            outputs=["dataset_inventory.json", "file_extensions.csv", "dataset_inventory.md"],
        )
    ]
