from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cardiomas.schemas.dataset import DatasetInfo, DatasetSource

_REGISTRY_FILE = Path(__file__).parent / "registry.yaml"


class DatasetRegistry:
    def __init__(self) -> None:
        self._datasets: dict[str, DatasetInfo] = {}
        self._load_builtin()

    def _load_builtin(self) -> None:
        if _REGISTRY_FILE.exists():
            with open(_REGISTRY_FILE) as f:
                data: dict[str, Any] = yaml.safe_load(f) or {}
            for name, entry in data.get("datasets", {}).items():
                entry["name"] = name
                entry["source_type"] = DatasetSource(entry.get("source_type", "url"))
                self._datasets[name] = DatasetInfo(**entry)

    def get(self, name: str) -> DatasetInfo | None:
        return self._datasets.get(name)

    def register(self, info: DatasetInfo) -> None:
        self._datasets[info.name] = info

    def list_names(self) -> list[str]:
        return sorted(self._datasets.keys())

    def all(self) -> list[DatasetInfo]:
        return list(self._datasets.values())


_registry: DatasetRegistry | None = None


def get_registry() -> DatasetRegistry:
    global _registry
    if _registry is None:
        _registry = DatasetRegistry()
    return _registry
