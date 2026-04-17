from __future__ import annotations

from cardiomas.schemas.config import RuntimeConfig


def memory_enabled(config: RuntimeConfig) -> bool:
    return config.memory.mode != "none"
