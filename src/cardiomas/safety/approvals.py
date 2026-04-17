from __future__ import annotations

from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.tools import ToolSpec


def approval_required(config: RuntimeConfig, spec: ToolSpec) -> bool:
    return bool(spec.requires_approval and config.safety.require_approval_for_actions)
