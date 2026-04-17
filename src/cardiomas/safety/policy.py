from __future__ import annotations

from cardiomas.schemas.config import RuntimeConfig


def web_allowed(config: RuntimeConfig) -> bool:
    return config.safety.allow_web_fetch


def action_tools_allowed(config: RuntimeConfig) -> bool:
    return config.safety.allow_action_tools
