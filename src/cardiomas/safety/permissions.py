from __future__ import annotations

from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.tools import ToolSpec
from cardiomas.safety.policy import action_tools_allowed, web_allowed


def tool_allowed(config: RuntimeConfig, spec: ToolSpec) -> tuple[bool, str]:
    if spec.name == "fetch_webpage" and not web_allowed(config):
        return False, "Web fetching is disabled by safety policy."
    if not spec.read_only and not action_tools_allowed(config):
        return False, "Action tools are disabled by safety policy."
    return True, ""
