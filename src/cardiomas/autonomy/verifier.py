from __future__ import annotations

from cardiomas.autonomy.policy import verify_python_ast, verify_script_text
from cardiomas.autonomy.workspace import AutonomyWorkspace
from cardiomas.schemas.config import RuntimeConfig


def verify_generated_tool_source(source: str, config: RuntimeConfig) -> list[str]:
    return verify_python_ast(source, config)


def verify_generated_tool(
    workspace: AutonomyWorkspace,
    session_id: str,
    artifact_slug: str,
    config: RuntimeConfig,
) -> list[str]:
    entrypoint = workspace.artifact_entrypoint(session_id, artifact_slug, "python")
    if not entrypoint.exists():
        return [f"Generated tool entrypoint does not exist: {entrypoint}"]

    errors = verify_generated_tool_source(entrypoint.read_text(encoding="utf-8"), config)
    if errors:
        return errors

    module = workspace.load_tool_module(session_id, artifact_slug)
    run = getattr(module, "run", None)
    if run is None or not callable(run):
        return ["Generated tool does not expose a callable run(payload) function."]
    return []


def verify_generated_script(script: str) -> list[str]:
    return verify_script_text(script)
