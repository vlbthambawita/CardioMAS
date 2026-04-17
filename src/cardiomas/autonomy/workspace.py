from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from uuid import uuid4

from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.tools import ToolSpec


class AutonomyWorkspace:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.root = config.autonomy_workspace_path
        self.root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        directory = self.root / "sessions" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def artifact_dir(self, session_id: str, artifact_slug: str) -> Path:
        directory = self.session_dir(session_id) / artifact_slug
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "outputs").mkdir(parents=True, exist_ok=True)
        return directory

    def artifact_output_dir(self, session_id: str, artifact_slug: str) -> Path:
        return self.artifact_dir(session_id, artifact_slug) / "outputs"

    def artifact_entrypoint(self, session_id: str, artifact_slug: str, kind: str) -> Path:
        filename = "tool.py" if kind == "python" else "script.sh"
        return self.artifact_dir(session_id, artifact_slug) / filename

    def write_artifact_package(
        self,
        session_id: str,
        artifact_slug: str,
        spec: ToolSpec,
        code: str,
        readme: str,
        prompt: dict,
        context: dict,
        kind: str,
    ) -> list[str]:
        directory = self.artifact_dir(session_id, artifact_slug)
        entrypoint = self.artifact_entrypoint(session_id, artifact_slug, kind)
        prompt_path = directory / "prompt.json"
        context_path = directory / "context.json"
        spec_path = directory / "tool_spec.json"
        readme_path = directory / "README.md"

        entrypoint.write_text(code, encoding="utf-8")
        if kind == "shell":
            entrypoint.chmod(0o755)
        prompt_path.write_text(json.dumps(prompt, indent=2, default=str), encoding="utf-8")
        context_path.write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
        spec_path.write_text(json.dumps(spec.model_dump(mode="json"), indent=2), encoding="utf-8")
        if readme:
            readme_path.write_text(readme, encoding="utf-8")

        written = [str(entrypoint), str(prompt_path), str(context_path), str(spec_path)]
        if readme:
            written.append(str(readme_path))
        return written

    def write_execution_record(
        self,
        session_id: str,
        artifact_slug: str,
        attempt: int,
        payload: dict,
        ok: bool,
        summary: str,
        error: str,
        stdout: str,
        stderr: str,
        extra: dict | None = None,
    ) -> list[str]:
        directory = self.artifact_dir(session_id, artifact_slug)
        suffix = f"{attempt:02d}"
        run_path = directory / f"run-{suffix}.json"
        stdout_path = directory / f"stdout-{suffix}.txt"
        stderr_path = directory / f"stderr-{suffix}.txt"
        record = {
            "attempt": attempt,
            "ok": ok,
            "summary": summary,
            "error": error,
            "payload": payload,
            "extra": extra or {},
        }
        run_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return [str(run_path), str(stdout_path), str(stderr_path)]

    def load_tool_module(self, session_id: str, artifact_slug: str) -> ModuleType:
        path = self.artifact_entrypoint(session_id, artifact_slug, "python")
        if not path.exists():
            raise FileNotFoundError(f"Generated tool entrypoint not found: {path}")
        module_name = f"cardiomas_generated_{artifact_slug}_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load generated tool module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
