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

    def tool_dir(self, tool_name: str) -> Path:
        directory = self.root / "tools" / tool_name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def tool_entrypoint(self, tool_name: str) -> Path:
        return self.tool_dir(tool_name) / "tool.py"

    def tool_spec_path(self, tool_name: str) -> Path:
        return self.tool_dir(tool_name) / "tool_spec.json"

    def write_tool_package(self, tool_name: str, spec: ToolSpec, code: str, readme: str = "") -> list[str]:
        directory = self.tool_dir(tool_name)
        entrypoint = directory / "tool.py"
        spec_path = directory / "tool_spec.json"
        readme_path = directory / "README.md"

        entrypoint.write_text(code, encoding="utf-8")
        spec_path.write_text(json.dumps(spec.model_dump(mode="json"), indent=2), encoding="utf-8")
        if readme:
            readme_path.write_text(readme, encoding="utf-8")

        written = [str(entrypoint), str(spec_path)]
        if readme:
            written.append(str(readme_path))
        return written

    def load_tool_spec(self, tool_name: str) -> ToolSpec | None:
        path = self.tool_spec_path(tool_name)
        if not path.exists():
            return None
        return ToolSpec.model_validate_json(path.read_text(encoding="utf-8"))

    def list_generated_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        tools_root = self.root / "tools"
        if not tools_root.exists():
            return specs
        for spec_path in sorted(tools_root.glob("*/tool_spec.json")):
            specs.append(ToolSpec.model_validate_json(spec_path.read_text(encoding="utf-8")))
        return specs

    def load_tool_module(self, tool_name: str) -> ModuleType:
        path = self.tool_entrypoint(tool_name)
        if not path.exists():
            raise FileNotFoundError(f"Generated tool entrypoint not found: {path}")
        module_name = f"cardiomas_generated_{tool_name}_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load generated tool module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def write_script(self, script_name: str, content: str) -> Path:
        scripts_dir = self.root / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        path = scripts_dir / script_name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
        return path
