from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cardiomas.autonomy.policy import script_codegen_allowed, tool_codegen_allowed
from cardiomas.autonomy.verifier import verify_generated_script, verify_generated_tool
from cardiomas.autonomy.workspace import AutonomyWorkspace
from cardiomas.coding.script_builder import build_shell_script
from cardiomas.coding.tool_builder import build_generated_tool_package
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.runtime import RepairTrace
from cardiomas.schemas.tools import ToolResult, ToolSpec


AUTONOMOUS_TOOL_NAMES = ("dataset_statistics", "generate_shell_script", "read_dataset_file")


class AutonomousToolManager:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.workspace = AutonomyWorkspace(config)
        self._traces: list[RepairTrace] = []

    @property
    def enabled(self) -> bool:
        return self.config.autonomy.enabled

    def reset_traces(self) -> None:
        self._traces = []

    def consume_traces(self) -> list[RepairTrace]:
        traces = list(self._traces)
        self._traces = []
        return traces

    def peek_traces(self) -> list[RepairTrace]:
        return list(self._traces)

    def tool_specs(self) -> list[ToolSpec]:
        if not self.enabled:
            return []
        specs: list[ToolSpec] = []
        if tool_codegen_allowed(self.config):
            specs.extend(
                [
                    ToolSpec(
                        name="read_dataset_file",
                        description="Generate or repair a dataset file reader and inspect a specific file or reader-compatible dataset path.",
                        category="autonomy",
                        generated=True,
                    ),
                    ToolSpec(
                        name="dataset_statistics",
                        description="Generate or repair a dataset statistics tool for tabular or mixed local datasets.",
                        category="autonomy",
                        generated=True,
                    ),
                ]
            )
        if script_codegen_allowed(self.config):
            specs.append(
                ToolSpec(
                    name="generate_shell_script",
                    description="Generate a bounded shell script in the autonomy workspace for a local dataset task.",
                    category="autonomy",
                    generated=True,
                )
            )
        for spec in self.workspace.list_generated_specs():
            if spec.name not in {item.name for item in specs}:
                specs.append(spec)
        return specs

    def read_dataset_file(self, target_path: str = "", dataset_path: str = "", max_preview_lines: int = 40) -> ToolResult:
        payload = {"target_path": target_path, "dataset_path": dataset_path, "max_preview_lines": max_preview_lines}
        return self._execute_generated_tool("read_dataset_file", payload)

    def dataset_statistics(self, dataset_path: str = "", target_file: str = "") -> ToolResult:
        payload = {"dataset_path": dataset_path, "target_file": target_file}
        return self._execute_generated_tool("dataset_statistics", payload)

    def generate_shell_script(self, task: str, dataset_path: str = "", script_name: str = "") -> ToolResult:
        trace = RepairTrace(
            tool_name="generate_shell_script",
            action="generate_script",
            attempt=1,
            workspace_path=str(self.workspace.root),
        )
        if not script_codegen_allowed(self.config):
            trace.ok = False
            trace.error = "Shell script generation is disabled by autonomy policy."
            self._traces.append(trace)
            return ToolResult(
                tool_name="generate_shell_script",
                ok=False,
                summary="",
                error=trace.error,
            )

        script = build_shell_script(task=task, dataset_path=dataset_path, config=self.config)
        errors = verify_generated_script(script)
        trace.verification = errors or ["script verification passed"]
        if errors:
            trace.ok = False
            trace.error = "; ".join(errors)
            self._traces.append(trace)
            return ToolResult(tool_name="generate_shell_script", ok=False, summary="", error=trace.error)

        normalized_name = script_name or _slug(task) or "generated_task.sh"
        if not normalized_name.endswith(".sh"):
            normalized_name = f"{normalized_name}.sh"
        script_path = self.workspace.write_script(normalized_name, script)
        trace.files_written = [str(script_path)]
        trace.retry_succeeded = True
        self._traces.append(trace)
        content = script_path.read_text(encoding="utf-8")
        evidence = [
            EvidenceChunk(
                chunk_id=f"generated-script:{normalized_name}",
                source_id="autonomy",
                source_label="generated-script",
                source_type="generated_script",
                title=normalized_name,
                content=content,
                uri=str(script_path),
                metadata={"path": str(script_path), "chunk_label": normalized_name},
                score=1.0,
            )
        ]
        return ToolResult(
            tool_name="generate_shell_script",
            ok=True,
            summary=f"Generated shell script at {script_path}",
            data={"script_path": str(script_path), "script_name": normalized_name, "task": task, "script": content},
            evidence=evidence,
        )

    def _execute_generated_tool(self, tool_name: str, payload: dict[str, Any]) -> ToolResult:
        if not tool_codegen_allowed(self.config):
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                summary="",
                error="Autonomous tool generation is disabled by autonomy policy.",
            )

        last_error = ""
        for attempt in range(1, self.config.autonomy.max_repair_attempts + 2):
            action = "generate" if attempt == 1 else "repair"
            trace = RepairTrace(
                tool_name=tool_name,
                action=action,
                attempt=attempt,
                workspace_path=str(self.workspace.tool_dir(tool_name)),
            )
            try:
                spec, code, readme = build_generated_tool_package(tool_name, payload, self.config, last_error=last_error)
                trace.files_written = self.workspace.write_tool_package(tool_name, spec, code, readme)
                errors = verify_generated_tool(self.workspace, tool_name, self.config)
                trace.verification = errors or ["tool verification passed"]
                if errors:
                    trace.ok = False
                    trace.error = "; ".join(errors)
                    last_error = trace.error
                    self._traces.append(trace)
                    continue

                module = self.workspace.load_tool_module(tool_name)
                result_payload = module.run(payload)
                result = _normalize_tool_result(tool_name, result_payload)
                if result.ok:
                    trace.retry_succeeded = True
                    self._traces.append(trace)
                    return result

                trace.ok = False
                trace.error = result.error or "Generated tool returned ok=False."
                last_error = trace.error
                self._traces.append(trace)
            except Exception as exc:
                trace.ok = False
                trace.error = str(exc)
                last_error = str(exc)
                self._traces.append(trace)

        return ToolResult(
            tool_name=tool_name,
            ok=False,
            summary="",
            error=f"Autonomous tool generation failed for {tool_name}: {last_error or 'unknown error'}",
        )


def _normalize_tool_result(tool_name: str, payload: Any) -> ToolResult:
    if isinstance(payload, ToolResult):
        return payload
    if not isinstance(payload, dict):
        return ToolResult(tool_name=tool_name, ok=False, summary="", error="Generated tool returned a non-dict payload.")

    evidence_items = payload.get("evidence", [])
    evidence: list[EvidenceChunk] = []
    for item in evidence_items:
        if isinstance(item, EvidenceChunk):
            evidence.append(item)
        elif isinstance(item, dict):
            evidence.append(EvidenceChunk.model_validate(item))

    data = payload.get("data", {})
    if isinstance(data, (list, str, int, float, bool)):
        data = {"value": data}
    if not isinstance(data, dict):
        data = {"value": json.dumps(data, default=str)}

    return ToolResult(
        tool_name=tool_name,
        ok=bool(payload.get("ok", False)),
        summary=str(payload.get("summary", "")),
        data=data,
        evidence=evidence,
        error=str(payload.get("error", "")),
    )


def _slug(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    return normalized[:48]
