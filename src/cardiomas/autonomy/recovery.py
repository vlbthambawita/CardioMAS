from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from cardiomas.autonomy.policy import script_codegen_allowed, tool_codegen_allowed, verify_python_ast
from cardiomas.autonomy.verifier import verify_generated_script, verify_generated_tool
from cardiomas.autonomy.workspace import AutonomyWorkspace
from cardiomas.coding.script_builder import build_shell_script
from cardiomas.coding.tool_builder import (
    GeneratedArtifactPackage,
    StandaloneScript,
    build_generated_tool_package,
    build_standalone_script,
)
from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.runtime import RepairTrace
from cardiomas.schemas.tools import ToolResult, ToolSpec


PYTHON_ARTIFACT_TOOL_NAME = "generate_python_artifact"
SHELL_ARTIFACT_TOOL_NAME = "generate_shell_artifact"


class AutonomousToolManager:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.workspace = AutonomyWorkspace(config)
        self._traces: list[RepairTrace] = []
        self._session_id = ""

    @property
    def enabled(self) -> bool:
        return self.config.autonomy.enabled

    def set_session(self, session_id: str) -> None:
        self._session_id = session_id

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
            specs.append(
                ToolSpec(
                    name=PYTHON_ARTIFACT_TOOL_NAME,
                    description="Generate, verify, and execute a query-specific Python artifact for local dataset reading or analysis.",
                    category="autonomy",
                    generated=True,
                )
            )
        if script_codegen_allowed(self.config):
            specs.append(
                ToolSpec(
                    name=SHELL_ARTIFACT_TOOL_NAME,
                    description="Generate a bounded shell artifact for a local dataset task and execute it only when policy allows.",
                    category="autonomy",
                    generated=True,
                )
            )
        return specs

    def generate_python_artifact(
        self,
        task: str,
        dataset_path: str = "",
        target_path: str = "",
        artifact_name: str = "",
    ) -> ToolResult:
        if self.config.autonomy.dataset_mode == "script_only":
            return self._write_standalone_script(task, dataset_path)
        payload = {
            "task": task,
            "dataset_path": dataset_path,
            "target_path": target_path,
            "artifact_name": artifact_name,
        }
        return self._execute_generated_python_artifact(payload)

    def _write_standalone_script(self, task: str, dataset_path: str) -> ToolResult:
        max_attempts = self.config.autonomy.max_repair_attempts + 1
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            trace = RepairTrace(
                tool_name=PYTHON_ARTIFACT_TOOL_NAME,
                action="write_script" if attempt == 1 else "repair_script",
                attempt=attempt,
                workspace_path=str(self.workspace.scripts_output_dir()),
            )
            try:
                script = build_standalone_script(task, dataset_path, self.config, last_error=last_error)
                errors = _verify_standalone_script(script.code, self.config)
                trace.verification = errors or ["syntax OK"]
                if errors:
                    trace.ok = False
                    trace.error = "; ".join(errors)
                    last_error = trace.error
                    self._traces.append(trace)
                    continue

                script_path = self.workspace.write_standalone_script(script)
                trace.files_written = [str(script_path)]

                # Phase 1: write only — no execution
                if not self.config.autonomy.execute_for_answer:
                    trace.retry_succeeded = True
                    self._traces.append(trace)
                    return _standalone_write_result(script, script_path, executed=False)

                # Phase 2: execute and feed output to responder
                if self.config.autonomy.require_approval_for_shell_execution:
                    trace.retry_succeeded = True
                    self._traces.append(trace)
                    return _standalone_write_result(
                        script, script_path, executed=False,
                        execution_error="execution skipped: require_approval_for_shell_execution=True",
                    )

                script.output_dir.mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(
                    ["python", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.config.autonomy.execution_timeout_seconds,
                    cwd=str(script.output_dir.parent),
                    check=False,
                )
                stdout = completed.stdout.strip()
                stderr = completed.stderr.strip()

                stdout_file = script.output_dir / f"stdout_{script.script_name.replace('.py', '')}.txt"
                stdout_file.write_text(stdout, encoding="utf-8")

                if completed.returncode == 0 and stdout:
                    trace.retry_succeeded = True
                    self._traces.append(trace)
                    return _standalone_execute_result(script, script_path, stdout)

                error_msg = stderr or f"Script exited with code {completed.returncode}."
                if not stdout:
                    error_msg = "Script produced no output. " + error_msg
                trace.ok = False
                trace.error = error_msg
                last_error = error_msg
                self._traces.append(trace)

            except subprocess.TimeoutExpired:
                trace.ok = False
                trace.error = f"Timed out after {self.config.autonomy.execution_timeout_seconds}s."
                last_error = trace.error
                self._traces.append(trace)
            except Exception as exc:
                trace.ok = False
                trace.error = str(exc)
                last_error = str(exc)
                self._traces.append(trace)

        return ToolResult(
            tool_name=PYTHON_ARTIFACT_TOOL_NAME,
            ok=False,
            summary="",
            error=f"Standalone script generation failed after {max_attempts} attempt(s): {last_error}",
        )

    def generate_shell_artifact(
        self,
        task: str,
        dataset_path: str = "",
        artifact_name: str = "",
        execute: bool = False,
    ) -> ToolResult:
        session_id = self._ensure_session()
        artifact_slug = artifact_name or _artifact_slug(task, dataset_path, prefix="shell")
        artifact_dir = self.workspace.artifact_dir(session_id, artifact_slug)
        trace = RepairTrace(
            tool_name=SHELL_ARTIFACT_TOOL_NAME,
            action="generate_artifact",
            attempt=1,
            workspace_path=str(artifact_dir),
        )
        if not script_codegen_allowed(self.config):
            trace.ok = False
            trace.error = "Shell artifact generation is disabled by autonomy policy."
            self._traces.append(trace)
            return ToolResult(tool_name=SHELL_ARTIFACT_TOOL_NAME, ok=False, summary="", error=trace.error)

        script = build_shell_script(task=task, dataset_path=dataset_path, config=self.config)
        errors = verify_generated_script(script)
        trace.verification = errors or ["script verification passed"]
        if errors:
            trace.ok = False
            trace.error = "; ".join(errors)
            self._traces.append(trace)
            return ToolResult(tool_name=SHELL_ARTIFACT_TOOL_NAME, ok=False, summary="", error=trace.error)

        spec = ToolSpec(
            name=SHELL_ARTIFACT_TOOL_NAME,
            description="Generated shell artifact saved in the autonomy workspace.",
            category="autonomy",
            generated=True,
        )
        prompt = {
            "tool_name": SHELL_ARTIFACT_TOOL_NAME,
            "task": task,
            "dataset_path": dataset_path,
            "execute_requested": execute,
        }
        context = {"dataset_path": dataset_path, "execution_policy": _shell_execution_policy(self.config, execute)}
        readme = (
            f"# {artifact_slug}\n\n"
            f"- Task: {task or '(none provided)'}\n"
            "- Contract: the generated shell artifact is saved in this directory and may only be executed when policy allows.\n"
        )
        trace.files_written = self.workspace.write_artifact_package(
            session_id=session_id,
            artifact_slug=artifact_slug,
            spec=spec,
            code=script,
            readme=readme,
            prompt=prompt,
            context=context,
            kind="shell",
        )

        script_path = self.workspace.artifact_entrypoint(session_id, artifact_slug, "shell")
        executed = bool(execute and not self.config.autonomy.require_approval_for_shell_execution)
        stdout = ""
        stderr = ""
        error = ""
        if executed:
            completed = subprocess.run(
                ["bash", str(script_path), dataset_path or "."],
                capture_output=True,
                text=True,
                cwd=artifact_dir,
                timeout=60,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            ok = completed.returncode == 0
            if not ok:
                error = stderr.strip() or f"Shell artifact exited with code {completed.returncode}."
            summary = (
                f"Generated and executed shell artifact at {script_path}"
                if ok
                else f"Generated shell artifact at {script_path}, but execution failed."
            )
            extra = {"executed": True, "returncode": completed.returncode}
        else:
            ok = True
            reason = (
                "execution skipped because approval is required by policy"
                if execute and self.config.autonomy.require_approval_for_shell_execution
                else "execution not requested"
            )
            summary = f"Generated shell artifact at {script_path}; {reason}."
            extra = {"executed": False, "reason": reason}

        run_files = self.workspace.write_execution_record(
            session_id=session_id,
            artifact_slug=artifact_slug,
            attempt=1,
            payload={"task": task, "dataset_path": dataset_path, "execute": execute},
            ok=ok,
            summary=summary,
            error=error,
            stdout=stdout,
            stderr=stderr,
            extra=extra,
        )
        trace.retry_succeeded = ok
        if not ok:
            trace.ok = False
            trace.error = error
        self._traces.append(trace)

        evidence = [
            EvidenceChunk(
                chunk_id=f"generated-script:{artifact_slug}",
                source_id="autonomy",
                source_label="generated-shell-artifact",
                source_type="generated_script",
                title=artifact_slug,
                content=script,
                uri=str(script_path),
                metadata={"path": str(script_path), "chunk_label": artifact_slug},
                score=1.0,
            )
        ]
        return ToolResult(
            tool_name=SHELL_ARTIFACT_TOOL_NAME,
            ok=ok,
            summary=summary,
            data={
                "artifact_slug": artifact_slug,
                "artifact_kind": "shell",
                "artifact_dir": str(artifact_dir),
                "artifact_entrypoint": str(script_path),
                "prompt_path": str(artifact_dir / "prompt.json"),
                "context_path": str(artifact_dir / "context.json"),
                "script_path": str(script_path),
                "executed": executed,
                "execution_records": run_files,
                "stdout": stdout,
                "stderr": stderr,
            },
            evidence=evidence,
            error=error,
        )

    def _execute_generated_python_artifact(self, payload: dict[str, Any]) -> ToolResult:
        if not tool_codegen_allowed(self.config):
            return ToolResult(
                tool_name=PYTHON_ARTIFACT_TOOL_NAME,
                ok=False,
                summary="",
                error="Autonomous Python artifact generation is disabled by autonomy policy.",
            )

        session_id = self._ensure_session()
        last_error = ""
        for attempt in range(1, self.config.autonomy.max_repair_attempts + 2):
            trace = RepairTrace(
                tool_name=PYTHON_ARTIFACT_TOOL_NAME,
                action="generate_artifact" if attempt == 1 else "repair_artifact",
                attempt=attempt,
                workspace_path=str(self.workspace.session_dir(session_id)),
            )
            try:
                package = build_generated_tool_package(PYTHON_ARTIFACT_TOOL_NAME, payload, self.config, last_error=last_error)
                trace.workspace_path = str(self.workspace.artifact_dir(session_id, package.artifact_slug))
                trace.files_written = self.workspace.write_artifact_package(
                    session_id=session_id,
                    artifact_slug=package.artifact_slug,
                    spec=package.spec,
                    code=package.code,
                    readme=package.readme,
                    prompt=package.prompt,
                    context=package.context,
                    kind=package.kind,
                )
                errors = verify_generated_tool(self.workspace, session_id, package.artifact_slug, self.config)
                trace.verification = errors or ["tool verification passed"]
                if errors:
                    trace.ok = False
                    trace.error = "; ".join(errors)
                    last_error = trace.error
                    self._traces.append(trace)
                    continue

                result = self._run_python_artifact(package, payload, session_id, attempt)
                if result.ok:
                    artifact_stdout = (result.data or {}).get("stdout", "")
                    if result.summary.strip() or artifact_stdout.strip():
                        trace.retry_succeeded = True
                        self._traces.append(trace)
                        return result
                    trace.ok = False
                    trace.error = (
                        "Artifact ran without errors but produced no output. "
                        "Read the actual data files and print computed results (e.g. unique patient count)."
                    )
                    last_error = trace.error
                    self._traces.append(trace)
                    continue

                trace.ok = False
                trace.error = result.error or "Generated Python artifact returned ok=False."
                last_error = trace.error
                self._traces.append(trace)
            except Exception as exc:
                trace.ok = False
                trace.error = str(exc)
                last_error = str(exc)
                self._traces.append(trace)

        return ToolResult(
            tool_name=PYTHON_ARTIFACT_TOOL_NAME,
            ok=False,
            summary="",
            error=f"Autonomous Python artifact generation failed: {last_error or 'unknown error'}",
        )

    def _run_python_artifact(
        self,
        package: GeneratedArtifactPackage,
        payload: dict[str, Any],
        session_id: str,
        attempt: int,
    ) -> ToolResult:
        artifact_dir = self.workspace.artifact_dir(session_id, package.artifact_slug)
        output_dir = self.workspace.artifact_output_dir(session_id, package.artifact_slug)
        execution_payload = dict(payload)
        execution_payload.update(
            {
                "artifact_output_dir": str(output_dir),
                "artifact_slug": package.artifact_slug,
                "session_id": session_id,
            }
        )
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            module = self.workspace.load_tool_module(session_id, package.artifact_slug)
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                raw_result = module.run(execution_payload)
            result = _normalize_tool_result(PYTHON_ARTIFACT_TOOL_NAME, raw_result)
        except Exception as exc:
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
            error = str(exc)
            run_files = self.workspace.write_execution_record(
                session_id=session_id,
                artifact_slug=package.artifact_slug,
                attempt=attempt,
                payload=execution_payload,
                ok=False,
                summary="",
                error=error,
                stdout=stdout,
                stderr=stderr,
                extra={"artifact_kind": "python"},
            )
            return ToolResult(
                tool_name=PYTHON_ARTIFACT_TOOL_NAME,
                ok=False,
                summary="",
                error=error,
                data={
                    "artifact_slug": package.artifact_slug,
                    "artifact_kind": "python",
                    "artifact_dir": str(artifact_dir),
                    "artifact_entrypoint": str(self.workspace.artifact_entrypoint(session_id, package.artifact_slug, "python")),
                    "execution_records": run_files,
                },
            )

        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()
        run_files = self.workspace.write_execution_record(
            session_id=session_id,
            artifact_slug=package.artifact_slug,
            attempt=attempt,
            payload=execution_payload,
            ok=result.ok,
            summary=result.summary,
            error=result.error,
            stdout=stdout,
            stderr=stderr,
            extra={"artifact_kind": "python"},
        )
        data = dict(result.data)
        data.update(
            {
                "artifact_slug": package.artifact_slug,
                "artifact_kind": "python",
                "artifact_dir": str(artifact_dir),
                "artifact_entrypoint": str(self.workspace.artifact_entrypoint(session_id, package.artifact_slug, "python")),
                "prompt_path": str(artifact_dir / "prompt.json"),
                "context_path": str(artifact_dir / "context.json"),
                "execution_records": run_files,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        evidence = list(result.evidence)
        if not evidence:
            evidence.append(
                EvidenceChunk(
                    chunk_id=f"generated-artifact:{package.artifact_slug}",
                    source_id="autonomy",
                    source_label="generated-python-artifact",
                    source_type="generated_artifact",
                    title=package.artifact_slug,
                    content=result.summary,
                    uri=str(output_dir / "result.json"),
                    metadata={"chunk_label": package.artifact_slug, "artifact_slug": package.artifact_slug},
                    score=1.0,
                )
            )
        return ToolResult(
            tool_name=PYTHON_ARTIFACT_TOOL_NAME,
            ok=result.ok,
            summary=result.summary,
            data=data,
            evidence=evidence,
            error=result.error,
        )

    def _ensure_session(self) -> str:
        if not self._session_id:
            self._session_id = f"adhoc-{uuid4().hex[:8]}"
        return self._session_id


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


def _artifact_slug(task: str, dataset_path: str, prefix: str) -> str:
    base = _slug(task or prefix)
    digest = hashlib.sha1(f"{prefix}|{dataset_path}|{task}".encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{base[:32] or 'artifact'}-{digest}"


def _verify_standalone_script(code: str, config: RuntimeConfig) -> list[str]:
    errors: list[str] = []
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error in generated script: {exc}"]
    if "def main()" not in code:
        errors.append("Missing main() function in generated script.")
    if '__name__ == "__main__"' not in code and "__name__ == '__main__'" not in code:
        errors.append("Missing if __name__ == '__main__' guard.")
    errors.extend(verify_python_ast(code, config))
    return errors


def _standalone_write_result(
    script: StandaloneScript,
    script_path: Path,
    executed: bool,
    execution_error: str = "",
) -> ToolResult:
    evidence = [
        EvidenceChunk(
            chunk_id=f"standalone-script:{script.script_name}",
            source_id="autonomy",
            source_label="generated-python-artifact",
            source_type="generated_artifact",
            title=script.script_name,
            content=f"Script written to: {script_path}\n{script.description}",
            uri=str(script_path),
            metadata={"chunk_label": script.script_name, "script_name": script.script_name},
            score=1.0,
        )
    ]
    return ToolResult(
        tool_name=PYTHON_ARTIFACT_TOOL_NAME,
        ok=True,
        summary=f"Script written to {script_path}",
        data={
            "script_path": str(script_path),
            "script_name": script.script_name,
            "description": script.description,
            "dataset_path": "",
            "output_dir": str(script.output_dir),
            "is_standalone": True,
            "executed": executed,
            "execution_stdout": "",
            "execution_error": execution_error,
        },
        evidence=evidence,
    )


def _standalone_execute_result(script: StandaloneScript, script_path: Path, stdout: str) -> ToolResult:
    evidence = [
        EvidenceChunk(
            chunk_id=f"script-output:{script.script_name}",
            source_id="autonomy",
            source_label="generated-python-artifact",
            source_type="script_output",
            title=f"Script output: {script.script_name}",
            content=stdout[:4000],
            uri=str(script_path),
            metadata={"chunk_label": script.script_name, "script_name": script.script_name},
            score=2.0,
        ),
        EvidenceChunk(
            chunk_id=f"standalone-script:{script.script_name}",
            source_id="autonomy",
            source_label="generated-python-artifact",
            source_type="generated_artifact",
            title=script.script_name,
            content=f"Script at: {script_path}\n{script.description}",
            uri=str(script_path),
            metadata={"chunk_label": script.script_name, "script_name": script.script_name},
            score=1.0,
        ),
    ]
    return ToolResult(
        tool_name=PYTHON_ARTIFACT_TOOL_NAME,
        ok=True,
        summary=stdout[:500],
        data={
            "script_path": str(script_path),
            "script_name": script.script_name,
            "description": script.description,
            "output_dir": str(script.output_dir),
            "is_standalone": True,
            "executed": True,
            "execution_stdout": stdout,
            "execution_error": "",
        },
        evidence=evidence,
    )


def _shell_execution_policy(config: RuntimeConfig, execute_requested: bool) -> dict[str, Any]:
    return {
        "execute_requested": execute_requested,
        "allowed_without_approval": not config.autonomy.require_approval_for_shell_execution,
        "require_approval_for_shell_execution": config.autonomy.require_approval_for_shell_execution,
    }


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
