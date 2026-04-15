"""
SessionRecorder — singleton that captures every LLM call and agent step
during a pipeline run and writes them to output/session_log/.

Usage in agents:
    from cardiomas.recorder import SessionRecorder
    rec = SessionRecorder.get()
    rec.start_step("analysis", "scan_files", inputs={"path": ...})
    rec.record_llm_call(agent="analysis", model="llama3.1:8b", ...)
    rec.end_step(outputs={"files": ...}, reasoning="...")
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from cardiomas.schemas.session import AgentStep, LLMCall, SessionLog


class SessionRecorder:
    """Thread-local pipeline-run recorder."""

    _instance: "SessionRecorder | None" = None

    def __init__(self) -> None:
        self._log: SessionLog | None = None
        self._current_step: AgentStep | None = None
        self._step_start_ms: float = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "SessionRecorder":
        if cls._instance is None:
            cls._instance = SessionRecorder()
        return cls._instance

    @classmethod
    def reset(cls) -> "SessionRecorder":
        """Create a fresh recorder (call at pipeline start)."""
        cls._instance = SessionRecorder()
        return cls._instance

    def start_session(
        self,
        dataset_name: str,
        user_options: dict,
        raw_requirement: str | None = None,
    ) -> str:
        from cardiomas import __version__

        session_id = str(uuid.uuid4())
        self._log = SessionLog(
            session_id=session_id,
            cardiomas_version=__version__,
            started_at=datetime.utcnow(),
            dataset_name=dataset_name,
            raw_requirement=raw_requirement,
            user_options=user_options,
        )
        return session_id

    # ── Step recording ─────────────────────────────────────────────────────

    def start_step(self, agent: str, action: str = "", inputs: dict | None = None) -> None:
        self._current_step = AgentStep(
            agent=agent,
            action=action,
            inputs=inputs or {},
        )
        self._step_start_ms = time.monotonic() * 1000

    def record_llm_call(
        self,
        agent: str,
        model: str = "",
        system_prompt: str = "",
        user_message: str = "",
        response: str = "",
        duration_ms: int = 0,
        compressed: bool = False,
        original_context_len: int = 0,
    ) -> None:
        call = LLMCall(
            agent=agent,
            model=model,
            system_prompt=system_prompt[:600],
            user_message=user_message[:600],
            response=response[:1200],
            duration_ms=duration_ms,
            compressed=compressed,
            original_context_len=original_context_len,
        )
        if self._current_step is not None:
            self._current_step.llm_calls.append(call)

    def end_step(
        self,
        outputs: dict | None = None,
        reasoning: str = "",
        skipped: bool = False,
        skip_reason: str = "",
    ) -> None:
        if self._current_step is None or self._log is None:
            return
        self._current_step.outputs = outputs or {}
        self._current_step.reasoning = reasoning
        self._current_step.skipped = skipped
        self._current_step.skip_reason = skip_reason
        self._current_step.duration_ms = int(time.monotonic() * 1000 - self._step_start_ms)
        self._log.agent_steps.append(self._current_step)
        self._current_step = None

    def add_orchestrator_reasoning(self, reasoning: str) -> None:
        if self._log is not None:
            self._log.orchestrator_reasoning.append(reasoning)

    def add_error(self, error: str) -> None:
        if self._log is not None:
            self._log.errors.append(error)

    # ── Session finish ─────────────────────────────────────────────────────

    def finish_session(self, status: str = "ok") -> SessionLog | None:
        if self._log is None:
            return None
        self._log.completed_at = datetime.utcnow()
        self._log.final_status = status
        return self._log

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, output_dir: Path) -> dict[str, Path]:
        """Save session.json, conversation.md, reasoning_trace.md."""
        if self._log is None:
            return {}
        log_dir = output_dir / "session_log"
        log_dir.mkdir(parents=True, exist_ok=True)

        session_json = log_dir / "session.json"
        session_json.write_text(
            json.dumps(self._log.model_dump(mode="json"), indent=2, default=str)
        )

        conv_md = log_dir / "conversation.md"
        conv_md.write_text(self._build_conversation_md())

        trace_md = log_dir / "reasoning_trace.md"
        trace_md.write_text(self._build_reasoning_trace_md())

        return {
            "session.json": session_json,
            "conversation.md": conv_md,
            "reasoning_trace.md": trace_md,
        }

    # ── Markdown builders ─────────────────────────────────────────────────

    def _build_conversation_md(self) -> str:
        if self._log is None:
            return ""
        lines: list[str] = [
            f"# CardioMAS Session — {self._log.dataset_name}",
            "",
            f"- **Session ID:** `{self._log.session_id}`",
            f"- **Version:** {self._log.cardiomas_version}",
            f"- **Started:** {self._log.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"- **Status:** {self._log.final_status}",
            "",
        ]
        if self._log.raw_requirement:
            lines += [
                "## User Requirement",
                "",
                f"> {self._log.raw_requirement}",
                "",
            ]
        for step in self._log.agent_steps:
            lines.append(f"## {step.agent.replace('_', ' ').title()} Agent")
            lines.append("")
            if step.skipped:
                lines.append(f"**Skipped:** {step.skip_reason}")
            else:
                lines.append(f"**Action:** {step.action}")
                if step.reasoning:
                    lines.append(f"\n**Reasoning:** {step.reasoning}")
                for call in step.llm_calls:
                    model_tag = f" `{call.model}`" if call.model else ""
                    lines.append(f"\n### LLM Call{model_tag}")
                    if call.compressed:
                        lines.append(f"*Context compressed: {call.original_context_len} → {len(call.user_message)} chars*")
                    lines.append(f"\n**Prompt:**")
                    lines.append(f"```\n{call.user_message}\n```")
                    lines.append(f"\n**Response:**")
                    lines.append(f"```\n{call.response}\n```")
            lines.append("")
        if self._log.errors:
            lines += ["## Errors", ""]
            for err in self._log.errors:
                lines.append(f"- {err}")
            lines.append("")
        return "\n".join(lines)

    def _build_reasoning_trace_md(self) -> str:
        if self._log is None:
            return ""
        lines: list[str] = [
            f"# Reasoning Trace — {self._log.dataset_name}",
            "",
            f"Session: `{self._log.session_id}`",
            "",
        ]
        for i, reason in enumerate(self._log.orchestrator_reasoning, 1):
            lines.append(f"## Step {i}")
            lines.append("")
            lines.append(reason)
            lines.append("")
        return "\n".join(lines)
