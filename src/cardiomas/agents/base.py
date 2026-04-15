from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from cardiomas.verbose import vprint, vprint_llm

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentOutputError(RuntimeError):
    """Raised when a structured agent cannot produce a valid output after retries."""


def load_skill(skill_name: str) -> str:
    """Load an agent skill .md file as a system prompt string."""
    skills_dir = Path(__file__).parent.parent / "skills"
    path = skills_dir / f"{skill_name}.md"
    if path.exists():
        return path.read_text()
    return f"You are an expert agent for {skill_name}. Be precise and cite sources."


def _compress_context(context: str, agent_name: str) -> tuple[str, bool, int]:
    """Compress context if it exceeds the configured threshold.

    Returns:
        (compressed_text, was_compressed, original_len)
    """
    import cardiomas.config as cfg
    from cardiomas.llm_factory import get_local_llm

    original_len = len(context)
    if original_len <= cfg.CONTEXT_COMPRESS_THRESHOLD:
        return context, False, original_len

    vprint(agent_name, f"[dim]context too long ({original_len} chars) — compressing with {cfg.CONTEXT_COMPRESS_MODEL}…[/dim]")
    try:
        compress_llm = get_local_llm(temperature=0.0, model=cfg.CONTEXT_COMPRESS_MODEL)
        messages = [
            SystemMessage(content=(
                f"Summarise the following for use by a {agent_name} agent. "
                "Preserve all field names, numbers, URLs, dataset names, and section references exactly. "
                "Be concise but complete for the agent's task. Output only the summary."
            )),
            HumanMessage(content=context[:8000]),  # hard cap to avoid recursion
        ]
        result = compress_llm.invoke(messages)
        compressed = result.content if hasattr(result, "content") else str(result)
        vprint(agent_name, f"[dim]compressed: {original_len} → {len(compressed)} chars[/dim]")
        return compressed, True, original_len
    except Exception as e:
        logger.warning(f"Context compression failed for {agent_name}: {e} — truncating")
        truncated = context[:cfg.CONTEXT_COMPRESS_THRESHOLD]
        return truncated, True, original_len


def run_agent(
    llm: BaseChatModel,
    skill_name: str,
    user_message: str,
    extra_context: str = "",
) -> str:
    """Run an agent with a skill system prompt and return the text response.

    Records the LLM call in SessionRecorder and handles context compression.
    """
    from cardiomas.recorder import SessionRecorder

    system_prompt = load_skill(skill_name)

    # Compress context if needed
    compressed_context, was_compressed, original_len = _compress_context(extra_context, skill_name)
    if compressed_context:
        system_prompt += f"\n\n## Context\n{compressed_context}"

    # Build display prompt (truncated for verbose display)
    display_prompt = (
        f"{user_message}\n\n---\nContext:\n{compressed_context[:600]}"
        if compressed_context else user_message
    )

    # Extract model name for display and recording
    model_name: str = (
        getattr(llm, "model", "")
        or getattr(llm, "model_name", "")
        or llm.__class__.__name__
    )
    vprint(skill_name, f"calling LLM [{model_name}]…")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    start_ms = time.monotonic() * 1000
    try:
        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        duration_ms = int(time.monotonic() * 1000 - start_ms)

        vprint_llm(skill_name, display_prompt, text, model_name=model_name)

        # Record in session recorder
        SessionRecorder.get().record_llm_call(
            agent=skill_name,
            model=model_name,
            system_prompt=system_prompt[:600],
            user_message=user_message[:600],
            response=text[:1200],
            duration_ms=duration_ms,
            compressed=was_compressed,
            original_context_len=original_len,
        )
        return text
    except Exception as e:
        logger.error(f"Agent {skill_name} failed: {e}")
        vprint(skill_name, f"[red]ERROR: {e}[/red]")
        SessionRecorder.get().add_error(f"{skill_name}: {e}")
        return f"ERROR: {e}"


def _schema_hint(output_schema: Type[BaseModel]) -> str:
    """Build a compact JSON schema hint to embed in the system prompt."""
    try:
        schema = output_schema.model_json_schema()
        props = schema.get("properties", {})
        lines = [f'  "{k}": {v.get("description", v.get("type", "?"))}' for k, v in props.items()]
        return "{\n" + ",\n".join(lines) + "\n}"
    except Exception:
        return "{}"


def run_structured_agent(
    llm: BaseChatModel,
    skill_name: str,
    user_message: str,
    output_schema: Type[T],
    extra_context: str = "",
    max_retries: int = 1,
) -> T:
    """Run an agent and return a validated Pydantic output model.

    Strategy:
    1. Embed the JSON schema in the system prompt.
    2. Call the LLM (using Ollama JSON mode when available).
    3. Parse response → validate against output_schema.
    4. On ValidationError, send a correction prompt (once).
    5. On second failure, raise AgentOutputError.

    Records all LLM calls in SessionRecorder.
    """
    from cardiomas.recorder import SessionRecorder

    system_prompt = load_skill(skill_name)
    schema_hint = _schema_hint(output_schema)
    system_prompt += (
        f"\n\n## Output Format\n"
        f"Respond with ONLY a valid JSON object matching this schema (no markdown, no extra text):\n"
        f"```json\n{schema_hint}\n```"
    )

    compressed_context, was_compressed, original_len = _compress_context(extra_context, skill_name)
    if compressed_context:
        system_prompt += f"\n\n## Context\n{compressed_context}"

    model_name: str = (
        getattr(llm, "model", "")
        or getattr(llm, "model_name", "")
        or llm.__class__.__name__
    )

    # Use Ollama JSON mode if available (guarantees syntactically valid JSON)
    invoke_llm = llm
    try:
        from langchain_ollama import ChatOllama
        if isinstance(llm, ChatOllama):
            invoke_llm = llm.model_copy(update={"format": "json"})
    except ImportError:
        pass

    def _call(messages: list) -> str:
        vprint(skill_name, f"calling LLM [{model_name}] (structured)…")
        start_ms = time.monotonic() * 1000
        response = invoke_llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        duration_ms = int(time.monotonic() * 1000 - start_ms)
        display_prompt = user_message[:300]
        vprint_llm(skill_name, display_prompt, text, model_name=model_name)
        SessionRecorder.get().record_llm_call(
            agent=skill_name,
            model=model_name,
            system_prompt=system_prompt[:600],
            user_message=user_message[:600],
            response=text[:1200],
            duration_ms=duration_ms,
            compressed=was_compressed,
            original_context_len=original_len,
        )
        return text

    def _parse(text: str) -> T:
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()
        # Find the outermost JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in response")
        data = json.loads(text[start:end])
        return output_schema.model_validate(data)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            text = _call(messages)
            return _parse(text)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            if attempt < max_retries:
                vprint(skill_name, f"[yellow]output validation failed (attempt {attempt + 1}): {exc} — retrying with correction[/yellow]")
                logger.warning(f"{skill_name}: structured output attempt {attempt + 1} failed: {exc}")
                # Add correction turn
                correction = (
                    f"Your previous response could not be parsed. Error: {exc}\n\n"
                    f"Respond with ONLY a valid JSON object matching the schema. No extra text."
                )
                messages = messages + [
                    HumanMessage(content=f"Previous response: {text[:400]}"),
                    HumanMessage(content=correction),
                ]
            else:
                logger.error(f"{skill_name}: structured output failed after {max_retries + 1} attempts: {exc}")
                SessionRecorder.get().add_error(f"{skill_name}: structured output failed: {exc}")
        except Exception as exc:
            last_error = exc
            logger.error(f"{skill_name}: unexpected error in structured agent: {exc}")
            SessionRecorder.get().add_error(f"{skill_name}: {exc}")
            break

    raise AgentOutputError(
        f"{skill_name} could not produce valid {output_schema.__name__} after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
