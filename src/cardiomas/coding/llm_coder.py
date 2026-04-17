from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from cardiomas.inference.base import ChatClient, ChatMessage, ChatRequest
from cardiomas.schemas.config import RuntimeConfig

_TABULAR_SUFFIXES = {".csv", ".tsv"}
_MAX_TABULAR_FILES = 5
_MAX_SAMPLE_FILES = 30
_MAX_COLUMNS_PER_FILE = 50


def discover_dataset_structure(dataset_path: str) -> dict[str, Any]:
    """Fast filesystem probe — no LLM. Returns actual column names from CSV/TSV files."""
    root = Path(dataset_path).expanduser()
    if not root.exists() or not root.is_dir():
        return {
            "dataset_root": dataset_path,
            "total_files": 0,
            "extension_counts": {},
            "tabular_files": [],
            "sample_files": [],
        }

    files = sorted(p for p in root.rglob("*") if p.is_file())

    extension_counts: dict[str, int] = {}
    for f in files:
        ext = f.suffix.lower() or "<no_ext>"
        extension_counts[ext] = extension_counts.get(ext, 0) + 1

    sample_files = [str(f.relative_to(root)) for f in files[:_MAX_SAMPLE_FILES]]

    tabular_files: list[dict[str, Any]] = []
    for f in files:
        if f.suffix.lower() not in _TABULAR_SUFFIXES:
            continue
        if len(tabular_files) >= _MAX_TABULAR_FILES:
            break
        try:
            delimiter = "\t" if f.suffix.lower() == ".tsv" else ","
            with f.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                header = next(reader, [])
                columns = header[:_MAX_COLUMNS_PER_FILE]
                row_count = sum(1 for _ in reader)
            tabular_files.append({
                "path": str(f.relative_to(root)),
                "absolute_path": str(f),
                "columns": columns,
                "row_count": row_count,
            })
        except Exception:
            pass

    return {
        "dataset_root": str(root),
        "total_files": len(files),
        "extension_counts": dict(sorted(extension_counts.items())),
        "tabular_files": tabular_files,
        "sample_files": sample_files,
    }


def _format_dataset_context(ctx: dict[str, Any]) -> str:
    lines = [
        f"Dataset root: {ctx.get('dataset_root', '(unknown)')}",
        f"Total files: {ctx.get('total_files', 0)}",
        f"File types: {ctx.get('extension_counts', {})}",
    ]
    tabular = ctx.get("tabular_files", [])
    if tabular:
        lines.append(f"\nTabular files ({len(tabular)} found):")
        for tf in tabular:
            lines.append(f"  File: {tf['path']}  ({tf['row_count']} rows)")
            lines.append(f"  Columns ({len(tf['columns'])}): {', '.join(tf['columns'])}")
    else:
        lines.append("\nNo tabular (CSV/TSV) files found.")
    sample = ctx.get("sample_files", [])
    if sample:
        lines.append(f"\nSample file paths (showing up to {len(sample)}):")
        for p in sample[:20]:
            lines.append(f"  {p}")
    return "\n".join(lines)


def _plan_computation(
    task: str,
    dataset_context: dict[str, Any],
    chat_client: ChatClient,
    config: RuntimeConfig,
) -> str:
    """LLM call: what should the script compute and how? Returns natural language plan."""
    context_text = _format_dataset_context(dataset_context)
    model = config.llm.resolved_code_model if config.llm else ""
    temperature = config.llm.code_temperature if config.llm else 0.2
    # Planning needs only a fraction of the token budget
    max_tokens = min((config.llm.code_max_tokens if config.llm else 800), 800)

    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are a medical dataset analysis expert and Python programmer. "
                "Your job is to plan what a standalone Python script must compute to answer a user's query. "
                "Be specific: name the exact files to read, columns to use, and operations to perform. "
                "Output a numbered list of steps. Do not write any code yet."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"{context_text}\n\n"
                f'User query: "{task}"\n\n'
                "List the exact steps the script must take to answer this query. Be specific about:\n"
                "- Which file(s) to read (use the exact relative paths shown above)\n"
                "- Which columns to use (use the exact column names shown above)\n"
                "- What to compute (counts, distributions, groupby, filters, etc.)\n"
                "- What to print as the final output"
            ),
        ),
    ]
    request = ChatRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=False,
    )
    response = chat_client.chat(request)
    return response.content.strip()


def _generate_code(
    task: str,
    dataset_path: str,
    output_dir: str,
    dataset_context: dict[str, Any],
    plan: str,
    chat_client: ChatClient,
    config: RuntimeConfig,
    last_error: str = "",
    previous_code: str = "",
) -> str:
    """LLM call: synthesize a complete Python script. Returns raw Python source."""
    context_text = _format_dataset_context(dataset_context)
    model = config.llm.resolved_code_model if config.llm else ""
    temperature = config.llm.code_temperature if config.llm else 0.2
    max_tokens = config.llm.code_max_tokens if config.llm else 4000

    system_prompt = (
        "You are an expert Python programmer generating standalone analysis scripts for medical ECG datasets.\n\n"
        "Rules — follow every one exactly:\n"
        "- Output ONLY valid Python code. No markdown fences. No explanations before or after the code.\n"
        "- The script must run with: python script.py  (no arguments, no interactive input)\n"
        f"- First line after imports: DATASET_PATH = {dataset_path!r}\n"
        f"- Second constant: OUTPUT_DIR = Path({output_dir!r})\n"
        "- At the end of main(), print results to stdout as JSON:\n"
        "    import json; print(json.dumps(result, indent=2, default=str))\n"
        "- Also write the same result dict to OUTPUT_DIR / 'results.json'\n"
        "- Structure: def main(): ...  then: if __name__ == '__main__': main()\n"
        "- Allowed imports only: csv, json, math, statistics, pathlib, collections, datetime, numpy, pandas\n"
        "- Wrap the body of main() in try/except Exception as exc; on error print and write:\n"
        "    {'ok': False, 'error': str(exc)}\n"
        "- Read real data files — never produce placeholder or hardcoded output\n"
        "- Use pandas for tabular data if available; fall back to csv module otherwise"
    )

    if last_error and previous_code:
        user_content = (
            f'Task: "{task}"\n\n'
            f"The previous script failed with this error:\n{last_error}\n\n"
            f"Previous script:\n{previous_code}\n\n"
            f"Dataset structure:\n{context_text}\n\n"
            "Fix the script so it runs without errors and produces correct output. "
            "Output the complete corrected Python script only. No markdown."
        )
    else:
        user_content = (
            f'Task: "{task}"\n\n'
            f"Dataset structure:\n{context_text}\n\n"
            f"Computation plan:\n{plan}\n\n"
            "Write a complete Python script that executes this plan and answers the task. "
            "Output Python code only. No markdown fences."
        )

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_content),
    ]
    request = ChatRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=False,
    )
    response = chat_client.chat(request)
    return _extract_python_code(response.content)


def _extract_python_code(raw: str) -> str:
    """Strip markdown code fences if the LLM included them despite instructions."""
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw if raw.endswith("\n") else raw + "\n"

    lines = raw.splitlines()
    inner: list[str] = []
    in_block = False
    for i, line in enumerate(lines):
        if i == 0 and line.startswith("```"):
            in_block = True
            continue
        if in_block and line.strip() == "```":
            break
        if in_block:
            inner.append(line)

    if inner:
        return "\n".join(inner) + "\n"
    return raw + "\n"


def synthesize_script(
    task: str,
    dataset_path: str,
    output_dir: str,
    config: RuntimeConfig,
    chat_client: ChatClient,
    last_error: str = "",
    previous_code: str = "",
) -> str:
    """Return a complete, standalone Python script as a string.

    Pipeline:
    1. Discover dataset structure (filesystem probe, no LLM)
    2. Plan computation (LLM, free-text)
    3. Synthesize code (LLM, raw Python output)

    On repair (last_error + previous_code set), skips re-planning and asks the LLM
    to fix the broken code directly given the error message.
    """
    dataset_context = discover_dataset_structure(dataset_path)

    if last_error and previous_code:
        # Repair path: give LLM the actual broken code and real error message
        return _generate_code(
            task=task,
            dataset_path=dataset_path,
            output_dir=output_dir,
            dataset_context=dataset_context,
            plan="",
            chat_client=chat_client,
            config=config,
            last_error=last_error,
            previous_code=previous_code,
        )

    plan = _plan_computation(task, dataset_context, chat_client, config)
    return _generate_code(
        task=task,
        dataset_path=dataset_path,
        output_dir=output_dir,
        dataset_context=dataset_context,
        plan=plan,
        chat_client=chat_client,
        config=config,
    )
