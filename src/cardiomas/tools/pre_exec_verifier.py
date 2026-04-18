"""Pre-execution argument validation for registered tools.

Catches the most common LLM argument errors before the tool runs:
- Missing or non-existent path for filesystem tools
- Empty task/description for code-generation tools

Returns a clear, actionable error message the agent can learn from,
rather than a raw Python exception traceback buried in the observation.
"""
from __future__ import annotations

from pathlib import Path

_PATH_TOOLS = {"list_folder_structure", "read_wfdb_dataset", "inspect_dataset", "lookup_csv_headings"}
_FILE_TOOLS = {"analyze_csv"}
_CODE_TOOLS = {"generate_python_artifact", "generate_shell_artifact"}


def verify_tool_args(name: str, args: dict) -> tuple[bool, str]:
    """Return ``(ok, error_message)``.

    ``ok=True`` means the args look valid — proceed with execution.
    ``ok=False`` means there is a clear argument problem — skip execution and
    return the error message as the observation so the agent can self-correct.
    """
    if name in _PATH_TOOLS:
        path = str(
            args.get("path") or args.get("dataset_path") or args.get("directory") or ""
        ).strip()
        if not path:
            return False, (
                f"'{name}' requires a 'path' argument (absolute directory path). "
                "Provide args={{\"path\": \"<absolute_path_to_dataset>\"}}."
            )
        p = Path(path)
        if not p.exists():
            return False, (
                f"Path does not exist: '{path}'. "
                "Use list_folder_structure on the parent directory to find the correct path."
            )
        if not p.is_dir():
            return False, f"'{path}' is a file, not a directory. Provide the parent directory."

    if name in _FILE_TOOLS:
        path = str(
            args.get("path") or args.get("file") or args.get("csv_path") or ""
        ).strip()
        if not path:
            return False, (
                f"'{name}' requires a 'path' argument (absolute path to the CSV file). "
                "Provide args={{\"path\": \"<absolute_path_to_csv>\"}}."
            )
        p = Path(path)
        if not p.exists():
            return False, (
                f"File does not exist: '{path}'. "
                "Use list_folder_structure on the dataset directory to find available CSV files."
            )
        if p.is_dir():
            return False, (
                f"'{path}' is a directory. "
                "Provide the path to a specific CSV file, not a directory."
            )

    if name in _CODE_TOOLS:
        task = str(
            args.get("task") or args.get("code") or
            args.get("description") or args.get("prompt") or ""
        ).strip()
        if not task:
            return False, (
                f"'{name}' requires a 'task' argument describing what to compute. "
                "Provide args={{\"task\": \"<description>\", \"path\": \"<dataset_path>\"}}."
            )

    return True, ""
