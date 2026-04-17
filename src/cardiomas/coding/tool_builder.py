from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import textwrap
from typing import Any

from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.tools import ToolSpec


@dataclass(slots=True)
class GeneratedArtifactPackage:
    artifact_slug: str
    spec: ToolSpec
    code: str
    readme: str
    prompt: dict[str, Any]
    context: dict[str, Any]
    kind: str = "python"


def build_generated_tool_package(
    tool_name: str,
    payload: dict[str, Any],
    config: RuntimeConfig,
    last_error: str = "",
) -> GeneratedArtifactPackage:
    if tool_name != "generate_python_artifact":
        raise ValueError(f"Unsupported generated tool: {tool_name}")

    task = str(payload.get("task") or "").strip()
    dataset_path = str(payload.get("dataset_path") or "").strip()
    target_path = str(payload.get("target_path") or "").strip()
    artifact_name = str(payload.get("artifact_name") or "").strip()
    mode = _infer_mode(task, target_path)
    context = _build_context(dataset_path, target_path, mode)
    prompt = {
        "tool_name": tool_name,
        "task": task,
        "dataset_path": dataset_path,
        "target_path": target_path,
        "mode": mode,
        "last_error": last_error,
        "execution_contract": {
            "entrypoint": "run(payload)",
            "writes_allowed": ["payload['artifact_output_dir']"],
            "reads_allowed": ["payload['dataset_path']", "payload['target_path']"],
            "response_keys": ["ok", "summary", "data", "evidence"],
        },
    }
    artifact_slug = artifact_name or _artifact_slug(task, dataset_path, target_path, mode)
    spec = ToolSpec(
        name="generate_python_artifact",
        description="Generate, verify, and execute a query-specific Python artifact for local dataset reading or analysis.",
        category="autonomy",
        generated=True,
    )
    readme = _build_readme(artifact_slug, task, mode)
    code = _render_python_tool_source(
        default_task=task,
        default_dataset_path=dataset_path,
        default_target_path=target_path,
        default_mode=mode,
    )
    return GeneratedArtifactPackage(
        artifact_slug=artifact_slug,
        spec=spec,
        code=code,
        readme=readme,
        prompt=prompt,
        context=context,
    )


def _infer_mode(task: str, target_path: str) -> str:
    lower = task.lower()
    if target_path or any(token in lower for token in ["read ", "open ", "show ", "preview", "header", "field", "explain"]):
        return "read_file"
    if any(
        token in lower
        for token in [
            "stat",
            "count",
            "distribution",
            "label",
            "class",
            "column",
            "metadata",
            "summary",
            "missing",
            "analy",
            "measure",
            "rows",
        ]
    ):
        return "analyze_dataset"
    return "inspect_directory"


def _build_context(dataset_path: str, target_path: str, mode: str) -> dict[str, Any]:
    root = Path(dataset_path).expanduser() if dataset_path else None
    sample_files: list[str] = []
    tabular_files: list[str] = []
    target_exists = False
    if root is not None and root.exists() and root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(root))
            if len(sample_files) < 20:
                sample_files.append(relative)
            if path.suffix.lower() in {".csv", ".tsv", ".json"} and len(tabular_files) < 10:
                tabular_files.append(relative)
        if target_path:
            candidate = Path(target_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            target_exists = candidate.exists()
    return {
        "mode": mode,
        "dataset_path": dataset_path,
        "target_path": target_path,
        "target_exists": target_exists,
        "sample_files": sample_files,
        "tabular_files": tabular_files,
    }


def _artifact_slug(task: str, dataset_path: str, target_path: str, mode: str) -> str:
    base = _slug(task or target_path or mode or "artifact")
    digest = hashlib.sha1(f"{dataset_path}|{target_path}|{task}|{mode}".encode("utf-8")).hexdigest()[:8]
    return f"{base[:36] or 'artifact'}-{digest}"


def _build_readme(artifact_slug: str, task: str, mode: str) -> str:
    return (
        f"# {artifact_slug}\n\n"
        f"- Mode: `{mode}`\n"
        f"- Task: {task or '(none provided)'}\n"
        "- Contract: the generated module exposes `run(payload)` and writes structured outputs under `outputs/`.\n"
    )


def _render_python_tool_source(
    default_task: str,
    default_dataset_path: str,
    default_target_path: str,
    default_mode: str,
) -> str:
    task_literal = json.dumps(default_task)
    dataset_literal = json.dumps(default_dataset_path)
    target_literal = json.dumps(default_target_path)
    mode_literal = json.dumps(default_mode)
    source = f'''
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_TASK = {task_literal}
DEFAULT_DATASET_PATH = {dataset_literal}
DEFAULT_TARGET_PATH = {target_literal}
DEFAULT_MODE = {mode_literal}
TEXT_SUFFIXES = {{".txt", ".md", ".hea", ".log", ".cfg", ".yaml", ".yml", ".html"}}
TABULAR_SUFFIXES = {{".csv", ".tsv"}}
JSON_SUFFIXES = {{".json"}}
SKIP_FILENAMES = {{"license.txt", "sha256sums.txt"}}


def run(payload):
    task = str(payload.get("task") or DEFAULT_TASK)
    dataset_path = str(payload.get("dataset_path") or DEFAULT_DATASET_PATH)
    target_path = str(payload.get("target_path") or DEFAULT_TARGET_PATH)
    mode = str(payload.get("mode") or DEFAULT_MODE)
    artifact_slug = str(payload.get("artifact_slug") or "generated-python-artifact")
    output_dir = Path(payload.get("artifact_output_dir") or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path:
        return _finalize(
            output_dir,
            artifact_slug,
            False,
            {{
                "mode": mode,
                "summary": "No dataset path was provided to the generated analysis artifact.",
                "error": "dataset_path is required",
                "task": task,
            }},
        )

    root = Path(dataset_path).expanduser()
    if not root.exists():
        return _finalize(
            output_dir,
            artifact_slug,
            False,
            {{
                "mode": mode,
                "summary": f"Dataset path not found: {{root}}",
                "error": f"Dataset path not found: {{root}}",
                "task": task,
                "dataset_path": str(root),
            }},
        )

    try:
        if target_path or mode == "read_file" or _query_requests_file(task):
            selected = _select_target_file(root, target_path, task)
            if selected is None:
                result = _summarize_directory(root)
                result["summary"] = "No matching file was found; returning a directory summary instead."
            else:
                result = _summarize_file(selected, root)
        elif mode == "inspect_directory":
            result = _summarize_directory(root)
        else:
            result = _summarize_dataset(root, task)
    except Exception as exc:
        return _finalize(
            output_dir,
            artifact_slug,
            False,
            {{
                "mode": mode,
                "summary": f"Generated analysis artifact failed: {{exc}}",
                "error": str(exc),
                "task": task,
                "dataset_path": str(root),
            }},
        )

    result["task"] = task
    result["dataset_path"] = str(root)
    return _finalize(output_dir, artifact_slug, True, result)


def _finalize(output_dir, artifact_slug, ok, result):
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    evidence = [_build_evidence(result, result_path, artifact_slug)]
    return {{
        "ok": ok,
        "summary": result.get("summary", ""),
        "data": dict(result, result_path=str(result_path)),
        "evidence": evidence,
        "error": result.get("error", ""),
    }}


def _build_evidence(result, result_path, artifact_slug):
    summary = result.get("summary", "")
    details = []
    if result.get("selected_path"):
        details.append(f"selected_path={{result['selected_path']}}")
    if result.get("columns"):
        details.append("columns=" + ", ".join(result["columns"][:12]))
    if result.get("preview_lines"):
        details.append("preview=" + " | ".join(result["preview_lines"][:5]))
    if result.get("sample_files"):
        details.append("sample_files=" + ", ".join(result["sample_files"][:8]))
    content = summary
    if details:
        content = summary + "\\n" + "\\n".join(details)
    return {{
        "chunk_id": f"generated-artifact:{{artifact_slug}}",
        "source_id": "autonomy",
        "source_label": "generated-python-artifact",
        "source_type": "generated_artifact",
        "title": result.get("mode", "generated_artifact"),
        "content": content.strip(),
        "uri": str(result_path),
        "metadata": {{"chunk_label": artifact_slug, "artifact_slug": artifact_slug}},
        "score": 1.0,
    }}


def _summarize_dataset(root, task):
    directory = _summarize_directory(root)
    files = _collect_files(root)
    tabular_files = [path for path in files if path.suffix.lower() in TABULAR_SUFFIXES]
    if tabular_files:
        selected = _best_scoring_file(tabular_files, root, task)
        table = _analyze_table(selected, root)
        directory.update(table)
        directory["mode"] = "analyze_dataset"
        directory["summary"] = (
            f"Generated analysis artifact examined `{{table['selected_path']}}` with "
            f"{{table.get('total_rows', 0)}} row(s) and {{len(table.get('columns', []))}} column(s); "
            f"the dataset root has {{directory.get('total_files', 0)}} file(s)."
        )
        return directory
    selected = _select_target_file(root, "", task)
    if selected is not None:
        result = _summarize_file(selected, root)
        result["mode"] = "analyze_dataset"
        return result
    directory["mode"] = "analyze_dataset"
    directory["summary"] = "Generated analysis artifact inspected the dataset directory but did not find a readable file."
    return directory


def _summarize_directory(root):
    files = _collect_files(root)
    extension_counts = Counter((path.suffix.lower() or "<no_extension>") for path in files)
    sample_files = [str(path.relative_to(root)) for path in files[:20]]
    return {{
        "mode": "inspect_directory",
        "summary": (
            f"Directory scan of `{{root.name or root}}` found {{len(files)}} file(s) across "
            f"{{len(extension_counts)}} extension group(s)."
        ),
        "total_files": len(files),
        "extension_counts": dict(sorted(extension_counts.items())),
        "sample_files": sample_files,
    }}


def _summarize_file(path, root):
    suffix = path.suffix.lower()
    if suffix in TABULAR_SUFFIXES:
        result = _analyze_table(path, root)
        result["mode"] = "read_file"
        return result
    if suffix in JSON_SUFFIXES:
        return _analyze_json(path, root)
    return _analyze_text(path, root)


def _analyze_table(path, root):
    delimiter = "\\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = list(reader.fieldnames or [])
        sample_rows = []
        total_rows = 0
        missing_cells = 0
        candidate_counts = {{}}
        numeric_values = {{}}

        for row in reader:
            total_rows += 1
            normalized = {{}}
            for column in columns:
                value = str(row.get(column, "") or "").strip()
                normalized[column] = value
                if not value:
                    missing_cells += 1
                    continue
                lowered = column.lower()
                if lowered in {{"label", "labels", "class", "classes", "target", "diagnosis", "split", "fold"}}:
                    counter = candidate_counts.setdefault(column, Counter())
                    if len(counter) < 32 or value in counter:
                        counter[value] += 1
                number = _to_float(value)
                if number is not None:
                    numeric_values.setdefault(column, []).append(number)
            if len(sample_rows) < 5:
                sample_rows.append(normalized)

    numeric_summary = {{}}
    for column, values in numeric_values.items():
        if not values:
            continue
        numeric_summary[column] = {{
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "mean": round(sum(values) / len(values), 4),
        }}
        if len(numeric_summary) >= 3:
            break

    relative_path = str(path.relative_to(root))
    missing_fraction = 0.0
    if total_rows and columns:
        missing_fraction = round(missing_cells / float(total_rows * len(columns)), 4)
    summary = (
        f"Tabular analysis for `{{relative_path}}` found {{total_rows}} row(s), "
        f"{{len(columns)}} column(s), missing_fraction={{missing_fraction}}."
    )
    if candidate_counts:
        first_column = next(iter(candidate_counts))
        summary += f" Candidate counts from `{{first_column}}`: {{dict(candidate_counts[first_column])}}."
    return {{
        "mode": "analyze_dataset",
        "summary": summary,
        "selected_path": relative_path,
        "columns": columns,
        "total_rows": total_rows,
        "missing_fraction": missing_fraction,
        "sample_rows": sample_rows,
        "class_counts": {{
            "column": next(iter(candidate_counts), ""),
            "counts": dict(candidate_counts[next(iter(candidate_counts))]) if candidate_counts else {{}},
        }},
        "numeric_summary": numeric_summary,
    }}


def _analyze_json(path, root):
    payload = json.loads(path.read_text(encoding="utf-8"))
    relative_path = str(path.relative_to(root))
    if isinstance(payload, list):
        sample = payload[:3]
        keys = sorted(sample[0].keys()) if sample and isinstance(sample[0], dict) else []
        summary = f"JSON analysis for `{{relative_path}}` found a list with {{len(payload)}} item(s)."
        return {{
            "mode": "read_file",
            "summary": summary,
            "selected_path": relative_path,
            "json_type": "list",
            "total_rows": len(payload),
            "columns": keys,
            "sample_rows": sample,
        }}
    if isinstance(payload, dict):
        keys = sorted(payload.keys())
        summary = f"JSON analysis for `{{relative_path}}` found an object with {{len(keys)}} key(s)."
        return {{
            "mode": "read_file",
            "summary": summary,
            "selected_path": relative_path,
            "json_type": "object",
            "columns": keys,
            "sample_rows": [{{key: payload[key] for key in keys[:10]}}],
        }}
    return {{
        "mode": "read_file",
        "summary": f"JSON analysis for `{{relative_path}}` found a scalar value.",
        "selected_path": relative_path,
        "json_type": type(payload).__name__,
        "sample_rows": [payload],
    }}


def _analyze_text(path, root):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    preview_lines = lines[:40]
    key_value_fields = []
    for line in preview_lines:
        if ":" not in line:
            continue
        field_name = line.split(":", 1)[0].strip()
        if field_name and field_name not in key_value_fields:
            key_value_fields.append(field_name)
        if len(key_value_fields) >= 12:
            break
    relative_path = str(path.relative_to(root))
    summary = f"Text preview for `{{relative_path}}` captured {{len(preview_lines)}} line(s)."
    return {{
        "mode": "read_file",
        "summary": summary,
        "selected_path": relative_path,
        "preview_lines": preview_lines,
        "detected_fields": key_value_fields,
        "line_count": len(lines),
    }}


def _collect_files(root):
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.lower() not in SKIP_FILENAMES:
            files.append(path)
    return files


def _select_target_file(root, target_path, task):
    if target_path:
        candidate = Path(target_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists() and candidate.is_file():
            return candidate
    files = _collect_files(root)
    if not files:
        return None
    return _best_scoring_file(files, root, task)


def _best_scoring_file(files, root, task):
    tokens = set(_tokenize(task))
    ranked = sorted(files, key=lambda path: (-_score_file(path, root, tokens), str(path)))
    best = ranked[0]
    if _score_file(best, root, tokens) <= 0 and files:
        preferred = [path for path in files if path.suffix.lower() in TABULAR_SUFFIXES | JSON_SUFFIXES | TEXT_SUFFIXES]
        return preferred[0] if preferred else files[0]
    return best


def _score_file(path, root, tokens):
    relative = str(path.relative_to(root)).lower()
    score = 0
    for token in tokens:
        if token and token in relative:
            score += 3
    suffix = path.suffix.lower()
    if suffix in TABULAR_SUFFIXES:
        score += 4
    elif suffix in JSON_SUFFIXES:
        score += 3
    elif suffix in TEXT_SUFFIXES:
        score += 2
    if path.name.lower() in SKIP_FILENAMES:
        score -= 10
    return score


def _query_requests_file(task):
    lower = task.lower()
    return any(token in lower for token in ["read ", "open ", "preview", "header", "field", "file", "show "])


def _tokenize(text):
    tokens = []
    current = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
            continue
        if len(current) >= 3:
            tokens.append("".join(current))
        current = []
    if len(current) >= 3:
        tokens.append("".join(current))
    return tokens


def _to_float(value):
    try:
        return float(value)
    except ValueError:
        return None
'''
    return textwrap.dedent(source).strip() + "\n"


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "artifact"
