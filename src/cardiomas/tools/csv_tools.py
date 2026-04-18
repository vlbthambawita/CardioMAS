"""CSV analysis tools.

Two tools:
- analyze_csv: column headings, dtypes, descriptive statistics, missing values, sample rows
- lookup_csv_headings: search documentation files in a dataset directory for column descriptions
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from cardiomas.schemas.tools import ToolResult

# Documentation file extensions to search
_DOC_EXTS = {".txt", ".md", ".rst", ".html", ".htm"}
_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB per file
_MAX_DOC_FILES = 40


# ── Tool 1: analyze_csv ───────────────────────────────────────────────────────

def analyze_csv(path: str, max_rows: int = 5) -> ToolResult:
    """Read a CSV (or TSV) file and return column headings, data types, descriptive
    statistics (count, mean, std, min, max, percentiles), missing-value counts,
    and sample rows. Works on large files by capping at 100 000 rows for statistics.

    Args:
        path: Absolute path to the CSV file.
        max_rows: Number of sample rows to include in the output (default 5).
    """
    p = Path(path)
    if not p.exists():
        return ToolResult(tool_name="analyze_csv", ok=False,
                          error=f"File not found: {path}",
                          summary=f"File not found: {path}")
    if not p.is_file():
        return ToolResult(tool_name="analyze_csv", ok=False,
                          error=f"Not a file: {path}",
                          summary=f"Path is a directory, not a CSV file: {path}")

    try:
        import pandas as pd

        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, low_memory=False, nrows=100_000)
        total_rows = len(df)
        total_cols = len(df.columns)

        # ── Per-column info ───────────────────────────────────────────────────
        columns_info: list[dict[str, Any]] = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            columns_info.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "missing": missing,
                "missing_pct": round(missing / total_rows * 100, 1) if total_rows > 0 else 0.0,
            })

        # ── Statistics (first 60 columns to keep output bounded) ──────────────
        stats: dict[str, dict[str, Any]] = {}
        for col in list(df.columns)[:60]:
            if df[col].dtype.kind in "iuf":  # integer, unsigned, float
                s = df[col].describe()
                stats[col] = {
                    "type": "numeric",
                    "count": int(s.get("count", 0)),
                    "mean": _r(s.get("mean")),
                    "std": _r(s.get("std")),
                    "min": _r(s.get("min")),
                    "25%": _r(s.get("25%")),
                    "50%": _r(s.get("50%")),
                    "75%": _r(s.get("75%")),
                    "max": _r(s.get("max")),
                }
            else:
                vc = df[col].value_counts()
                stats[col] = {
                    "type": "categorical",
                    "count": int(df[col].count()),
                    "unique": int(df[col].nunique()),
                    "top": str(vc.index[0]) if len(vc) else "",
                    "top_freq": int(vc.iloc[0]) if len(vc) else 0,
                    "sample_values": [str(v) for v in vc.index[:5].tolist()],
                }

        # ── Sample rows ───────────────────────────────────────────────────────
        sample_rows = [
            {k: (None if isinstance(v, float) and math.isnan(v) else v)
             for k, v in row.items()}
            for row in df.head(max(1, max_rows)).to_dict(orient="records")
        ]

        data: dict[str, Any] = {
            "file": str(p),
            "file_size_bytes": p.stat().st_size,
            "total_rows": total_rows,
            "total_cols": total_cols,
            "columns": columns_info,
            "statistics": stats,
            "sample_rows": sample_rows,
        }

        # ── Human-readable summary (used as evidence by the responder) ─────────
        lines = [
            f"CSV Analysis: {p.name}",
            f"File size: {p.stat().st_size // 1024} KB  |  Rows: {total_rows}  |  Columns: {total_cols}",
            "",
            "Column headings and types:",
        ]
        for ci in columns_info:
            miss = f"  ← {ci['missing_pct']}% missing" if ci["missing"] else ""
            lines.append(f"  {ci['name']}  [{ci['dtype']}]{miss}")

        lines += ["", f"Statistics (first {min(60, total_cols)} columns):"]
        for col, s in list(stats.items())[:30]:
            if s["type"] == "numeric":
                lines.append(
                    f"  {col}: mean={s['mean']}, std={s['std']}, "
                    f"min={s['min']}, max={s['max']}, missing={_missing_pct(columns_info, col)}%"
                )
            else:
                lines.append(
                    f"  {col}: {s['unique']} unique values, "
                    f"top='{s['top']}' ({s['top_freq']}×), "
                    f"sample={s['sample_values']}"
                )
        if total_cols > 30:
            lines.append(f"  ... ({total_cols - 30} more columns available in data)")

        lines += ["", f"Sample rows (first {min(max_rows, total_rows)}):"]
        for i, row in enumerate(sample_rows):
            items = list(row.items())
            preview = ", ".join(f"{k}={v}" for k, v in items[:8])
            if len(items) > 8:
                preview += f", … (+{len(items) - 8} more)"
            lines.append(f"  [{i + 1}] {preview}")

        all_cols = ", ".join(list(df.columns)[:15])
        if total_cols > 15:
            all_cols += f", … (+{total_cols - 15} more)"
        brief = (
            f"CSV '{p.name}': {total_rows} rows × {total_cols} columns. "
            f"Columns: {all_cols}."
        )

        return ToolResult(
            tool_name="analyze_csv",
            ok=True,
            summary="\n".join(lines),
            data=data,
        )

    except ImportError:
        return ToolResult(
            tool_name="analyze_csv", ok=False,
            error="pandas is required for analyze_csv but is not installed.",
            summary="pandas not available.",
        )
    except Exception as exc:
        return ToolResult(
            tool_name="analyze_csv", ok=False,
            error=str(exc),
            summary=f"Failed to analyze CSV '{Path(path).name}': {exc}",
        )


# ── Tool 2: lookup_csv_headings ───────────────────────────────────────────────

def lookup_csv_headings(path: str, headings: str) -> ToolResult:
    """Search documentation files inside a dataset directory to find the meaning
    and description of CSV column headings. Looks through README files, data
    dictionaries, codebooks, and any plain-text or Markdown files for mentions
    of each heading and returns surrounding context lines.

    Args:
        path: Absolute path to the dataset directory (or any subdirectory) to search.
        headings: Comma-separated column names to look up,
                  e.g. ``"age,sex,label,ecg_id,strat_fold"``.
    """
    p = Path(path)
    if not p.exists():
        return ToolResult(tool_name="lookup_csv_headings", ok=False,
                          error=f"Path not found: {path}",
                          summary=f"Path not found: {path}")

    search_dir = p if p.is_dir() else p.parent

    # Parse heading list
    heading_list = [h.strip() for h in headings.replace(";", ",").split(",") if h.strip()]
    if not heading_list:
        return ToolResult(tool_name="lookup_csv_headings", ok=False,
                          error="No headings provided.",
                          summary="No headings to look up — provide a comma-separated list.")

    doc_files = _find_doc_files(search_dir)
    if not doc_files:
        return ToolResult(
            tool_name="lookup_csv_headings", ok=True,
            summary=f"No documentation files found under '{search_dir}'. "
                    "Try a parent directory or check that README/txt files exist.",
            data={"headings": heading_list, "results": {}, "files_searched": []},
        )

    results: dict[str, list[dict[str, Any]]] = {h: [] for h in heading_list}
    files_searched: list[str] = []

    for doc_path in doc_files[:_MAX_DOC_FILES]:
        try:
            if doc_path.stat().st_size > _MAX_FILE_SIZE:
                continue
            text = doc_path.read_text(encoding="utf-8", errors="replace")
            file_lines = text.splitlines()
            rel = str(doc_path.relative_to(search_dir))
            files_searched.append(rel)
            for heading in heading_list:
                hits = _context_for_heading(file_lines, heading, rel)
                results[heading].extend(hits)
        except Exception:
            continue

    # Build readable output
    lines = [
        f"Searched {len(files_searched)} documentation file(s) under '{search_dir.name}':",
        "",
    ]
    for heading in heading_list:
        matches = results[heading][:4]  # cap at 4 matches per heading
        lines.append(f"### {heading}")
        if matches:
            for m in matches:
                lines.append(f"  Source: {m['file']}  (line {m['line']})")
                for ctx in m["context"]:
                    lines.append(f"    {ctx}")
                lines.append("")
        else:
            lines.append("  No description found in documentation files.")
            lines.append("")

    found = [h for h in heading_list if results[h]]
    summary_line = (
        f"Found descriptions for {len(found)}/{len(heading_list)} heading(s) "
        f"across {len(files_searched)} file(s)."
    )
    lines.insert(0, summary_line)
    lines.insert(1, "")

    return ToolResult(
        tool_name="lookup_csv_headings",
        ok=True,
        summary="\n".join(lines),
        data={
            "headings": heading_list,
            "results": {h: results[h][:4] for h in heading_list},
            "files_searched": files_searched,
        },
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_doc_files(directory: Path) -> list[Path]:
    """Return documentation files ordered by relevance (README/codebook first)."""
    priority_stems = {
        "readme", "data_dictionary", "codebook", "variables",
        "description", "labels", "fieldnames", "columns", "metadata",
        "header", "features", "schema",
    }
    priority: list[Path] = []
    others: list[Path] = []
    try:
        for p in sorted(directory.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in _DOC_EXTS:
                continue
            if p.stat().st_size > _MAX_FILE_SIZE:
                continue
            if any(kw in p.stem.lower() for kw in priority_stems):
                priority.append(p)
            else:
                others.append(p)
    except PermissionError:
        pass
    return priority + others


def _context_for_heading(
    lines: list[str],
    heading: str,
    rel_path: str,
    ctx: int = 3,
) -> list[dict[str, Any]]:
    """Return up to 5 matches with ±ctx surrounding lines."""
    heading_lower = heading.lower()
    matches: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        if heading_lower not in line.lower():
            continue
        start = max(0, i - ctx)
        end = min(len(lines), i + ctx + 1)
        context = [ln.rstrip() for ln in lines[start:end] if ln.strip()]
        matches.append({"file": rel_path, "line": i + 1, "context": context[:8]})
        if len(matches) >= 5:
            break
    return matches


def _r(v: Any) -> float:
    """Round to 4 decimal places, handling NaN/None safely."""
    try:
        f = float(v)
        return round(f, 4) if not math.isnan(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _missing_pct(columns_info: list[dict], col: str) -> float:
    for ci in columns_info:
        if ci["name"] == col:
            return ci["missing_pct"]
    return 0.0
