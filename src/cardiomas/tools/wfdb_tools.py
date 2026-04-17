"""WFDB ECG dataset inspection tool.

Reads PhysioNet WFDB records from a directory without executing signal data
(which can be gigabytes). Header files (.hea) are plain text and are parsed
directly. Annotation files and signal-file presence are detected from the
filesystem. The ``wfdb`` Python library is used opportunistically for richer
metadata — if it is unavailable or incompatible, the tool falls back to its
own parser so it never fails silently.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cardiomas.schemas.tools import ToolResult

logger = logging.getLogger(__name__)

# Maximum number of records to inspect in detail (avoids O(n) wall-time on
# large PhysioNet datasets with tens of thousands of recordings).
_MAX_DETAIL_RECORDS = 20

# Known WFDB annotation file extensions (from the WFDB spec).
_ANNOTATION_EXTENSIONS = {
    ".atr", ".ann", ".ecg", ".ari", ".bwr", ".cba", ".cpc", ".man",
    ".marker", ".nsr", ".pu", ".pu0", ".pu1", ".q1c", ".q2c", ".q3c",
    ".qrs", ".qt1", ".qt2", ".qti", ".r", ".rhy", ".stu", ".stf",
    ".trigger", ".vs",
}


def read_wfdb_dataset(dataset_path: str, max_records: int = _MAX_DETAIL_RECORDS) -> ToolResult:
    """Inspect a WFDB-format ECG dataset directory and return structured metadata.

    WFDB (WaveForm DataBase) is the standard file format used by PhysioNet
    (e.g., PTB-XL, MIT-BIH, CPSC) and many other ECG databases. A WFDB record
    consists of:
    - A header file (``<record>.hea``) — plain text with sampling frequency,
      number of signals, signal names, units, ADC gain, and duration.
    - One or more signal files (``<record>.dat``, ``.edf``, ``.mat``, etc.) —
      binary files containing the raw ECG waveforms.
    - Optional annotation files (``<record>.atr``, ``.ecg``, ``.ann``, etc.) —
      binary files storing beat labels, rhythm annotations, and QRS locations.

    This tool reads every ``.hea`` header file it finds (up to *max_records*
    for detailed parsing), extracts signal metadata without loading raw signal
    data, detects annotation file types, and returns a compact but complete
    summary that lets the agent understand which leads are recorded, at what
    sampling rate, how long each recording is, what annotation types are
    present, and how many records the dataset contains.

    Use this tool when:
    - The dataset contains ``.hea`` files and you need to know signal names,
      leads, or sampling frequency before writing analysis code.
    - You want to confirm which annotation extensions are present (e.g. whether
      rhythm labels or beat labels are available).
    - You need the total record count, duration distribution, or unit labels
      for downstream computation.

    Args:
        dataset_path: Path to the root directory of the WFDB dataset. May be a
            flat directory (all ``.hea`` files at the top level) or a nested
            one (one sub-directory per record, as in PTB-XL).
        max_records: Maximum number of records to parse in detail. The total
            count is always accurate; only metadata aggregation is capped.

    Returns:
        ToolResult whose ``summary`` is a human-readable report and whose
        ``data`` dict contains:
        - ``total_records`` — number of ``.hea`` files found
        - ``records_inspected`` — how many were parsed for metadata
        - ``signal_names`` — sorted list of all unique lead/signal names seen
        - ``sampling_frequencies`` — Counter of fs values (Hz)
        - ``signal_counts`` — Counter of number-of-signals per record
        - ``duration_seconds`` — dict with min/max/mean duration in seconds
        - ``units`` — sorted list of all unique unit strings seen
        - ``annotation_types`` — sorted list of annotation extensions found
        - ``signal_file_formats`` — sorted list of signal file extensions found
        - ``sample_records`` — list of per-record dicts for the first few records
        - ``wfdb_library_used`` — whether the wfdb Python library was used
    """
    root = Path(dataset_path)
    if not root.exists():
        return ToolResult(
            tool_name="read_wfdb_dataset",
            ok=False,
            summary="",
            error=f"Path not found: {dataset_path}",
        )
    if not root.is_dir():
        return ToolResult(
            tool_name="read_wfdb_dataset",
            ok=False,
            summary="",
            error=f"Path is not a directory: {dataset_path}",
        )

    hea_files = sorted(root.rglob("*.hea"))
    if not hea_files:
        return ToolResult(
            tool_name="read_wfdb_dataset",
            ok=False,
            summary="",
            error=(
                f"No WFDB header files (.hea) found under {dataset_path}. "
                "This directory does not appear to contain a WFDB dataset."
            ),
        )

    total_records = len(hea_files)
    detail_files = hea_files[:max_records]

    # Detect annotation and signal file extensions present in the dataset
    annotation_types: set[str] = set()
    signal_file_formats: set[str] = set()
    for f in root.rglob("*"):
        if f.is_file() and not f.suffix == ".hea":
            ext = f.suffix.lower()
            if ext in _ANNOTATION_EXTENSIONS:
                annotation_types.add(ext)
            elif ext in {".dat", ".edf", ".mat", ".bin", ".d16", ".d12"}:
                signal_file_formats.add(ext)

    # Try wfdb library — catch both ImportError and any runtime error from the
    # known pandas 3.x incompatibility in wfdb 4.3.x.
    wfdb_available = False
    wfdb_mod = None
    try:
        import wfdb as _wfdb  # type: ignore[import-untyped]
        wfdb_mod = _wfdb
        wfdb_available = True
    except Exception:
        pass

    all_sig_names: list[str] = []
    all_units: list[str] = []
    fs_counter: Counter[float] = Counter()
    nsig_counter: Counter[int] = Counter()
    durations: list[float] = []
    sample_records: list[dict[str, Any]] = []
    parse_errors = 0

    for hea in detail_files:
        record_stem = str(hea.with_suffix(""))
        try:
            if wfdb_mod is not None:
                meta = _read_header_wfdb(wfdb_mod, record_stem)
            else:
                meta = _read_header_manual(hea)

            all_sig_names.extend(meta.get("sig_name", []))
            all_units.extend(u for u in meta.get("units", []) if u)
            fs = meta.get("fs")
            if fs:
                fs_counter[float(fs)] += 1
            nsig = meta.get("n_sig")
            if nsig:
                nsig_counter[int(nsig)] += 1
            sig_len = meta.get("sig_len")
            if fs and sig_len:
                durations.append(sig_len / fs)

            sample_records.append({
                "record": hea.stem,
                "path": str(hea.relative_to(root)),
                **meta,
            })
        except Exception as exc:
            logger.debug("Failed to parse %s: %s", hea, exc)
            parse_errors += 1

    # Aggregate
    unique_sig_names = sorted(set(all_sig_names))
    unique_units = sorted(set(all_units))
    duration_stats: dict[str, float] = {}
    if durations:
        duration_stats = {
            "min": round(min(durations), 2),
            "max": round(max(durations), 2),
            "mean": round(sum(durations) / len(durations), 2),
        }

    # ── Build human-readable summary ──────────────────────────────────────────
    lines: list[str] = [
        f"WFDB dataset at: {root}",
        f"Total records (.hea files): {total_records}",
        f"Records inspected for metadata: {len(detail_files)}"
        + (f" (of {total_records})" if total_records > max_records else ""),
    ]

    if parse_errors:
        lines.append(f"Header parse errors: {parse_errors}")

    if fs_counter:
        fs_str = ", ".join(f"{int(fs)} Hz × {n}" for fs, n in sorted(fs_counter.items()))
        lines.append(f"Sampling frequencies: {fs_str}")

    if nsig_counter:
        nsig_str = ", ".join(f"{n} signal(s) × {c}" for n, c in sorted(nsig_counter.items()))
        lines.append(f"Signals per record: {nsig_str}")

    if unique_sig_names:
        leads = ", ".join(unique_sig_names)
        lines.append(f"Signal/lead names: {leads}")

    if unique_units:
        lines.append(f"Units: {', '.join(unique_units)}")

    if duration_stats:
        lines.append(
            f"Duration — min: {duration_stats['min']}s, "
            f"max: {duration_stats['max']}s, "
            f"mean: {duration_stats['mean']}s"
        )

    if annotation_types:
        lines.append(f"Annotation file types: {', '.join(sorted(annotation_types))}")
    else:
        lines.append("Annotation files: none found")

    if signal_file_formats:
        lines.append(f"Signal file formats: {', '.join(sorted(signal_file_formats))}")

    lines.append(f"WFDB Python library used: {wfdb_available}")

    if sample_records:
        lines.append(f"\nFirst {min(3, len(sample_records))} record(s):")
        for rec in sample_records[:3]:
            fs_val = rec.get("fs", "?")
            n_sig = rec.get("n_sig", "?")
            names = ", ".join(rec.get("sig_name", [])) or "?"
            sig_len = rec.get("sig_len")
            dur = f"{sig_len / fs_val:.1f}s" if sig_len and fs_val and fs_val != "?" else "?"
            lines.append(f"  {rec['record']}: {n_sig} signals @ {fs_val} Hz, {dur}, leads=[{names}]")

    summary = "\n".join(lines)

    return ToolResult(
        tool_name="read_wfdb_dataset",
        ok=True,
        summary=summary,
        data={
            "dataset_path": str(root),
            "total_records": total_records,
            "records_inspected": len(detail_files),
            "signal_names": unique_sig_names,
            "sampling_frequencies": dict(fs_counter),
            "signal_counts": dict(nsig_counter),
            "duration_seconds": duration_stats,
            "units": unique_units,
            "annotation_types": sorted(annotation_types),
            "signal_file_formats": sorted(signal_file_formats),
            "sample_records": sample_records,
            "wfdb_library_used": wfdb_available,
        },
    )


# ── Header parsers ─────────────────────────────────────────────────────────────

def _read_header_wfdb(wfdb_mod: Any, record_stem: str) -> dict[str, Any]:
    """Read a WFDB header using the wfdb Python library."""
    rec = wfdb_mod.rdheader(record_stem)
    return {
        "fs": getattr(rec, "fs", None),
        "n_sig": getattr(rec, "n_sig", None),
        "sig_len": getattr(rec, "sig_len", None),
        "sig_name": list(getattr(rec, "sig_name", []) or []),
        "units": list(getattr(rec, "units", []) or []),
        "adc_gain": list(getattr(rec, "adc_gain", []) or []),
        "comments": list(getattr(rec, "comments", []) or []),
    }


def _read_header_manual(hea_path: Path) -> dict[str, Any]:
    """Parse a WFDB header file without the wfdb library.

    WFDB header format (first non-comment line):
        <record> <n_sig> <fs>[/<cfs>(<base_counter>)] [<sig_len> [<base_time> [<base_date>]]]

    Each subsequent signal line:
        <filename> <fmt>[x<spf>][:<skew>][+<byte_offset>] <gain>[/<baseline>]<(units)> \
        <adc_res> <adc_zero> <init_val> <checksum> <block_size> <description...>
    """
    lines = hea_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Strip comment lines (start with #)
    data_lines = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    if not data_lines:
        raise ValueError("Empty header file")

    # ── Record line ──────────────────────────────────────────────────────────
    parts = data_lines[0].split()
    n_sig: int | None = None
    fs: float | None = None
    sig_len: int | None = None

    if len(parts) >= 2:
        try:
            n_sig = int(parts[1])
        except ValueError:
            pass
    if len(parts) >= 3:
        # fs may look like "500", "500/1000", "500/1000(0)"
        fs_str = re.split(r"[/\(]", parts[2])[0]
        try:
            fs = float(fs_str)
        except ValueError:
            pass
    if len(parts) >= 4:
        try:
            sig_len = int(parts[3])
        except ValueError:
            pass

    # ── Signal lines ─────────────────────────────────────────────────────────
    sig_name: list[str] = []
    units: list[str] = []
    adc_gain: list[float] = []

    expected_sig_lines = n_sig if n_sig is not None else 0
    for sig_line in data_lines[1: 1 + expected_sig_lines]:
        sig_parts = sig_line.split()
        if not sig_parts:
            continue

        # Gain field (3rd token): "<gain>(/<baseline>)<(units)>" e.g. "200(0)mV" or "200/0(0)mV"
        gain_str = sig_parts[2] if len(sig_parts) > 2 else ""
        gain_val: float | None = None
        unit_val = ""
        gain_match = re.match(r"^([0-9.eE+\-]+)", gain_str)
        if gain_match:
            try:
                gain_val = float(gain_match.group(1))
            except ValueError:
                pass
        unit_match = re.search(r"\(([^)]+)\)([^\s]*)", gain_str)
        if unit_match:
            # Format: "200/0(0)mV" → unit after the parenthetical
            unit_val = unit_match.group(2)
        else:
            # Format: "200 mV" (space-separated unit as 4th token)
            if len(sig_parts) > 3 and not sig_parts[3].lstrip("-").isdigit():
                unit_val = sig_parts[3]

        # Description: everything after the 8th token (block_size)
        description = " ".join(sig_parts[8:]) if len(sig_parts) > 8 else ""
        sig_name.append(description or f"sig{len(sig_name)}")
        units.append(unit_val)
        if gain_val is not None:
            adc_gain.append(gain_val)

    return {
        "fs": fs,
        "n_sig": n_sig,
        "sig_len": sig_len,
        "sig_name": sig_name,
        "units": units,
        "adc_gain": adc_gain,
        "comments": [],
    }
