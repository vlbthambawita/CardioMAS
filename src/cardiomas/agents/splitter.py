from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from cardiomas import __version__
from cardiomas.schemas.split import ReproducibilityConfig, SplitManifest
from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.splitters.strategies import check_overlap, deterministic_split
from cardiomas.verbose import vprint

logger = logging.getLogger(__name__)


class SplitIntegrityError(RuntimeError):
    """Raised when generated splits fail an integrity check."""


def splitter_agent(state: GraphState) -> GraphState:
    """Generate reproducible train/val/test splits from real record IDs."""
    opts = state.user_options
    info = state.dataset_info
    analysis = state.analysis_report or {}
    paper = state.paper_findings or {}
    state.execution_log.append(LogEntry(agent="splitter", action="start"))

    # ── Determine ratios ───────────────────────────────────────────────────
    if opts.custom_split:
        total = sum(opts.custom_split.values())
        ratios = {k: v / total for k, v in opts.custom_split.items()}
    else:
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

    seed = opts.seed

    # ── Determine strategy ─────────────────────────────────────────────────
    paper_methodology = paper.get("split_methodology", "") or paper.get("analysis", "")
    use_official = (
        not opts.ignore_official
        and paper.get("found")
        and paper_methodology
        and "official" in paper_methodology.lower()
    )
    strategy = "official" if use_official else ("custom" if opts.custom_split else "deterministic")

    # ── Load record IDs — Phase 2: DatasetMap takes priority ──────────────
    record_ids = _load_record_ids(state, analysis)

    if not record_ids:
        msg = (
            "No record IDs could be loaded from the dataset. "
            "Ensure --local-path points to a directory with CSV, WFDB, HDF5, or EDF files. "
            "Synthetic ID fallback has been removed in V3 to preserve reproducibility guarantees."
        )
        state.errors.append(f"splitter: {msg}")
        vprint("splitter", f"[red]ABORT: {msg}[/red]")
        return state

    vprint("splitter", f"loaded {len(record_ids)} real record IDs (strategy={strategy})")

    # ── Run split ──────────────────────────────────────────────────────────
    splits = deterministic_split(record_ids, ratios, seed=seed)

    # ── Phase 4: integrity verification ───────────────────────────────────
    try:
        from cardiomas.agents.verification import verify_split_integrity
        verify_split_integrity(splits, record_ids)
        vprint("splitter", "split integrity verified — no overlap, full coverage")
    except ImportError:
        _basic_overlap_check(splits, state)
    except SplitIntegrityError as exc:
        state.errors.append(f"splitter integrity: {exc}")
        vprint("splitter", f"[red]INTEGRITY FAILURE: {exc}[/red]")
        return state

    # ── Build dataset checksum from real IDs ───────────────────────────────
    # Use DatasetMap checksum if available (preferred), else compute from loaded IDs
    dataset_checksum = _get_dataset_checksum(state, record_ids, ratios, seed)

    # ── Build manifest ─────────────────────────────────────────────────────
    repro = ReproducibilityConfig(
        cardiomas_version=__version__,
        seed=seed,
        dataset_name=info.name if info else "unknown",
        dataset_source_url=info.source_url if info else None,
        dataset_checksum=dataset_checksum,
        split_strategy=strategy,
        split_ratios=ratios,
        stratify_by=[opts.stratify_by] if opts.stratify_by else None,
        group_by=None,
        timestamp=datetime.utcnow(),
    )

    manifest = SplitManifest(
        dataset_name=info.name if info else "unknown",
        cardiomas_version=__version__,
        reproducibility_config=repro,
        splits=splits,
        split_stats={k: {"count": len(v)} for k, v in splits.items()},
    )
    state.proposed_splits = manifest

    # ── Save outputs locally ───────────────────────────────────────────────
    out_dir = Path(opts.output_dir) / manifest.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_file = out_dir / "splits.json"
    meta_file = out_dir / "split_metadata.json"

    splits_payload = {
        "dataset_name": manifest.dataset_name,
        "split_version": manifest.split_version,
        "cardiomas_version": manifest.cardiomas_version,
        "reproducibility_config": repro.model_dump(mode="json"),
        "splits": splits,
        "split_stats": manifest.split_stats,
    }
    splits_file.write_text(json.dumps(splits_payload, indent=2))
    meta_file.write_text(json.dumps(repro.model_dump(mode="json"), indent=2))

    state.local_output_dir = str(out_dir)
    vprint("splitter", f"saved splits → {splits_file}")
    vprint("splitter", f"saved metadata → {meta_file}")

    state.execution_log.append(LogEntry(
        agent="splitter", action="complete",
        detail=f"{len(record_ids)} records → {list(splits.keys())} → {out_dir}",
    ))
    return state


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_record_ids(state: GraphState, analysis: dict) -> list[str]:
    """Load real record IDs — DatasetMap > analysis report > CSV fallback."""

    # 1. Phase 2: DatasetMap (most reliable)
    dataset_map = getattr(state, "dataset_map", None)
    if dataset_map is None and analysis.get("dataset_map"):
        dataset_map = analysis["dataset_map"]

    if dataset_map is not None:
        ids = None
        if hasattr(dataset_map, "all_record_ids"):
            ids = dataset_map.all_record_ids
        elif isinstance(dataset_map, dict):
            ids = dataset_map.get("all_record_ids")
        if ids:
            return list(ids)

    # 2. CSV fallback (pre-Phase 2 behaviour)
    opts = state.user_options
    info = state.dataset_info
    local_path = opts.local_path or (
        str(info.local_path) if info and info.local_path else ""
    )
    if local_path and Path(local_path).exists():
        try:
            import pandas as pd
            from cardiomas.tools.data_tools import list_dataset_files

            files = list_dataset_files.invoke(
                {"path": local_path, "max_depth": 2}
            ).get("files", [])
            csv_files = [
                Path(local_path) / f["path"]
                for f in files if f["suffix"] in (".csv", ".tsv")
            ]
            id_field = (
                analysis.get("id_field")
                or (info.ecg_id_field if info else None)
                or "record_id"
            )
            for csv_path in csv_files[:3]:
                try:
                    df = pd.read_csv(csv_path)
                    if id_field in df.columns:
                        return df[id_field].astype(str).tolist()
                    return df.iloc[:, 0].astype(str).tolist()
                except Exception:
                    continue
        except Exception as exc:
            logger.warning(f"splitter CSV fallback failed: {exc}")

    return []


def _basic_overlap_check(splits: dict, state: GraphState) -> None:
    """Basic overlap check when verification module is not available."""
    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = check_overlap(splits[names[i]], splits[names[j]])
            if overlap:
                state.errors.append(
                    f"Split overlap between {names[i]} and {names[j]}: {len(overlap)} IDs"
                )


def _get_dataset_checksum(
    state: GraphState, record_ids: list[str], ratios: dict, seed: int
) -> str:
    """Return dataset checksum from DatasetMap (preferred) or compute from IDs."""
    dataset_map = getattr(state, "dataset_map", None)
    if dataset_map is not None:
        chk = None
        if hasattr(dataset_map, "dataset_checksum"):
            chk = dataset_map.dataset_checksum
        elif isinstance(dataset_map, dict):
            chk = dataset_map.get("dataset_checksum")
        if chk:
            return chk

    # Fallback: compute checksum from loaded IDs (same algorithm as DatasetMapper)
    payload = json.dumps(
        {"ids": sorted(record_ids), "ratios": ratios, "seed": seed}
    )
    return hashlib.sha256(payload.encode()).hexdigest()
