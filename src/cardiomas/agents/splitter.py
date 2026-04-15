from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from cardiomas import __version__
from cardiomas.agents.base import run_agent
from cardiomas.schemas.split import SplitManifest, ReproducibilityConfig
from cardiomas.schemas.state import GraphState, LogEntry
from cardiomas.verbose import vprint
from cardiomas.splitters.strategies import (
    PatientStratifiedSplit,
    RecordStratifiedSplit,
    OfficialSplit,
    CustomSplit,
    deterministic_split,
    check_overlap,
)

logger = logging.getLogger(__name__)


def splitter_agent(state: GraphState) -> GraphState:
    """Generate reproducible train/val/test splits based on analysis and paper findings."""
    from cardiomas.llm_factory import get_llm
    from pathlib import Path
    import pandas as pd

    opts = state.user_options
    info = state.dataset_info
    analysis = state.analysis_report or {}
    paper = state.paper_findings or {}
    state.execution_log.append(LogEntry(agent="splitter", action="start"))

    # Determine ratios
    if opts.custom_split:
        total = sum(opts.custom_split.values())
        ratios = {k: v / total for k, v in opts.custom_split.items()}
    else:
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

    seed = opts.seed

    # Determine strategy
    use_official = (
        not opts.ignore_official
        and paper.get("found")
        and "official" in paper.get("analysis", "").lower()
        and paper.get("analysis", "").lower().count("yes") >= 1
    )

    # Try to load record IDs from local metadata
    record_ids: list[str] = []
    local_path = opts.local_path or (str(info.local_path) if info and info.local_path else "")

    if local_path and Path(local_path).exists():
        from cardiomas.tools.data_tools import list_dataset_files
        files = list_dataset_files.invoke({"path": local_path, "max_depth": 2}).get("files", [])
        csv_files = [Path(local_path) / f["path"] for f in files if f["suffix"] in (".csv", ".tsv")]
        for csv_path in csv_files[:3]:
            try:
                df = pd.read_csv(csv_path)
                id_field = info.ecg_id_field if info else "record_id"
                if id_field in df.columns:
                    record_ids = df[id_field].astype(str).tolist()
                    break
                # fallback: use first column
                record_ids = df.iloc[:, 0].astype(str).tolist()
                break
            except Exception:
                continue

    if not record_ids:
        # Generate synthetic IDs from analysis report estimate
        num_records = info.num_records if info else 1000
        record_ids = [f"record_{i:06d}" for i in range(num_records or 1000)]
        logger.warning(f"No real record IDs found; using {len(record_ids)} synthetic IDs for split demo.")

    # Run split
    splits = deterministic_split(record_ids, ratios, seed=seed)

    # Verify no overlap
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            overlap = check_overlap(splits[split_names[i]], splits[split_names[j]])
            if overlap:
                state.errors.append(f"Split overlap between {split_names[i]} and {split_names[j]}: {len(overlap)} IDs")

    # Build checksum
    payload = json.dumps({"ids": sorted(record_ids), "ratios": ratios, "seed": seed})
    checksum = hashlib.sha256(payload.encode()).hexdigest()

    repro = ReproducibilityConfig(
        cardiomas_version=__version__,
        seed=seed,
        dataset_name=info.name if info else "unknown",
        dataset_source_url=info.source_url if info else None,
        dataset_checksum=checksum,
        split_strategy="official" if use_official else ("custom" if opts.custom_split else "deterministic"),
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

    # ── Save outputs locally ──────────────────────────────────────────────
    out_dir = Path(opts.output_dir) / manifest.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_file = out_dir / "splits.json"
    meta_file   = out_dir / "split_metadata.json"

    splits_payload = {
        "dataset_name":          manifest.dataset_name,
        "split_version":         manifest.split_version,
        "cardiomas_version":     manifest.cardiomas_version,
        "reproducibility_config": repro.model_dump(mode="json"),
        "splits":                splits,
        "split_stats":           manifest.split_stats,
    }
    splits_file.write_text(json.dumps(splits_payload, indent=2))
    meta_file.write_text(json.dumps(repro.model_dump(mode="json"), indent=2))

    state.local_output_dir = str(out_dir)
    vprint("splitter", f"saved splits → {splits_file}")
    vprint("splitter", f"saved metadata → {meta_file}")

    state.execution_log.append(
        LogEntry(agent="splitter", action="complete",
                 detail=f"{len(record_ids)} records → {list(splits.keys())} → {out_dir}")
    )
    return state
