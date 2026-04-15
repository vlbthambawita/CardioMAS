from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


@tool
def run_deterministic_split(
    record_ids: list[str],
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministically split record_ids into named subsets by ratio.
    Same inputs always yield same splits. Returns dict of split_name -> [ids]."""
    from cardiomas.splitters.strategies import deterministic_split
    splits = deterministic_split(record_ids, ratios, seed)
    return {"splits": splits, "sizes": {k: len(v) for k, v in splits.items()}}


@tool
def run_stratified_split(
    record_ids: list[str],
    labels: list[str],
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, Any]:
    """Stratified split that preserves label distribution.
    Returns dict of split_name -> [ids]."""
    from cardiomas.splitters.strategies import stratified_split
    splits = stratified_split(record_ids, labels, ratios, seed)
    return {"splits": splits, "sizes": {k: len(v) for k, v in splits.items()}}


@tool
def check_split_overlap(split_a: list[str], split_b: list[str]) -> dict[str, Any]:
    """Check for overlapping record IDs between two splits.
    Returns overlap list (must be empty for valid splits)."""
    from cardiomas.splitters.strategies import check_overlap
    overlap = check_overlap(split_a, split_b)
    return {"overlap": overlap, "overlap_count": len(overlap), "is_clean": len(overlap) == 0}


@tool
def compute_split_stats(splits: dict[str, list[str]], metadata_path: str = "", label_field: str = "") -> dict[str, Any]:
    """Compute per-split statistics including sizes and label distributions.
    metadata_path: optional CSV with record metadata. label_field: column to stratify by."""
    import pandas as pd

    stats: dict[str, Any] = {
        "sizes": {k: len(v) for k, v in splits.items()},
        "total": sum(len(v) for v in splits.values()),
    }

    if metadata_path and label_field:
        try:
            df = pd.read_csv(metadata_path)
            for split_name, ids in splits.items():
                subset = df[df.iloc[:, 0].astype(str).isin(ids)]
                if label_field in subset.columns:
                    vc = subset[label_field].value_counts().to_dict()
                    stats.setdefault("label_distributions", {})[split_name] = {str(k): int(v) for k, v in vc.items()}
        except Exception as e:
            stats["metadata_error"] = str(e)

    return stats
