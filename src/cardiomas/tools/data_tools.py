from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def download_dataset(source: str, dest_path: str = "") -> dict[str, Any]:
    """Download a dataset from the given source URL or path.
    Returns dict with 'local_path' and 'status'."""
    from cardiomas.datasets.loaders import get_loader
    try:
        loader = get_loader(source)
        dest = Path(dest_path) if dest_path else None
        local_path = loader.load(source, dest)
        return {"local_path": str(local_path), "status": "ok"}
    except Exception as e:
        return {"local_path": "", "status": "error", "error": str(e)}


@tool
def list_dataset_files(path: str, max_depth: int = 2) -> dict[str, Any]:
    """List files in a dataset directory up to max_depth levels deep.
    Returns dict with 'files' list (path, size, type)."""
    root = Path(path)
    if not root.exists():
        return {"files": [], "error": f"Path not found: {path}"}
    files = []
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts)
        if depth > max_depth:
            continue
        if p.is_file():
            files.append({
                "path": str(p.relative_to(root)),
                "size_bytes": p.stat().st_size,
                "suffix": p.suffix.lower(),
            })
    return {"files": files[:500], "total_found": len(files)}


@tool
def read_csv_metadata(path: str, nrows: int = 5) -> dict[str, Any]:
    """Read a CSV metadata file and return its schema and sample rows.
    Returns dict with 'columns', 'dtypes', 'sample', 'total_rows'."""
    import pandas as pd
    try:
        df = pd.read_csv(path, nrows=nrows)
        full = pd.read_csv(path)
        return {
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample": df.head(nrows).to_dict(orient="records"),
            "total_rows": len(full),
        }
    except Exception as e:
        return {"columns": [], "sample": [], "error": str(e)}


@tool
def read_wfdb_header(record_path: str) -> dict[str, Any]:
    """Read a WFDB .hea header file and return metadata (no signals loaded).
    Returns dict with record metadata."""
    try:
        import wfdb
        record = wfdb.rdheader(record_path.replace(".hea", ""))
        return {
            "record_name": record.record_name,
            "n_sig": record.n_sig,
            "fs": record.fs,
            "sig_len": record.sig_len,
            "sig_name": record.sig_name,
            "units": record.units,
            "comments": record.comments,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def inspect_hdf5(path: str) -> dict[str, Any]:
    """Inspect an HDF5 file and return its structure (keys and dataset shapes).
    Returns dict with nested structure."""
    try:
        import h5py

        def _inspect(obj: Any, depth: int = 0) -> Any:
            if depth > 4:
                return "..."
            if isinstance(obj, h5py.Dataset):
                return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            elif isinstance(obj, h5py.Group):
                return {k: _inspect(v, depth + 1) for k, v in list(obj.items())[:20]}
            return str(type(obj))

        with h5py.File(path, "r") as f:
            return {"structure": _inspect(f)}
    except Exception as e:
        return {"error": str(e)}


@tool
def compute_statistics(data_path: str, columns: list[str]) -> dict[str, Any]:
    """Compute distribution statistics for specified columns of a CSV file.
    Returns per-column counts, value_counts, and basic stats."""
    import pandas as pd
    try:
        df = pd.read_csv(data_path)
        result: dict[str, Any] = {}
        for col in columns:
            if col not in df.columns:
                result[col] = {"error": "column not found"}
                continue
            s = df[col]
            stats: dict[str, Any] = {
                "count": int(s.count()),
                "null_count": int(s.isna().sum()),
            }
            if s.dtype in ("int64", "float64"):
                stats.update({
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "median": float(s.median()),
                })
            else:
                vc = s.value_counts().head(20).to_dict()
                stats["value_counts"] = {str(k): int(v) for k, v in vc.items()}
            result[col] = stats
        return {"stats": result}
    except Exception as e:
        return {"error": str(e)}
