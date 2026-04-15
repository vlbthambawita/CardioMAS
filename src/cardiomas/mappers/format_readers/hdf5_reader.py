"""
HDF5 reader for the DatasetMapper.

Used for datasets like MIMIC-IV-ECG that store signals in HDF5 format.
Extracts top-level dataset keys as record IDs and reads attribute metadata.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_hdf5(path: str) -> dict[str, Any]:
    """
    Inspect a single HDF5 file.

    Returns:
        record_ids: top-level keys (each treated as a record ID)
        fields: attribute keys found on the root or first group
        sample_values: sample attribute values
    """
    try:
        import h5py
    except ImportError:
        logger.debug("h5py not installed — HDF5 reader skipped")
        return {"record_ids": [], "fields": [], "sample_values": {}, "error": "h5py not installed"}

    try:
        with h5py.File(path, "r") as f:
            top_keys = list(f.keys())
            record_ids = [str(k) for k in top_keys[:50_000]]

            # Collect attribute names from root and first group
            fields: list[str] = list(f.attrs.keys())
            sample_values: dict[str, Any] = {}
            for k, v in list(f.attrs.items())[:10]:
                try:
                    sample_values[k] = v.tolist() if hasattr(v, "tolist") else str(v)
                except Exception:
                    pass

            if top_keys:
                try:
                    first = f[top_keys[0]]
                    if hasattr(first, "attrs"):
                        for k, v in list(first.attrs.items())[:10]:
                            fields.append(k)
                            try:
                                sample_values[k] = v.tolist() if hasattr(v, "tolist") else str(v)
                            except Exception:
                                pass
                except Exception:
                    pass

        return {
            "record_ids": sorted(set(record_ids)),
            "fields": list(dict.fromkeys(fields)),  # deduplicate, preserve order
            "sample_values": sample_values,
        }
    except Exception as exc:
        logger.warning(f"hdf5_reader: failed to read {path}: {exc}")
        return {"record_ids": [], "fields": [], "sample_values": {}, "error": str(exc)}


def read_hdf5_directory(path: str) -> dict[str, Any]:
    """Scan a directory for HDF5 files and aggregate record IDs."""
    root = Path(path)
    h5_files = sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.h5"))

    all_ids: list[str] = []
    fields: list[str] = []
    sample_values: dict[str, Any] = {}

    for h5_path in h5_files[:10]:  # read up to 10 files to limit scan time
        result = read_hdf5(str(h5_path))
        all_ids.extend(result.get("record_ids", []))
        if not fields:
            fields = result.get("fields", [])
            sample_values = result.get("sample_values", {})

    return {
        "record_ids": sorted(set(all_ids)),
        "fields": fields,
        "sample_values": sample_values,
        "total_files": len(h5_files),
    }
