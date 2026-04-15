"""
Numpy reader for the DatasetMapper.

Some ECG datasets ship signals as .npy or .npz arrays.
Infers record count from the first axis of the array.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_numpy_directory(path: str) -> dict[str, Any]:
    """
    Scan a directory for numpy array files and infer record count.

    For .npz files also reads the archive key names.
    Record IDs are synthetic indices since numpy arrays have no built-in IDs.

    Returns:
        record_ids: synthetic IDs like ["npy_000000", ...]
        fields: array key names (from .npz) or ["signal_array"]
        sample_values: shape/dtype metadata
    """
    try:
        import numpy as np
    except ImportError:
        logger.debug("numpy not installed — numpy reader skipped")
        return {"record_ids": [], "fields": [], "sample_values": {}, "error": "numpy not installed"}

    root = Path(path)
    npy_files = sorted(root.rglob("*.npy"))
    npz_files = sorted(root.rglob("*.npz"))

    total_records = 0
    fields: list[str] = []
    sample_values: dict[str, Any] = {}

    # .npy files: each file = one array, first dim = record count
    for npy_path in npy_files[:5]:
        try:
            arr = np.load(str(npy_path), mmap_mode="r")
            total_records += arr.shape[0] if arr.ndim > 0 else 1
            if not fields:
                fields = ["signal_array"]
                sample_values = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            break
        except Exception as exc:
            logger.debug(f"numpy_reader: could not read {npy_path}: {exc}")

    # .npz files: archive of named arrays
    for npz_path in npz_files[:5]:
        try:
            archive = np.load(str(npz_path), allow_pickle=False)
            keys = list(archive.files)
            fields = fields or keys
            if keys and not sample_values:
                arr = archive[keys[0]]
                total_records += arr.shape[0] if arr.ndim > 0 else 1
                sample_values = {
                    "keys": keys,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }
            break
        except Exception as exc:
            logger.debug(f"numpy_reader: could not read {npz_path}: {exc}")

    record_ids = [f"npy_{i:06d}" for i in range(total_records)]

    return {
        "record_ids": record_ids,
        "fields": fields,
        "sample_values": sample_values,
        "total_files": len(npy_files) + len(npz_files),
        "is_synthetic_ids": True,
    }
