"""
WFDB reader for the DatasetMapper.

Extracts record IDs from PhysioNet WFDB datasets by scanning for .hea header files.
Each .hea file corresponds to one ECG recording — the stem is the record ID.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_wfdb_directory(path: str) -> dict[str, Any]:
    """
    Scan a directory for WFDB record headers (.hea files).

    Returns:
        record_ids: list of record ID strings (filename stems)
        fields: list of signal names from the first header parsed
        sample_values: metadata from the first header
        total_records: int
    """
    root = Path(path)
    hea_files = sorted(root.rglob("*.hea"))

    if not hea_files:
        return {"record_ids": [], "fields": [], "sample_values": {}, "total_records": 0}

    record_ids = [f.stem for f in hea_files]

    # Parse first header for metadata
    fields: list[str] = []
    sample_values: dict[str, Any] = {}
    try:
        import wfdb
        rec = wfdb.rdheader(str(hea_files[0].with_suffix("")))
        fields = list(rec.sig_name) if hasattr(rec, "sig_name") else []
        sample_values = {
            "fs": getattr(rec, "fs", None),
            "n_sig": getattr(rec, "n_sig", None),
            "sig_len": getattr(rec, "sig_len", None),
            "units": getattr(rec, "units", []),
        }
    except ImportError:
        logger.debug("wfdb library not installed — header metadata skipped")
        # Can still extract record IDs from filenames
    except Exception as exc:
        logger.debug(f"wfdb header parse failed: {exc}")

    return {
        "record_ids": sorted(set(record_ids)),
        "fields": fields,
        "sample_values": sample_values,
        "total_records": len(record_ids),
    }
