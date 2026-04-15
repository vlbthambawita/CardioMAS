"""
EDF (European Data Format) reader for the DatasetMapper.

Each .edf file is treated as one ECG recording.
The file stem becomes the record ID.
Reads signal channel names and patient info from the EDF header.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_edf_directory(path: str) -> dict[str, Any]:
    """
    Scan a directory for EDF files and extract record IDs + header metadata.

    Returns:
        record_ids: list of EDF file stems
        fields: signal channel names from first EDF header
        sample_values: header attributes from first file
    """
    root = Path(path)
    edf_files = sorted(root.rglob("*.edf"))

    if not edf_files:
        return {"record_ids": [], "fields": [], "sample_values": {}, "total_files": 0}

    record_ids = [f.stem for f in edf_files]
    fields: list[str] = []
    sample_values: dict[str, Any] = {}

    # Try to parse first EDF header for channel names
    try:
        import pyedflib
        with pyedflib.EdfReader(str(edf_files[0])) as ef:
            fields = ef.getSignalLabels()
            sample_values = {
                "patient_info": ef.getPatientCode(),
                "recording_info": ef.getRecordingAdditional(),
                "start_date": str(ef.getStartdatetime()),
                "n_signals": ef.signals_in_file,
                "sample_frequencies": ef.getSampleFrequencies().tolist(),
            }
    except ImportError:
        logger.debug("pyedflib not installed — EDF header metadata skipped")
    except Exception as exc:
        logger.debug(f"edf_reader: header parse failed: {exc}")

    return {
        "record_ids": sorted(set(record_ids)),
        "fields": fields,
        "sample_values": sample_values,
        "total_files": len(edf_files),
    }
