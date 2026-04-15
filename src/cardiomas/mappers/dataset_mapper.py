"""
DatasetMapper — builds a DatasetMap by recursively scanning a dataset directory.

Dispatches to format-specific readers (CSV, WFDB, HDF5, EDF, numpy) and
merges the results into a single authoritative DatasetMap.

The DatasetMap is the single source of truth for:
- record IDs (used by splitter — replaces synthetic fallback)
- patient → record mapping (used by security agent for leakage detection)
- dataset checksum (used in ReproducibilityConfig)
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from cardiomas.mappers.schemas import DatasetMap, FileInventory

logger = logging.getLogger(__name__)


class DatasetMapper:
    """
    Orchestrates multi-format scanning of an ECG dataset directory.

    Usage:
        mapper = DatasetMapper("/data/ptb-xl/")
        dataset_map = mapper.build()
    """

    def __init__(self, root_path: str) -> None:
        self.root = Path(root_path).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.root}")

    def build(self) -> DatasetMap:
        """Run all format readers and merge into a single DatasetMap."""
        logger.info(f"DatasetMapper: scanning {self.root}")

        inventory: list[FileInventory] = []
        format_distribution: dict[str, int] = {}
        all_record_ids: list[str] = []
        patient_record_map: dict[str, list[str]] = {}
        id_field = "record_id"
        patient_id_field = None
        label_field = None
        label_type = "none"
        label_values: list = []
        missing_fracs: list[float] = []
        all_fields: list[str] = []
        metadata_files: list[str] = []
        signal_files: list[str] = []

        # ── 1. CSV / TSV metadata files ────────────────────────────────────
        csv_result = self._scan_csv()
        if csv_result.get("record_ids"):
            all_record_ids.extend(csv_result["record_ids"])
            id_field = csv_result.get("id_field", id_field)
            patient_id_field = csv_result.get("patient_id_field")
            label_field = csv_result.get("label_field")
            label_type = csv_result.get("label_type", "none")
            label_values = csv_result.get("label_values", [])
            missing_fracs.append(csv_result.get("missing_data_fraction", 0.0))
            all_fields.extend(csv_result.get("fields", []))
            metadata_files.extend(csv_result.get("file_paths", []))
            if csv_result.get("patient_record_map"):
                _merge_patient_map(patient_record_map, csv_result["patient_record_map"])
            format_distribution["csv"] = csv_result.get("total_files", 0)

        # ── 2. WFDB records ────────────────────────────────────────────────
        wfdb_result = self._scan_wfdb()
        if wfdb_result.get("record_ids"):
            wfdb_ids = wfdb_result["record_ids"]
            signal_files.extend([str(self.root / f"{rid}.hea") for rid in wfdb_ids[:100]])
            # Only use WFDB IDs as primary if CSV gave none
            if not all_record_ids:
                all_record_ids.extend(wfdb_ids)
                id_field = "wfdb_record"
            all_fields.extend(wfdb_result.get("fields", []))
            format_distribution["wfdb"] = len(wfdb_ids)

        # ── 3. HDF5 ────────────────────────────────────────────────────────
        hdf5_result = self._scan_hdf5()
        if hdf5_result.get("record_ids"):
            hdf5_ids = hdf5_result["record_ids"]
            signal_files.extend([str(p) for p in sorted(self.root.rglob("*.hdf5"))[:5]])
            if not all_record_ids:
                all_record_ids.extend(hdf5_ids)
                id_field = "hdf5_key"
            all_fields.extend(hdf5_result.get("fields", []))
            format_distribution["hdf5"] = len(hdf5_ids)

        # ── 4. EDF ─────────────────────────────────────────────────────────
        edf_result = self._scan_edf()
        if edf_result.get("record_ids"):
            edf_ids = edf_result["record_ids"]
            signal_files.extend([str(p) for p in sorted(self.root.rglob("*.edf"))[:5]])
            if not all_record_ids:
                all_record_ids.extend(edf_ids)
                id_field = "edf_stem"
            all_fields.extend(edf_result.get("fields", []))
            format_distribution["edf"] = len(edf_ids)

        # ── 5. Numpy (last resort — synthetic IDs only) ────────────────────
        if not all_record_ids:
            numpy_result = self._scan_numpy()
            if numpy_result.get("record_ids"):
                all_record_ids.extend(numpy_result["record_ids"])
                all_fields.extend(numpy_result.get("fields", []))
                format_distribution["numpy"] = numpy_result.get("total_files", 0)
                logger.warning(
                    "DatasetMapper: only numpy arrays found — record IDs are synthetic. "
                    "Reproducibility claim will be limited."
                )

        # ── Deduplicate and sort ───────────────────────────────────────────
        all_record_ids = sorted(set(all_record_ids))
        all_fields = list(dict.fromkeys(all_fields))  # deduplicate, preserve order

        # ── Compute dataset checksum ───────────────────────────────────────
        checksum = _sha256_ids(all_record_ids)

        # ── Total file count ───────────────────────────────────────────────
        all_files = list(self.root.rglob("*"))
        total_files = sum(1 for f in all_files if f.is_file())
        format_distribution["total"] = total_files

        missing_frac = float(sum(missing_fracs) / len(missing_fracs)) if missing_fracs else 0.0

        dataset_map = DatasetMap(
            root_path=str(self.root),
            total_files=total_files,
            format_distribution=format_distribution,
            all_record_ids=all_record_ids,
            patient_record_map=patient_record_map,
            id_field=id_field,
            patient_id_field=patient_id_field,
            label_field=label_field,
            label_values=label_values[:50],
            label_type=label_type,  # type: ignore[arg-type]
            metadata_files=metadata_files[:20],
            signal_files=signal_files[:20],
            file_inventory=inventory,
            missing_data_fraction=missing_frac,
            available_fields=all_fields[:100],
            dataset_checksum=checksum,
        )

        logger.info(
            f"DatasetMapper: complete — {len(all_record_ids)} records"
            f" checksum={checksum[:12]}…"
        )
        return dataset_map

    # ── Private scan helpers ───────────────────────────────────────────────

    def _scan_csv(self) -> dict:
        """Scan all CSV/TSV files and return merged metadata."""
        from cardiomas.mappers.format_readers.csv_reader import read_csv

        csv_files = sorted(self.root.rglob("*.csv")) + sorted(self.root.rglob("*.tsv"))
        if not csv_files:
            return {}

        merged_ids: list[str] = []
        merged_patient_map: dict[str, list[str]] = {}
        id_field = "record_id"
        patient_id_field = None
        label_field = None
        label_type = "none"
        label_values: list = []
        missing_fracs: list[float] = []
        all_fields: list[str] = []
        file_paths: list[str] = []

        for csv_path in csv_files[:10]:
            result = read_csv(str(csv_path))
            if "error" in result:
                continue
            file_paths.append(str(csv_path.relative_to(self.root)))
            merged_ids.extend(result.get("record_ids", []))
            _merge_patient_map(merged_patient_map, result.get("patient_record_map", {}))
            missing_fracs.append(result.get("missing_data_fraction", 0.0))
            # Use field info from the first successful CSV
            if not all_fields:
                id_field = result.get("id_field", id_field)
                patient_id_field = result.get("patient_id_field")
                label_field = result.get("label_field")
                label_type = result.get("label_type", "none")
                label_values = result.get("label_values", [])
                all_fields = result.get("fields", [])

        return {
            "record_ids": sorted(set(merged_ids)),
            "patient_record_map": merged_patient_map,
            "id_field": id_field,
            "patient_id_field": patient_id_field,
            "label_field": label_field,
            "label_type": label_type,
            "label_values": label_values,
            "fields": all_fields,
            "missing_data_fraction": (
                sum(missing_fracs) / len(missing_fracs) if missing_fracs else 0.0
            ),
            "total_files": len(csv_files),
            "file_paths": file_paths,
        }

    def _scan_wfdb(self) -> dict:
        from cardiomas.mappers.format_readers.wfdb_reader import read_wfdb_directory
        return read_wfdb_directory(str(self.root))

    def _scan_hdf5(self) -> dict:
        from cardiomas.mappers.format_readers.hdf5_reader import read_hdf5_directory
        return read_hdf5_directory(str(self.root))

    def _scan_edf(self) -> dict:
        from cardiomas.mappers.format_readers.edf_reader import read_edf_directory
        return read_edf_directory(str(self.root))

    def _scan_numpy(self) -> dict:
        from cardiomas.mappers.format_readers.numpy_reader import read_numpy_directory
        return read_numpy_directory(str(self.root))


# ── Helpers ────────────────────────────────────────────────────────────────

def _merge_patient_map(
    dest: dict[str, list[str]], src: dict[str, list[str]]
) -> None:
    for pid, rids in src.items():
        dest.setdefault(pid, []).extend(rids)


def _sha256_ids(record_ids: list[str]) -> str:
    """Compute SHA-256 of the sorted record ID list (same as splitter seeding)."""
    if not record_ids:
        return ""
    payload = json.dumps(sorted(record_ids), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()
