from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cardiomas.tools.security_tools import scan_for_pii, validate_split_file, check_patient_leakage


class TestScanForPii:
    def test_no_pii(self):
        result = scan_for_pii.invoke({"data": "rec_001 rec_002 ecg_2024"})
        assert not result["has_pii"]

    def test_detects_email(self):
        result = scan_for_pii.invoke({"data": "contact: user@example.com"})
        assert result["has_pii"]
        types = [f["type"] for f in result["findings"]]
        assert "email" in types

    def test_clean_record_ids(self):
        ids = [f"ecg_{i:06d}" for i in range(1000)]
        data = json.dumps({"splits": {"train": ids[:700], "test": ids[700:]}})
        result = scan_for_pii.invoke({"data": data})
        assert not result["has_pii"]


class TestValidateSplitFile:
    def _write_split(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_valid_split_file(self):
        path = self._write_split({"splits": {"train": ["r001", "r002"], "test": ["r003"]}})
        result = validate_split_file.invoke({"path": path})
        assert result["valid"]
        assert result["issues"] == []

    def test_missing_splits_key(self):
        path = self._write_split({"train": ["r001"]})
        result = validate_split_file.invoke({"path": path})
        assert not result["valid"]
        assert any("splits" in i for i in result["issues"])

    def test_nonexistent_file(self):
        result = validate_split_file.invoke({"path": "/nonexistent/file.json"})
        assert not result["valid"]


class TestCheckPatientLeakage:
    def test_no_leakage(self):
        splits = {"train": ["r1", "r2", "r3"], "test": ["r4", "r5"]}
        mapping = {"r1": "p1", "r2": "p2", "r3": "p3", "r4": "p4", "r5": "p5"}
        result = check_patient_leakage.invoke({"splits": splits, "patient_mapping": mapping})
        assert not result["leakage_detected"]

    def test_detects_leakage(self):
        splits = {"train": ["r1", "r2"], "test": ["r3", "r4"]}
        mapping = {"r1": "patient_A", "r2": "patient_B", "r3": "patient_A", "r4": "patient_C"}
        result = check_patient_leakage.invoke({"splits": splits, "patient_mapping": mapping})
        assert result["leakage_detected"]
        assert any("patient_A" in str(d) for d in result["leakage_details"])
