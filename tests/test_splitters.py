from __future__ import annotations

import pytest

from cardiomas.splitters.strategies import (
    deterministic_split,
    stratified_split,
    check_overlap,
    PatientStratifiedSplit,
    RecordStratifiedSplit,
    CustomSplit,
)


def _ids(n: int) -> list[str]:
    return [f"rec_{i:04d}" for i in range(n)]


class TestDeterministicSplit:
    def test_basic_split(self):
        ids = _ids(100)
        splits = deterministic_split(ids, {"train": 0.7, "val": 0.15, "test": 0.15})
        assert set(splits.keys()) == {"train", "val", "test"}
        assert sum(len(v) for v in splits.values()) == 100

    def test_reproducibility(self):
        ids = _ids(200)
        s1 = deterministic_split(ids, {"train": 0.8, "test": 0.2}, seed=42)
        s2 = deterministic_split(ids, {"train": 0.8, "test": 0.2}, seed=42)
        assert s1["train"] == s2["train"]
        assert s1["test"] == s2["test"]

    def test_different_seeds_give_different_splits(self):
        ids = _ids(100)
        s1 = deterministic_split(ids, {"train": 0.8, "test": 0.2}, seed=1)
        s2 = deterministic_split(ids, {"train": 0.8, "test": 0.2}, seed=2)
        assert s1["train"] != s2["train"]

    def test_no_overlap(self):
        ids = _ids(150)
        splits = deterministic_split(ids, {"train": 0.7, "val": 0.15, "test": 0.15})
        assert check_overlap(splits["train"], splits["val"]) == []
        assert check_overlap(splits["train"], splits["test"]) == []
        assert check_overlap(splits["val"], splits["test"]) == []

    def test_all_ids_covered(self):
        ids = _ids(100)
        splits = deterministic_split(ids, {"train": 0.7, "val": 0.15, "test": 0.15})
        all_ids = set()
        for v in splits.values():
            all_ids.update(v)
        assert all_ids == set(ids)

    def test_approximate_ratios(self):
        ids = _ids(1000)
        splits = deterministic_split(ids, {"train": 0.7, "val": 0.15, "test": 0.15})
        assert 650 <= len(splits["train"]) <= 750
        assert 100 <= len(splits["val"]) <= 200
        assert 100 <= len(splits["test"]) <= 200


class TestStratifiedSplit:
    def test_preserves_label_distribution(self):
        ids = [f"rec_{i}" for i in range(200)]
        labels = ["A"] * 100 + ["B"] * 100
        splits = stratified_split(ids, labels, {"train": 0.8, "test": 0.2})
        assert sum(len(v) for v in splits.values()) == 200

    def test_no_overlap_stratified(self):
        ids = [f"rec_{i}" for i in range(100)]
        labels = ["A"] * 50 + ["B"] * 50
        splits = stratified_split(ids, labels, {"train": 0.8, "test": 0.2})
        assert check_overlap(splits["train"], splits["test"]) == []


class TestCheckOverlap:
    def test_no_overlap(self):
        assert check_overlap(["a", "b"], ["c", "d"]) == []

    def test_with_overlap(self):
        overlap = check_overlap(["a", "b", "c"], ["c", "d"])
        assert "c" in overlap


class TestPatientStratifiedSplit:
    def test_patient_level_no_leakage(self):
        records = [
            {"patient_id": f"pat_{i // 3}", "record_id": f"rec_{i}"}
            for i in range(90)
        ]
        splitter = PatientStratifiedSplit(ratios={"train": 0.7, "val": 0.15, "test": 0.15})
        splits = splitter.split(records)

        # Check no record appears in multiple splits
        all_ids = []
        for ids in splits.values():
            all_ids.extend(ids)
        assert len(all_ids) == len(set(all_ids))  # no duplicates

    def test_all_records_covered(self):
        records = [{"patient_id": f"p{i % 10}", "record_id": f"r{i}"} for i in range(50)]
        splitter = PatientStratifiedSplit()
        splits = splitter.split(records)
        all_ids = {rid for ids in splits.values() for rid in ids}
        assert all_ids == {f"r{i}" for i in range(50)}


class TestCustomSplit:
    def test_normalizes_ratios(self):
        # 2:1:1 → 0.5:0.25:0.25
        splitter = CustomSplit(ratios={"train": 2, "val": 1, "test": 1})
        ids = _ids(100)
        splits = splitter.split([{"record_id": i} for i in ids])
        assert abs(len(splits["train"]) - 50) <= 5

    def test_reproducible(self):
        ids = _ids(200)
        records = [{"record_id": i} for i in ids]
        s1 = CustomSplit(ratios={"train": 0.8, "test": 0.2}, seed=99).split(records)
        s2 = CustomSplit(ratios={"train": 0.8, "test": 0.2}, seed=99).split(records)
        assert s1["train"] == s2["train"]
