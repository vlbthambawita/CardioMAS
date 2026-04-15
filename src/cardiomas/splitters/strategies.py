from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

import numpy as np


def _deterministic_seed(record_ids: list[str], seed: int, strategy: str) -> int:
    """Derive a deterministic integer seed from inputs."""
    payload = json.dumps({"ids": sorted(record_ids), "seed": seed, "strategy": strategy})
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return int(digest[:16], 16)


def deterministic_split(
    record_ids: list[str],
    ratios: dict[str, float],
    seed: int = 42,
    strategy: str = "record",
) -> dict[str, list[str]]:
    """Split record_ids deterministically. Same inputs always yield same output."""
    ids = sorted(record_ids)
    derived_seed = _deterministic_seed(ids, seed, strategy)
    rng = np.random.default_rng(derived_seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    splits: dict[str, list[str]] = {}
    total = len(shuffled)
    start = 0
    items = list(ratios.items())
    for i, (name, ratio) in enumerate(items):
        if i == len(items) - 1:
            splits[name] = shuffled[start:]
        else:
            end = start + math.floor(total * ratio)
            splits[name] = shuffled[start:end]
            start = end
    return splits


def stratified_split(
    record_ids: list[str],
    labels: list[str],
    ratios: dict[str, float],
    seed: int = 42,
    group_key: list[str] | None = None,
) -> dict[str, list[str]]:
    """Stratified split preserving label distribution."""
    if len(record_ids) != len(labels):
        raise ValueError("record_ids and labels must have the same length")

    # Group by label
    label_groups: dict[str, list[str]] = defaultdict(list)
    for rid, label in zip(record_ids, labels):
        label_groups[label].append(rid)

    # Split each group deterministically
    accumulated: dict[str, list[str]] = defaultdict(list)
    for label, ids in sorted(label_groups.items()):
        group_splits = deterministic_split(ids, ratios, seed=seed, strategy=f"stratified_{label}")
        for split_name, split_ids in group_splits.items():
            accumulated[split_name].extend(split_ids)

    # Shuffle each split for good measure
    result: dict[str, list[str]] = {}
    for split_name, ids in accumulated.items():
        derived = _deterministic_seed(ids, seed, f"shuffle_{split_name}")
        rng = np.random.default_rng(derived)
        shuffled = list(ids)
        rng.shuffle(shuffled)
        result[split_name] = shuffled
    return result


def check_overlap(split_a: list[str], split_b: list[str]) -> list[str]:
    """Return IDs that appear in both splits (should be empty)."""
    return list(set(split_a) & set(split_b))


class PatientStratifiedSplit:
    """Group by patient, stratify by label. Prevents patient leakage."""

    def __init__(
        self,
        ratios: dict[str, float] | None = None,
        seed: int = 42,
        stratify_field: str | None = None,
    ):
        self.ratios = ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
        self.seed = seed
        self.stratify_field = stratify_field

    def split(
        self,
        records: list[dict[str, Any]],
        patient_id_field: str = "patient_id",
        record_id_field: str = "record_id",
    ) -> dict[str, list[str]]:
        # Group records by patient
        patient_to_records: dict[str, list[str]] = defaultdict(list)
        patient_labels: dict[str, str] = {}
        for rec in records:
            pid = str(rec.get(patient_id_field, rec.get(record_id_field, "unknown")))
            rid = str(rec[record_id_field])
            patient_to_records[pid].append(rid)
            if self.stratify_field and self.stratify_field in rec:
                patient_labels[pid] = str(rec[self.stratify_field])

        patient_ids = sorted(patient_to_records.keys())
        labels = [patient_labels.get(pid, "unknown") for pid in patient_ids]

        if self.stratify_field and any(l != "unknown" for l in labels):
            patient_splits = stratified_split(patient_ids, labels, self.ratios, self.seed)
        else:
            patient_splits = deterministic_split(patient_ids, self.ratios, self.seed, "patient")

        result: dict[str, list[str]] = {}
        for split_name, pids in patient_splits.items():
            result[split_name] = []
            for pid in pids:
                result[split_name].extend(patient_to_records[pid])
        return result


class RecordStratifiedSplit:
    """Stratify by label at record level (when no patient IDs available)."""

    def __init__(
        self,
        ratios: dict[str, float] | None = None,
        seed: int = 42,
        stratify_field: str | None = None,
    ):
        self.ratios = ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
        self.seed = seed
        self.stratify_field = stratify_field

    def split(self, records: list[dict[str, Any]], record_id_field: str = "record_id") -> dict[str, list[str]]:
        ids = [str(r[record_id_field]) for r in records]
        if self.stratify_field:
            labels = [str(r.get(self.stratify_field, "unknown")) for r in records]
            return stratified_split(ids, labels, self.ratios, self.seed)
        return deterministic_split(ids, self.ratios, self.seed, "record")


class OfficialSplit:
    """Use predefined official split mapping."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping  # {record_id: split_name}

    def split(self, records: list[dict[str, Any]], record_id_field: str = "record_id") -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for rec in records:
            rid = str(rec[record_id_field])
            split_name = self.mapping.get(rid, "train")
            result[split_name].append(rid)
        return dict(result)


class CustomSplit:
    """User-defined ratios with optional stratification."""

    def __init__(self, ratios: dict[str, float], seed: int = 42, stratify_field: str | None = None):
        total = sum(ratios.values())
        self.ratios = {k: v / total for k, v in ratios.items()}
        self.seed = seed
        self.stratify_field = stratify_field

    def split(self, records: list[dict[str, Any]], record_id_field: str = "record_id") -> dict[str, list[str]]:
        ids = [str(r[record_id_field]) for r in records]
        if self.stratify_field:
            labels = [str(r.get(self.stratify_field, "unknown")) for r in records]
            return stratified_split(ids, labels, self.ratios, self.seed)
        return deterministic_split(ids, self.ratios, self.seed, "custom")
