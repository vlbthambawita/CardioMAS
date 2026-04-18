from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PersistentMemory:
    """File-backed cross-session memory store.

    Entries are stored in a JSON file at ``store_path``.
    Lookup uses token-overlap similarity (cosine over bag-of-words).
    Max entries are enforced with LRU eviction.
    """

    _FAILURE_PHRASES = (
        "insufficient", "could not produce", "unable to",
        "no information", "cannot answer",
    )

    def __init__(self, store_path: Path, max_entries: int = 200) -> None:
        self._path = store_path
        self._max = max_entries
        self._entries: list[dict[str, Any]] = self._load()
        self._purge_failures()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_similar(self, query: str, threshold: float = 0.70) -> dict[str, Any] | None:
        """Return the most similar past entry if similarity ≥ threshold, else None.

        Only grounded past answers are considered.
        """
        if not self._entries:
            return None
        q_vec = _bow(query)
        best_score = -1.0
        best_entry: dict[str, Any] | None = None
        for entry in self._entries:
            if not entry.get("grounded"):
                continue
            e_vec = _bow(entry.get("query", ""))
            score = _cosine(q_vec, e_vec)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score >= threshold and best_entry is not None:
            best_entry["_similarity"] = round(best_score, 3)
            return best_entry
        return None

    def store(
        self,
        query: str,
        answer: str,
        grounded: bool,
        evidence_ids: list[str] | None = None,
    ) -> None:
        """Persist a query-answer pair. Evicts oldest entry if over capacity."""
        entry: dict[str, Any] = {
            "query": query,
            "answer": answer,
            "grounded": grounded,
            "evidence_ids": evidence_ids or [],
            "stored_at": datetime.now(UTC).isoformat(),
        }
        # Deduplicate: remove existing entry for the same query if present
        self._entries = [e for e in self._entries if e.get("query") != query]
        self._entries.append(entry)
        # LRU eviction: oldest entries are at the front
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]
        self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _purge_failures(self) -> None:
        """Remove stale entries whose answer contains known failure phrases."""
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if not any(p in e.get("answer", "").lower() for p in self._FAILURE_PHRASES)
        ]
        if len(self._entries) < before:
            self._save()

    def _load(self) -> list[dict[str, Any]]:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._entries, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass


def _bow(text: str) -> Counter:
    tokens = [t.lower() for t in text.split() if len(t) > 2]
    return Counter(tokens)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in a if k in b)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_persistent_memory(output_dir: str, max_entries: int = 200) -> PersistentMemory:
    path = Path(output_dir) / "agent_memory.json"
    return PersistentMemory(store_path=path, max_entries=max_entries)
