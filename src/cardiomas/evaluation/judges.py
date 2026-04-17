from __future__ import annotations


def contains_all(answer: str, expected_substrings: list[str]) -> bool:
    lower = answer.lower()
    return all(item.lower() in lower for item in expected_substrings)
