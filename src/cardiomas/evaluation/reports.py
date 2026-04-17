from __future__ import annotations


def summarize_benchmark(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for result in results if result.get("passed"))
    return {"total": total, "passed": passed, "failed": total - passed}
