from __future__ import annotations

from pydantic import BaseModel

from cardiomas.agentic.runtime import AgenticRuntime


class BenchmarkCase(BaseModel):
    query: str
    expected_substrings: list[str]


def run_benchmark(runtime: AgenticRuntime, cases: list[BenchmarkCase]) -> list[dict]:
    results = []
    for case in cases:
        response = runtime.query(case.query)
        passed = all(item.lower() in response.answer.lower() for item in case.expected_substrings)
        results.append({"query": case.query, "passed": passed, "answer": response.answer})
    return results
