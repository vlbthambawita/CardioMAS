from __future__ import annotations

from cardiomas.schemas.config import RuntimeConfig
from cardiomas.schemas.evidence import Citation, EvidenceChunk


def compose_answer(
    query: str,
    config: RuntimeConfig,
    evidence: list[EvidenceChunk],
    aggregate: dict,
    warnings: list[str],
) -> tuple[str, list[Citation]]:
    lines: list[str] = []
    citations: list[Citation] = []

    calculations = aggregate.get("calculations", [])
    if calculations:
        latest = calculations[-1]
        lines.append(f"Calculation result: `{latest['expression']} = {latest['result']}`.")

    dataset_info = aggregate.get("dataset_inspection")
    if dataset_info:
        lines.append(
            "Dataset inspection: "
            f"{dataset_info['total_files']} file(s), "
            f"extensions={dataset_info['extension_counts']}."
        )
        if dataset_info.get("csv_headers"):
            first_name, headers = next(iter(dataset_info["csv_headers"].items()))
            lines.append(f"Sample CSV schema from `{first_name}`: {', '.join(headers) if headers else '(empty)'}.")

    if evidence:
        lines.append("Retrieved evidence:")
        for chunk in evidence[: config.response.max_citations]:
            snippet = _clean_snippet(chunk.content)
            lines.append(f"- {snippet}")
            citations.append(chunk.citation())

    if aggregate.get("web_pages"):
        latest_page = aggregate["web_pages"][-1]
        lines.append(f"Fetched webpage title: {latest_page.get('title', latest_page.get('url', 'web page'))}.")

    if not lines:
        lines.append("I could not produce a grounded answer from the available knowledge sources and tools.")

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")

    answer = "\n".join(lines)
    return answer, citations


def _clean_snippet(text: str, limit: int = 220) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= limit:
        return snippet
    trimmed = snippet[:limit].rsplit(" ", 1)[0]
    return f"{trimmed}..."
