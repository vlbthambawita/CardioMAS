from __future__ import annotations

from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.tools import ToolResult


def aggregate_results(results: list[ToolResult]) -> tuple[list[EvidenceChunk], dict]:
    evidence_by_id: dict[str, EvidenceChunk] = {}
    aggregate: dict = {
        "calculations": [],
        "dataset_inspection": None,
        "web_pages": [],
        "dataset_statistics": None,
        "file_reads": [],
        "generated_scripts": [],
    }

    for result in results:
        for chunk in result.evidence:
            existing = evidence_by_id.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                evidence_by_id[chunk.chunk_id] = chunk

        if result.tool_name == "calculate" and result.ok:
            aggregate["calculations"].append(result.data)
        elif result.tool_name == "inspect_dataset" and result.ok:
            aggregate["dataset_inspection"] = result.data
        elif result.tool_name == "fetch_webpage" and result.ok:
            aggregate["web_pages"].append(result.data)
        elif result.tool_name == "dataset_statistics" and result.ok:
            aggregate["dataset_statistics"] = result.data
        elif result.tool_name == "read_dataset_file" and result.ok:
            aggregate["file_reads"].append(result.data)
        elif result.tool_name == "generate_shell_script" and result.ok:
            aggregate["generated_scripts"].append(result.data)

    evidence = sorted(evidence_by_id.values(), key=lambda item: item.score, reverse=True)
    return evidence, aggregate
