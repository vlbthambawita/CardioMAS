from __future__ import annotations

import hashlib

from cardiomas.schemas.evidence import EvidenceChunk
from cardiomas.schemas.tools import ToolResult

# Tools whose summaries are too long or binary to be useful as grader evidence.
_SKIP_SYNTHETIC_EVIDENCE = {"generate_python_artifact", "generate_shell_artifact"}


def aggregate_results(results: list[ToolResult]) -> tuple[list[EvidenceChunk], dict]:
    evidence_by_id: dict[str, EvidenceChunk] = {}
    aggregate: dict = {
        "calculations": [],
        "dataset_inspection": None,
        "web_pages": [],
        "generated_python_artifacts": [],
        "generated_shell_artifacts": [],
        "standalone_scripts": [],
    }

    for result in results:
        # Collect real retrieval chunks (from retrieve_corpus)
        for chunk in result.evidence:
            existing = evidence_by_id.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                evidence_by_id[chunk.chunk_id] = chunk

        # For tools that produce a text summary but no EvidenceChunks, create a
        # synthetic chunk so the answer grader and responder have full context.
        if (
            result.ok
            and not result.evidence
            and result.summary
            and result.tool_name not in _SKIP_SYNTHETIC_EVIDENCE
        ):
            content = result.summary[:2000]
            chunk_id = "tool-" + hashlib.md5(
                f"{result.tool_name}:{content}".encode()
            ).hexdigest()[:12]
            if chunk_id not in evidence_by_id:
                evidence_by_id[chunk_id] = EvidenceChunk(
                    chunk_id=chunk_id,
                    source_id=result.tool_name,
                    source_label=result.tool_name,
                    source_type="tool_output",
                    title=f"Output of {result.tool_name}",
                    content=content,
                    uri="",
                    score=0.95,
                )

        if result.tool_name == "calculate" and result.ok:
            aggregate["calculations"].append(result.data)
        elif result.tool_name == "inspect_dataset" and result.ok:
            aggregate["dataset_inspection"] = result.data
        elif result.tool_name in {"fetch_webpage", "read_dataset_website"} and result.ok:
            aggregate["web_pages"].append(result.data)
        elif result.tool_name == "generate_python_artifact" and result.ok:
            if result.data.get("is_standalone"):
                aggregate["standalone_scripts"].append(result.data)
            else:
                aggregate["generated_python_artifacts"].append(result.data)
        elif result.tool_name == "generate_shell_artifact" and result.ok:
            aggregate["generated_shell_artifacts"].append(result.data)

    evidence = sorted(evidence_by_id.values(), key=lambda item: item.score, reverse=True)
    return evidence, aggregate
