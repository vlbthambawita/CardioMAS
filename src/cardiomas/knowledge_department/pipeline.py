from __future__ import annotations

from cardiomas.knowledge_department.models import DatasetKnowledgeBundle, PageKnowledge
from cardiomas.knowledge_department.scraper import fetch_html, parse_page
from cardiomas.knowledge_department.writer import slugify, write_knowledge_bundle
from cardiomas.organization.base import DepartmentHead, WorkerAgent
from cardiomas.shared.messages import ArtifactRef, ArtifactStatus, DepartmentReport, TaskMessage


def fetch_and_parse_page(url: str, rate_limit_seconds: float = 0.0) -> PageKnowledge:
    html, provenance = fetch_html(url, rate_limit_seconds=rate_limit_seconds)
    return parse_page(url, html, provenance)


def build_knowledge_bundle(dataset_name: str, urls: list[str], output_root: str) -> tuple[DatasetKnowledgeBundle, dict[str, str], list[str]]:
    pages: list[PageKnowledge] = []
    notes: list[str] = []
    for url in urls:
        page = fetch_and_parse_page(url, rate_limit_seconds=0.2)
        pages.append(page)
        if page.provenance.status != "ok":
            notes.append(f"{url}: {page.provenance.error or page.provenance.status}")

    bundle = DatasetKnowledgeBundle(
        dataset_name=dataset_name,
        slug=slugify(dataset_name),
        pages=pages,
    )
    output_paths = write_knowledge_bundle(bundle, output_root)
    return bundle, output_paths, notes


class KnowledgeWorker(WorkerAgent):
    def __init__(self) -> None:
        super().__init__(name="knowledge_worker", department_name="knowledge")

    def execute(self, message: TaskMessage) -> DepartmentReport:
        dataset_name = message.inputs["dataset_name"]
        urls = list(message.inputs.get("knowledge_urls", []))
        output_root = message.inputs["output_root"]

        bundle, output_paths, notes = build_knowledge_bundle(dataset_name, urls, output_root)

        artifacts = [
            ArtifactRef(
                name="knowledge_overview",
                path=output_paths["overview.json"],
                artifact_type="json",
                summary=f"Knowledge overview for {dataset_name}",
                status=ArtifactStatus.complete,
                provenance=urls,
            ),
            ArtifactRef(
                name="knowledge_references",
                path=output_paths["references.json"],
                artifact_type="json",
                summary=f"Source references for {dataset_name}",
                status=ArtifactStatus.complete,
                provenance=urls,
            ),
            ArtifactRef(
                name="knowledge_notes",
                path=output_paths["notes.md"],
                artifact_type="markdown",
                summary=f"Human-readable notes for {dataset_name}",
                status=ArtifactStatus.complete,
                provenance=urls,
            ),
            ArtifactRef(
                name="knowledge_schema",
                path=output_paths["schema.json"],
                artifact_type="json",
                summary=f"Artifact schema for {dataset_name}",
                status=ArtifactStatus.complete,
                provenance=urls,
            ),
        ]

        return DepartmentReport(
            department="knowledge",
            summary=f"Collected knowledge from {len(bundle.pages)} source page(s).",
            artifacts=artifacts,
            notes=notes,
        )


class KnowledgeDepartmentHead(DepartmentHead):
    def __init__(self) -> None:
        super().__init__(
            name="knowledge_head",
            department_name="knowledge",
            workers=[KnowledgeWorker()],
            input_contract="dataset_name, knowledge_urls[], output_root",
            output_contract="overview.json, references.json, notes.md, schema.json",
        )
