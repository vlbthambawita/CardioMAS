from __future__ import annotations

import json
import re
from pathlib import Path

from cardiomas.knowledge_department.models import DatasetKnowledgeBundle


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "dataset"


def write_knowledge_bundle(bundle: DatasetKnowledgeBundle, output_root: str) -> dict[str, str]:
    base = Path(output_root) / "knowledge" / "datasets" / bundle.slug
    base.mkdir(parents=True, exist_ok=True)

    overview_path = base / "overview.json"
    references_path = base / "references.json"
    notes_path = base / "notes.md"
    schema_path = base / "schema.json"

    overview = bundle.overview or _build_overview(bundle)
    references = bundle.references or _build_references(bundle)
    schema = bundle.schema_definition or _build_schema(bundle)
    notes = bundle.notes_markdown or _build_notes(bundle)

    overview_path.write_text(json.dumps(overview, indent=2), encoding="utf-8")
    references_path.write_text(json.dumps(references, indent=2), encoding="utf-8")
    notes_path.write_text(notes, encoding="utf-8")
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    return {
        "overview.json": str(overview_path),
        "references.json": str(references_path),
        "notes.md": str(notes_path),
        "schema.json": str(schema_path),
    }


def _build_overview(bundle: DatasetKnowledgeBundle) -> dict:
    return {
        "dataset_name": bundle.dataset_name,
        "source_count": len(bundle.pages),
        "source_urls": [page.url for page in bundle.pages],
        "pages": [
            {
                "url": page.url,
                "title": page.title,
                "description": page.description,
                "headings": page.headings[:5],
                "metadata": page.metadata,
                "provenance": page.provenance.model_dump(mode="json"),
            }
            for page in bundle.pages
        ],
    }


def _build_references(bundle: DatasetKnowledgeBundle) -> list[dict]:
    return [
        {
            "url": page.url,
            "title": page.title,
            "headings": page.headings[:10],
            "links": page.links[:20],
            "tables": [table.model_dump(mode="json") for table in page.tables],
            "provenance": page.provenance.model_dump(mode="json"),
        }
        for page in bundle.pages
    ]


def _build_schema(bundle: DatasetKnowledgeBundle) -> dict:
    return {
        "artifact_schema_version": 1,
        "page_fields": [
            "url",
            "title",
            "description",
            "headings",
            "content_blocks",
            "tables",
            "links",
            "metadata",
            "text_excerpt",
            "provenance",
        ],
        "dataset_name": bundle.dataset_name,
        "slug": bundle.slug,
    }


def _build_notes(bundle: DatasetKnowledgeBundle) -> str:
    lines = [f"# Knowledge Notes: {bundle.dataset_name}", ""]
    for page in bundle.pages:
        lines.extend(
            [
                f"## {page.title or page.url}",
                f"- Source: {page.url}",
                f"- Fetch method: {page.provenance.fetch_method}",
                f"- Status: {page.provenance.status}",
            ]
        )
        if page.description:
            lines.append(f"- Description: {page.description}")
        if page.headings:
            lines.append(f"- Headings: {', '.join(page.headings[:5])}")
        if page.content_blocks:
            lines.append("")
            lines.append(page.content_blocks[0])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
