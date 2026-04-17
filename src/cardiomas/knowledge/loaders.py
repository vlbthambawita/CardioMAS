from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from cardiomas.schemas.config import KnowledgeSource
from cardiomas.schemas.evidence import KnowledgeDocument


DEFAULT_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".pdf"}


def load_source(source: KnowledgeSource) -> list[KnowledgeDocument]:
    if source.kind in {"local_dir", "dataset_dir"}:
        return _load_directory(source)
    if source.kind == "local_file":
        return [_load_file_document(source, Path(source.path or ""))]
    if source.kind == "pdf":
        if source.path:
            return [_load_pdf_document(source, Path(source.path))]
        return [_load_remote_pdf_document(source, source.url or "")]
    if source.kind == "web_page":
        return [_load_web_document(source, source.url or "")]
    raise ValueError(f"Unsupported source kind: {source.kind}")


def _load_directory(source: KnowledgeSource) -> list[KnowledgeDocument]:
    root = Path(source.path or "")
    paths = root.rglob("*") if source.recursive else root.iterdir()
    docs: list[KnowledgeDocument] = []
    allowed = {ext.lower() for ext in source.include_extensions} or DEFAULT_EXTENSIONS
    for path in sorted(paths):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        if path.suffix.lower() == ".pdf":
            docs.append(_load_pdf_document(source, path))
        else:
            docs.append(_load_file_document(source, path))
    return docs


def _load_file_document(source: KnowledgeSource, path: Path) -> KnowledgeDocument:
    if path.suffix.lower() == ".csv":
        text = _csv_as_text(path)
    elif path.suffix.lower() == ".json":
        text = json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return KnowledgeDocument(
        doc_id=f"{source.id}:{path.name}",
        source_id=source.id,
        source_label=source.label,
        source_type=source.kind,
        uri=str(path),
        title=path.name,
        content=text,
        metadata={"path": str(path)},
    )


def _load_pdf_document(source: KnowledgeSource, path: Path) -> KnowledgeDocument:
    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return KnowledgeDocument(
        doc_id=f"{source.id}:{path.name}",
        source_id=source.id,
        source_label=source.label,
        source_type="pdf",
        uri=str(path),
        title=path.name,
        content=text,
        metadata={"path": str(path)},
    )


def _load_remote_pdf_document(source: KnowledgeSource, url: str) -> KnowledgeDocument:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    pdf_path = Path(source.metadata.get("download_path", "downloaded.pdf"))
    pdf_path.write_bytes(response.content)
    return _load_pdf_document(source, pdf_path)


def _load_web_document(source: KnowledgeSource, url: str) -> KnowledgeDocument:
    response = requests.get(url, timeout=30, headers={"User-Agent": "CardioMAS/2.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "iframe"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else url
    text = " ".join(soup.get_text(separator=" ").split())
    return KnowledgeDocument(
        doc_id=f"{source.id}:{_safe_title(title)}",
        source_id=source.id,
        source_label=source.label,
        source_type="web_page",
        uri=url,
        title=title,
        content=text,
        metadata={"url": url},
    )


def _csv_as_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(_take(reader, 25))
    return "\n".join(", ".join(cell for cell in row) for row in rows)


def _take(items: Iterable[list[str]], limit: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in items:
        rows.append(item)
        if len(rows) >= limit:
            break
    return rows


def _safe_title(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-") or "web"
