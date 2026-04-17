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


DEFAULT_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".pdf", ".html"}
DATASET_SKIP_FILENAMES = {"license.txt", "sha256sums.txt"}


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
        if source.kind == "dataset_dir" and path.name.lower() in DATASET_SKIP_FILENAMES:
            continue
        if source.kind == "dataset_dir" and path.suffix.lower() == ".html" and len(path.relative_to(root).parts) > 2:
            continue
        if path.suffix.lower() not in allowed:
            continue
        if path.suffix.lower() == ".pdf":
            docs.append(_load_pdf_document(source, path))
        elif path.suffix.lower() == ".html":
            docs.append(_load_local_html_document(source, path, root))
        else:
            docs.append(_load_file_document(source, path, root=root))
    return docs


def _load_file_document(source: KnowledgeSource, path: Path, root: Path | None = None) -> KnowledgeDocument:
    relative_path = str(path.relative_to(root)) if root is not None else path.name
    if path.suffix.lower() == ".csv":
        text = _csv_as_text(path, relative_path)
    elif path.suffix.lower() == ".json":
        text = f"JSON file: {relative_path}\n" + json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)
    else:
        text = f"File: {relative_path}\n" + path.read_text(encoding="utf-8", errors="ignore")
    return KnowledgeDocument(
        doc_id=f"{source.id}:{relative_path}",
        source_id=source.id,
        source_label=source.label,
        source_type=source.kind,
        uri=str(path),
        title=relative_path,
        content=text,
        metadata={"path": str(path), "relative_path": relative_path},
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


def _load_local_html_document(source: KnowledgeSource, path: Path, root: Path) -> KnowledgeDocument:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "iframe"]):
        tag.decompose()
    relative_path = str(path.relative_to(root))
    title = soup.title.get_text(strip=True) if soup.title else relative_path
    text = " ".join(soup.get_text(separator=" ").split())
    return KnowledgeDocument(
        doc_id=f"{source.id}:{relative_path}",
        source_id=source.id,
        source_label=source.label,
        source_type=source.kind,
        uri=str(path),
        title=relative_path,
        content=f"HTML file: {relative_path}\nTitle: {title}\n{text}",
        metadata={"path": str(path), "relative_path": relative_path, "html_title": title},
    )


def _csv_as_text(path: Path, relative_path: str) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(_take(reader, 25))
    headers = rows[0] if rows else []
    body = "\n".join(", ".join(cell for cell in row) for row in rows[1:])
    return (
        f"CSV file: {relative_path}\n"
        f"Columns: {', '.join(headers) if headers else '(none)'}\n"
        f"Sample rows:\n{body}"
    ).strip()


def _take(items: Iterable[list[str]], limit: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in items:
        rows.append(item)
        if len(rows) >= limit:
            break
    return rows


def _safe_title(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-") or "web"
