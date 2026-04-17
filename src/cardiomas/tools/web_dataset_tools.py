"""Tool for reading and understanding dataset documentation websites.

Dataset documentation pages (PhysioNet, HuggingFace, Kaggle, Zenodo, etc.)
follow predictable patterns: a description section, a files listing, a
variables/signals table, a license statement, and a citation block. This
module fetches such pages and extracts those sections so the agent receives
concise, structured information rather than a raw HTML dump.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from cardiomas.schemas.tools import ToolResult

# Headings whose content is especially relevant for dataset understanding.
_DATASET_HEADINGS = {
    "description", "abstract", "overview", "background", "about",
    "files", "data files", "file listing", "file structure", "directory",
    "signals", "signal", "leads", "channels",
    "variables", "columns", "fields", "attributes", "features",
    "annotations", "labels", "annotation", "label",
    "format", "data format", "file format",
    "usage", "how to use", "getting started",
    "license", "terms", "access policy",
    "citation", "cite", "references", "how to cite",
    "dataset", "data collection", "cohort", "population",
    "sampling frequency", "sampling rate", "frequency",
    "statistics", "summary statistics",
}

# Inline phrases that carry key dataset facts — extracted into a separate dict.
_KEYWORD_PATTERNS = {
    "sampling_rate":  re.compile(r"(\d+(?:\.\d+)?)\s*Hz", re.I),
    "n_records":      re.compile(r"(\d[\d,]*)\s+(?:records?|subjects?|patients?|participants?|recordings?|waveforms?)", re.I),
    "n_leads":        re.compile(r"(\d+)[\s-]*(?:lead|channel|signal)s?\b", re.I),
    "duration":       re.compile(r"(\d+(?:\.\d+)?)\s*(?:second|minute|hour)s?\b", re.I),
    "doi":            re.compile(r"(?:doi|DOI)[:\s]+([^\s,;\"<>]+)", re.I),
    "license":        re.compile(r"(CC[\s\-](?:BY|BY-SA|BY-NC|BY-NC-SA|BY-ND|BY-NC-ND)(?:[\s\-]\d+\.\d+)?|Open\s+Data\s+Commons\s+\w+|PhysioNet\s+Restricted\s+Health\s+Data)", re.I),
}

_MAX_SECTION_CHARS = 600   # per heading section
_MAX_TABLE_ROWS    = 25
_MAX_TABLES        = 4
_MAX_TOTAL_CHARS   = 6000  # guard for summary length


def read_dataset_website(
    url: str,
    config: Any | None = None,
) -> ToolResult:
    """Fetch a dataset documentation website and extract structured metadata.

    This tool is designed specifically for dataset documentation pages such as
    PhysioNet records, HuggingFace dataset cards, Zenodo deposits, Kaggle
    dataset pages, and any similar page that describes a medical or research
    dataset. It goes beyond a generic webpage fetch by:

    - Extracting named sections (Description, Files, Signals/Leads, Variables,
      Format, Annotations, License, Citation) and returning their text.
    - Parsing HTML tables that typically contain file listings, variable
      dictionaries, or signal metadata.
    - Detecting dataset-specific facts inline: sampling frequency, number of
      records/subjects, number of leads, recording duration, DOI, and license.
    - Returning both a human-readable summary and a structured ``data`` dict
      that downstream tools or the code-generation pipeline can use directly.

    When to use this tool:
    - Before writing analysis code, call this to learn what files the dataset
      has, what columns or signals are available, and what the data format is.
    - When the user asks about a dataset's description, cohort, or methodology.
    - When you need the citation, DOI, or license for a dataset.
    - When ``list_folder_structure`` or ``read_wfdb_dataset`` did not give
      enough context about what the data represents.

    The dataset website URLs are configured in the YAML under ``sources`` with
    ``kind: web_page``. You can also call this tool with any URL the user
    mentions in their question.

    Args:
        url: Full URL of the dataset documentation page to fetch and parse.
             Must start with ``http://`` or ``https://``.

    Returns:
        ToolResult whose ``summary`` contains a compact, section-by-section
        report and whose ``data`` dict includes:
        - ``url`` — the fetched URL
        - ``title`` — page title
        - ``meta_description`` — HTML meta description (if present)
        - ``sections`` — dict mapping heading text → extracted paragraph text
        - ``tables`` — list of tables (each a list-of-lists of cell strings)
        - ``key_facts`` — dict of auto-detected dataset facts (sampling rate,
          record count, lead count, duration, DOI, license)
        - ``links`` — list of links that look like data file downloads
    """
    if not url or not url.startswith(("http://", "https://")):
        return ToolResult(
            tool_name="read_dataset_website",
            ok=False,
            summary="",
            error=(
                f"Invalid URL: {url!r}. "
                "Provide a full URL starting with http:// or https://."
            ),
        )

    # ── Fetch ────────────────────────────────────────────────────────────────
    try:
        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; CardioMAS/2.0; "
                    "+https://github.com/vlbthambawita/CardioMAS)"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return ToolResult(
            tool_name="read_dataset_website",
            ok=False,
            summary="",
            error=f"Failed to fetch {url}: {exc}",
        )

    # ── Parse HTML ───────────────────────────────────────────────────────────
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "iframe", "aside",
                     "header", "form", "button", "noscript"]):
        tag.decompose()

    # ── Title and meta ───────────────────────────────────────────────────────
    title = soup.title.get_text(strip=True) if soup.title else url
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if isinstance(meta_tag, Tag):
        meta_desc = str(meta_tag.get("content", "")).strip()

    # ── Sections by heading ──────────────────────────────────────────────────
    sections: dict[str, str] = {}
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        heading_text = heading.get_text(strip=True)
        heading_lower = heading_text.lower().strip(":.!? ")
        if not any(kw in heading_lower for kw in _DATASET_HEADINGS):
            continue
        content_parts: list[str] = []
        for sibling in heading.next_siblings:
            if not isinstance(sibling, Tag):
                continue
            if sibling.name in ("h1", "h2", "h3", "h4"):
                break
            text = sibling.get_text(separator=" ", strip=True)
            if text:
                content_parts.append(text)
            if sum(len(p) for p in content_parts) >= _MAX_SECTION_CHARS:
                break
        body = " ".join(content_parts)[:_MAX_SECTION_CHARS]
        if body:
            sections[heading_text] = body

    # ── Tables ───────────────────────────────────────────────────────────────
    tables: list[list[list[str]]] = []
    for table in soup.find_all("table")[: _MAX_TABLES]:
        rows: list[list[str]] = []
        for row in table.find_all("tr")[: _MAX_TABLE_ROWS]:
            cells = [
                cell.get_text(separator=" ", strip=True)
                for cell in row.find_all(["td", "th"])
            ]
            if any(cells):
                rows.append(cells)
        if rows:
            tables.append(rows)

    # ── Key dataset facts (regex over full text) ─────────────────────────────
    full_text = soup.get_text(separator=" ")
    full_text_clean = " ".join(full_text.split())

    key_facts: dict[str, list[str]] = {}
    for fact_name, pattern in _KEYWORD_PATTERNS.items():
        matches = pattern.findall(full_text_clean)
        if matches:
            # Deduplicate while preserving order
            seen: set[str] = set()
            deduped = [m if isinstance(m, str) else m for m in matches]
            unique = []
            for m in deduped:
                key = m.strip()
                if key not in seen:
                    seen.add(key)
                    unique.append(key)
            key_facts[fact_name] = unique[:6]

    # ── Download / data file links ───────────────────────────────────────────
    data_extensions = {".zip", ".tar", ".gz", ".csv", ".tsv", ".mat",
                       ".hdf5", ".h5", ".dat", ".edf", ".wfdb", ".json",
                       ".parquet", ".feather", ".npz", ".npy"}
    data_links: list[dict[str, str]] = []
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    for a_tag in soup.find_all("a", href=True):
        href = str(a_tag["href"]).strip()
        ext = "." + href.rsplit(".", 1)[-1].lower() if "." in href else ""
        label = a_tag.get_text(strip=True)
        if ext in data_extensions or any(kw in href.lower() for kw in ["download", "/files/", "/data/"]):
            abs_href = urljoin(base, href)
            if abs_href not in {lnk["href"] for lnk in data_links}:
                data_links.append({"href": abs_href, "label": label[:80]})
        if len(data_links) >= 20:
            break

    # ── Build human-readable summary ─────────────────────────────────────────
    lines: list[str] = [f"Dataset website: {url}", f"Title: {title}"]

    if meta_desc:
        lines.append(f"Description: {meta_desc[:300]}")

    if key_facts:
        lines.append("\nKey facts detected:")
        for fact, values in key_facts.items():
            lines.append(f"  {fact}: {', '.join(values)}")

    if sections:
        lines.append("\nDocumentation sections:")
        for heading, body in sections.items():
            lines.append(f"\n[{heading}]")
            lines.append(body[:_MAX_SECTION_CHARS])

    if tables:
        lines.append(f"\nTables found: {len(tables)}")
        for i, rows in enumerate(tables[:2]):
            lines.append(f"\nTable {i + 1} (first {min(5, len(rows))} rows):")
            for row in rows[:5]:
                lines.append("  | " + " | ".join(row) + " |")

    if data_links:
        lines.append(f"\nData/download links ({len(data_links)} found):")
        for lnk in data_links[:8]:
            lines.append(f"  {lnk['label'] or '(no label)'}: {lnk['href']}")

    summary = "\n".join(lines)
    if len(summary) > _MAX_TOTAL_CHARS:
        summary = summary[:_MAX_TOTAL_CHARS] + "\n... (truncated)"

    return ToolResult(
        tool_name="read_dataset_website",
        ok=True,
        summary=summary,
        data={
            "url": url,
            "title": title,
            "meta_description": meta_desc,
            "sections": sections,
            "tables": tables,
            "key_facts": key_facts,
            "links": data_links,
        },
    )


def _configured_web_urls(config: Any) -> list[tuple[str, str]]:
    """Return (label, url) pairs for all web_page sources in config."""
    result = []
    if config is None:
        return result
    for source in getattr(config, "sources", []):
        if getattr(source, "kind", "") == "web_page" and source.url:
            result.append((source.label or source.url, source.url))
    return result
