from __future__ import annotations

import logging
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from cardiomas.knowledge_department.models import ExtractedTable, PageKnowledge, SourceProvenance

logger = logging.getLogger(__name__)


def _can_fetch(url: str, user_agent: str = "CardioMASBot") -> tuple[bool, str]:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    try:
        parser.set_url(robots_url)
        parser.read()
        return parser.can_fetch(user_agent, url), ""
    except Exception as exc:
        return True, f"robots.txt unavailable: {exc}"


def fetch_html(
    url: str,
    timeout_seconds: float = 20.0,
    rate_limit_seconds: float = 0.0,
    respect_robots: bool = True,
) -> tuple[str, SourceProvenance]:
    allowed = True
    robots_note = ""
    if respect_robots:
        allowed, robots_note = _can_fetch(url)

    provenance = SourceProvenance(
        url=url,
        robots_allowed=allowed,
        rate_limit_seconds=rate_limit_seconds,
    )

    if not allowed:
        provenance.status = "blocked"
        provenance.error = "Blocked by robots.txt"
        return "", provenance

    if rate_limit_seconds > 0:
        time.sleep(rate_limit_seconds)

    scrapling_error = ""
    try:
        from scrapling.fetchers import Fetcher

        response = Fetcher.fetch(
            url,
            timeout=int(timeout_seconds * 1000),
            disable_resources=True,
            google_search=False,
        )
        html = getattr(response, "text", "")
        if callable(html):
            html = html()
        if not isinstance(html, str):
            html = str(html)
        provenance.fetch_method = "scrapling"
        provenance.error = robots_note
        return html, provenance
    except Exception as exc:
        scrapling_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Scrapling fetch failed for %s: %s", url, exc)

    try:
        response = requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": "CardioMASBot/1.0 (+https://github.com/vlbthambawita/CardioMAS)"},
        )
        response.raise_for_status()
        provenance.fetch_method = "requests"
        provenance.error = robots_note or scrapling_error
        return response.text, provenance
    except Exception as exc:
        provenance.fetch_method = "requests"
        provenance.status = "error"
        provenance.error = robots_note or f"{scrapling_error}; {type(exc).__name__}: {exc}"
        return "", provenance


def parse_page(url: str, html: str, provenance: SourceProvenance) -> PageKnowledge:
    soup = BeautifulSoup(html or "", "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    description = ""
    description_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    if description_tag and description_tag.get("content"):
        description = description_tag["content"].strip()

    headings = _unique_texts([tag.get_text(" ", strip=True) for tag in soup.find_all(["h1", "h2", "h3"])])
    content_blocks = _extract_content_blocks(soup)
    tables = _extract_tables(soup)
    links = _unique_texts([urljoin(url, tag["href"]) for tag in soup.find_all("a", href=True)])
    metadata = {
        "title": title,
        "description": description,
        "canonical": (
            soup.find("link", attrs={"rel": "canonical"}).get("href", "")
            if soup.find("link", attrs={"rel": "canonical"})
            else ""
        ),
    }
    text_excerpt = " ".join(content_blocks).strip()[:4000]

    return PageKnowledge(
        url=url,
        title=title,
        description=description,
        headings=headings[:20],
        content_blocks=content_blocks[:40],
        tables=tables[:5],
        links=links[:50],
        metadata={key: value for key, value in metadata.items() if value},
        text_excerpt=text_excerpt,
        provenance=provenance,
    )


def _extract_content_blocks(soup: BeautifulSoup) -> list[str]:
    selectors = [
        "main p",
        "article p",
        "section p",
        "p",
        "li",
    ]
    blocks: list[str] = []
    for selector in selectors:
        for element in soup.select(selector):
            text = element.get_text(" ", strip=True)
            if len(text) >= 30:
                blocks.append(text)
        if blocks:
            break

    if not blocks and soup.body:
        blocks = [line.strip() for line in soup.body.get_text("\n", strip=True).splitlines() if len(line.strip()) >= 30]

    return _unique_texts(blocks)


def _extract_tables(soup: BeautifulSoup) -> list[ExtractedTable]:
    extracted: list[ExtractedTable] = []
    for table in soup.find_all("table"):
        headers = [cell.get_text(" ", strip=True) for cell in table.find_all("th")]
        rows: list[list[str]] = []
        for row in table.find_all("tr")[1:6]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if headers or rows:
            extracted.append(ExtractedTable(headers=headers, rows=rows))
    return extracted


def _unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = " ".join(value.split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique
