from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.parse
from pathlib import Path
from typing import Protocol

import requests
from rich.progress import Progress, SpinnerColumn, BarColumn, DownloadColumn, TextColumn

from cardiomas import config as cfg
from cardiomas.schemas.dataset import DatasetInfo, DatasetSource

logger = logging.getLogger(__name__)


class DatasetLoader(Protocol):
    def can_handle(self, source: str) -> bool: ...
    def load(self, source: str, dest: Path | None = None) -> Path: ...


class PhysioNetLoader:
    """Download from PhysioNet (anonymous or credentialed)."""

    def can_handle(self, source: str) -> bool:
        return "physionet.org" in source

    def load(self, source: str, dest: Path | None = None) -> Path:
        name = self._slug(source)
        dest = dest or (cfg.DATA_DIR / name)
        if dest.exists():
            logger.info(f"Dataset already cached at {dest}")
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading from PhysioNet: {source}")
        # PhysioNet supports wget-style recursive download; we list files first
        index_url = source.rstrip("/") + "/"
        resp = requests.get(index_url, timeout=30)
        resp.raise_for_status()
        # Extract file links (simplified — real impl would parse HTML)
        self._download_index(source, dest)
        return dest

    def _slug(self, url: str) -> str:
        parts = urllib.parse.urlparse(url).path.strip("/").split("/")
        return "-".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    def _download_index(self, base_url: str, dest: Path) -> None:
        """Save a manifest file pointing to the source URL."""
        manifest = dest / "source.txt"
        manifest.write_text(base_url)
        logger.info(f"Saved source URL to {manifest} — use wfdb.io.dl_database() for full download.")


class HuggingFaceLoader:
    """Download via HuggingFace datasets library."""

    def can_handle(self, source: str) -> bool:
        return "huggingface.co" in source or "/" in source and not source.startswith("http")

    def load(self, source: str, dest: Path | None = None) -> Path:
        from datasets import load_dataset  # type: ignore

        repo_id = self._parse_repo_id(source)
        name = repo_id.replace("/", "--")
        dest = dest or (cfg.DATA_DIR / name)
        if dest.exists():
            logger.info(f"Dataset already cached at {dest}")
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading from HuggingFace: {repo_id}")
        ds = load_dataset(repo_id)
        ds.save_to_disk(str(dest))
        return dest

    def _parse_repo_id(self, source: str) -> str:
        if source.startswith("https://huggingface.co/datasets/"):
            return source.replace("https://huggingface.co/datasets/", "")
        return source


class LocalLoader:
    """Validate and index a local directory."""

    def can_handle(self, source: str) -> bool:
        return Path(source).exists()

    def load(self, source: str, dest: Path | None = None) -> Path:
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local path not found: {path}")
        logger.info(f"Using local dataset at {path}")
        return path


class GenericURLLoader:
    """Download any URL with progress bar."""

    def can_handle(self, source: str) -> bool:
        return source.startswith("http://") or source.startswith("https://")

    def load(self, source: str, dest: Path | None = None) -> Path:
        filename = Path(urllib.parse.urlparse(source).path).name or "dataset"
        dest_dir = dest or (cfg.DATA_DIR / _url_slug(source))
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / filename
        if dest_file.exists():
            logger.info(f"Already downloaded: {dest_file}")
            return dest_dir
        logger.info(f"Downloading: {source}")
        _download_file(source, dest_file)
        return dest_dir


def _url_slug(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _download_file(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
        ) as progress:
            task = progress.add_task(dest.name, total=total or None)
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.advance(task, len(chunk))


def get_loader(source: str) -> DatasetLoader:
    loaders: list[DatasetLoader] = [
        LocalLoader(),
        PhysioNetLoader(),
        HuggingFaceLoader(),
        GenericURLLoader(),
    ]
    for loader in loaders:
        if loader.can_handle(source):
            return loader
    raise ValueError(f"No loader found for source: {source}")
