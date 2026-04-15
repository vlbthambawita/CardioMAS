from __future__ import annotations

import base64
import logging
from typing import Any

from cardiomas import config as cfg

logger = logging.getLogger(__name__)


def update_github_page(dataset_name: str, hf_url: str, dry_run: bool = False) -> dict[str, Any]:
    """Append a new dataset entry to the CardioMAS GitHub README."""
    if dry_run:
        logger.info(f"[dry-run] Would update GitHub README for {dataset_name}")
        return {"status": "dry_run"}

    if not cfg.GITHUB_TOKEN:
        return {"status": "error", "error": "GITHUB_TOKEN not set"}

    try:
        import requests

        api_url = f"https://api.github.com/repos/{cfg.GITHUB_REPO}/contents/README.md"
        headers = {
            "Authorization": f"Bearer {cfg.GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        r = requests.get(api_url, headers=headers, timeout=15)
        r.raise_for_status()
        current_content = base64.b64decode(r.json()["content"]).decode()
        sha = r.json()["sha"]

        # Insert/update the dataset table entry
        entry_line = f"| {dataset_name} | [Splits]({hf_url}) |\n"
        marker = "<!-- DATASETS TABLE -->"
        if marker in current_content and dataset_name not in current_content:
            current_content = current_content.replace(marker, marker + "\n" + entry_line)
        elif dataset_name not in current_content:
            current_content += f"\n## Available Datasets\n\n{entry_line}"

        payload = {
            "message": f"docs: add {dataset_name} splits link",
            "content": base64.b64encode(current_content.encode()).decode(),
            "sha": sha,
        }
        r2 = requests.put(api_url, json=payload, headers=headers, timeout=15)
        r2.raise_for_status()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
