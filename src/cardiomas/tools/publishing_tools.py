from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def check_hf_repo(repo_id: str, dataset_name: str) -> dict[str, Any]:
    """Check if a dataset already has splits on the HuggingFace repo.
    Returns dict with 'exists' bool and 'metadata' if found."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset")
            target = f"datasets/{dataset_name}/splits.json"
            if target in files:
                content = api.hf_hub_download(repo_id, target, repo_type="dataset")
                with open(content) as f:
                    data = json.load(f)
                return {"exists": True, "metadata": data}
            return {"exists": False}
        except Exception:
            return {"exists": False}
    except Exception as e:
        return {"exists": False, "error": str(e)}


@tool
def push_to_hf(repo_id: str, files: dict[str, str], commit_message: str) -> dict[str, Any]:
    """Push files to a HuggingFace dataset repo.
    files: dict mapping repo_path -> local_path or content string.
    Returns dict with 'url' on success."""
    import tempfile
    import os
    try:
        from huggingface_hub import HfApi
        from cardiomas import config as cfg

        api = HfApi(token=cfg.HF_TOKEN or None)
        uploads = []
        tmp_files = []

        for repo_path, content_or_path in files.items():
            import pathlib
            if pathlib.Path(content_or_path).exists():
                local_path = content_or_path
            else:
                tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                tmp.write(content_or_path)
                tmp.close()
                local_path = tmp.name
                tmp_files.append(local_path)
            uploads.append((local_path, repo_path))

        for local_path, path_in_repo in uploads:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )

        for p in tmp_files:
            os.unlink(p)

        return {"url": f"https://huggingface.co/datasets/{repo_id}", "status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def update_github_readme(repo: str, file_path: str, content: str) -> dict[str, Any]:
    """Update a file in a GitHub repository.
    repo: 'owner/repo', file_path: path in repo, content: new file content."""
    try:
        import base64
        import requests
        from cardiomas import config as cfg

        if not cfg.GITHUB_TOKEN:
            return {"status": "error", "error": "GITHUB_TOKEN not set"}

        api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
        headers = {
            "Authorization": f"Bearer {cfg.GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        }
        # Get current SHA
        r = requests.get(api_url, headers=headers, timeout=15)
        sha = r.json().get("sha", "") if r.status_code == 200 else ""

        payload: dict[str, Any] = {
            "message": f"chore: update {file_path} via CardioMAS",
            "content": base64.b64encode(content.encode()).decode(),
        }
        if sha:
            payload["sha"] = sha

        r2 = requests.put(api_url, json=payload, headers=headers, timeout=15)
        if r2.status_code in (200, 201):
            return {"status": "ok", "url": r2.json().get("content", {}).get("html_url", "")}
        return {"status": "error", "error": r2.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}
