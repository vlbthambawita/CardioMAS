from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from cardiomas import config as cfg
from cardiomas.schemas.split import SplitManifest

logger = logging.getLogger(__name__)


def publish_to_hf(manifest: SplitManifest, dry_run: bool = False) -> dict[str, Any]:
    """Push split manifest and analysis report to vlbthambawita/ECGBench."""
    if dry_run:
        logger.info(f"[dry-run] Would publish {manifest.dataset_name} to {cfg.HF_REPO_ID}")
        return {"status": "dry_run", "dataset": manifest.dataset_name}

    if not cfg.HF_TOKEN:
        return {"status": "error", "error": "HF_TOKEN not set. Run: export HF_TOKEN=<your_token>"}

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=cfg.HF_TOKEN)
        base_path = f"datasets/{manifest.dataset_name}"

        # Build files to upload
        splits_data = {
            "dataset_name": manifest.dataset_name,
            "split_version": manifest.split_version,
            "author": manifest.author,
            "description": manifest.description,
            "cardiomas_version": manifest.cardiomas_version,
            "reproducibility_config": manifest.reproducibility_config.model_dump(mode="json"),
            "splits": manifest.splits,
            "split_stats": manifest.split_stats,
        }
        meta_data = manifest.reproducibility_config.model_dump(mode="json")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            splits_file = tmp / "splits.json"
            meta_file = tmp / "split_metadata.json"
            splits_file.write_text(json.dumps(splits_data, indent=2))
            meta_file.write_text(json.dumps(meta_data, indent=2))

            api.upload_file(
                path_or_fileobj=str(splits_file),
                path_in_repo=f"{base_path}/splits.json",
                repo_id=cfg.HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"add splits for {manifest.dataset_name} (CardioMAS v{manifest.cardiomas_version})",
            )
            api.upload_file(
                path_or_fileobj=str(meta_file),
                path_in_repo=f"{base_path}/split_metadata.json",
                repo_id=cfg.HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"add split metadata for {manifest.dataset_name}",
            )

        url = f"https://huggingface.co/datasets/{cfg.HF_REPO_ID}/tree/main/{base_path}"
        logger.info(f"Published {manifest.dataset_name} to {url}")
        return {"status": "ok", "url": url, "dataset": manifest.dataset_name}

    except Exception as e:
        logger.error(f"HF publish failed: {e}")
        return {"status": "error", "error": str(e)}
