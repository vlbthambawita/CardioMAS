from __future__ import annotations

from typing import Any

from cardiomas.schemas.state import UserOptions


class CardioMAS:
    """High-level Python API for CardioMAS."""

    def __init__(
        self,
        ollama_model: str = "llama3.1:8b",
        seed: int = 42,
        use_cloud_llm: bool = False,
        data_dir: str = "",
        agent_llms: dict[str, str] | None = None,
    ) -> None:
        import cardiomas.config as cfg
        import os

        if ollama_model:
            os.environ["OLLAMA_MODEL"] = ollama_model
            cfg.OLLAMA_MODEL = ollama_model
        if data_dir:
            from pathlib import Path
            cfg.DATA_DIR = Path(data_dir).expanduser()
            cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Apply per-agent LLM map to config
        if agent_llms:
            for agent, model in agent_llms.items():
                if agent != "default":
                    cfg.set_agent_llm(agent, model)
            if "default" in agent_llms:
                cfg.OLLAMA_MODEL = agent_llms["default"]

        self.seed = seed
        self.use_cloud_llm = use_cloud_llm
        self._agent_llms: dict[str, str] = agent_llms or {}

    def analyze(
        self,
        source: str,
        output_dir: str = "output",
        force: bool = False,
        use_cloud_llm: bool | None = None,
        custom_split: dict[str, float] | None = None,
        ignore_official: bool = False,
        stratify_by: str | None = None,
        push_to_hf: bool = False,
        local_path: str | None = None,
        requirement: str | None = None,
        agent_llms: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run full pipeline on a dataset source. Saves outputs locally.

        Args:
            source: URL or local path to the ECG dataset.
            output_dir: Directory for local output files.
            force: Re-run even if already analyzed.
            use_cloud_llm: Use cloud LLM instead of local Ollama.
            custom_split: e.g. {"train": 0.7, "val": 0.15, "test": 0.15}
            ignore_official: Ignore official splits from the paper.
            stratify_by: Metadata field to stratify by.
            push_to_hf: Also publish to HuggingFace (requires HF_TOKEN).
            local_path: Explicit local data directory (skips download).
            requirement: Natural language requirement string (V2).
            agent_llms: Per-agent model overrides, e.g. {"coder": "deepseek-coder:6.7b"}.

        Returns:
            dict with pipeline state fields (splits, local_output_dir, errors, …).
        """
        from cardiomas.graph.workflow import run_pipeline

        # Merge instance-level and call-level agent_llm maps
        merged_llms = {**self._agent_llms, **(agent_llms or {})}

        options = UserOptions(
            dataset_source=source,
            local_path=local_path,
            output_dir=output_dir,
            force_reanalysis=force,
            use_cloud_llm=use_cloud_llm if use_cloud_llm is not None else self.use_cloud_llm,
            seed=self.seed,
            custom_split=custom_split,
            ignore_official=ignore_official,
            stratify_by=stratify_by,
            push_to_hf=push_to_hf,
            requirement=requirement,
            agent_llm_map=merged_llms,
        )
        state = run_pipeline(source, options)
        return state.model_dump(mode="json")

    def organize(
        self,
        dataset_dir: str,
        dataset_name: str | None = None,
        knowledge_urls: list[str] | None = None,
        goal: str = "Build reusable dataset knowledge and analysis artifacts",
        output_dir: str = "organization_output",
        approve: bool = False,
    ) -> dict[str, Any]:
        """Run the organization-style workflow on a local dataset directory."""
        from pathlib import Path

        from cardiomas.organization import build_default_organization

        dataset_path = Path(dataset_dir)
        resolved_name = dataset_name or dataset_path.name
        result = build_default_organization().run(
            goal=goal,
            dataset_name=resolved_name,
            dataset_dir=str(dataset_path),
            knowledge_urls=knowledge_urls or [],
            output_dir=output_dir,
            approve=approve,
        )
        return result.model_dump(mode="json")

    def status(self, dataset_name: str) -> dict[str, Any]:
        """Check if a dataset has splits on HuggingFace."""
        from cardiomas import config as cfg
        from cardiomas.tools.publishing_tools import check_hf_repo

        return check_hf_repo.invoke({"repo_id": cfg.HF_REPO_ID, "dataset_name": dataset_name})

    def get_splits(self, dataset_name: str) -> dict[str, list[str]]:
        """Retrieve splits for a dataset from HuggingFace."""
        result = self.status(dataset_name)
        if not result.get("exists"):
            raise ValueError(f"No splits found for '{dataset_name}' on HuggingFace vlbthambawita/ECGBench")
        return result.get("metadata", {}).get("splits", {})

    def contribute(self, dataset_name: str, split_file: str) -> dict[str, Any]:
        """Package and push community splits to vlbthambawita/ECGBench."""
        from cardiomas import config as cfg
        from cardiomas.tools.security_tools import validate_split_file
        from cardiomas.tools.publishing_tools import push_to_hf

        validation = validate_split_file.invoke({"path": split_file})
        if not validation["valid"]:
            raise ValueError(f"Invalid split file: {validation['issues']}")

        return push_to_hf.invoke({
            "repo_id": cfg.HF_REPO_ID,
            "files": {f"datasets/{dataset_name}/community_splits.json": split_file},
            "commit_message": f"community: add splits for {dataset_name}",
        })
