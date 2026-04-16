from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="cardiomas",
    help="CardioMAS — Cardio Multi-Agent System for reproducible ECG dataset splits",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    dataset_source: Annotated[str, typer.Argument(help="URL (HF, PhysioNet) or local path")],
    local_path: Annotated[Optional[str], typer.Option("--local-path", help="Explicit local data path")] = None,
    output_dir: Annotated[str, typer.Option("--output-dir")] = "output",
    force: Annotated[bool, typer.Option("--force-reanalysis")] = False,
    use_cloud_llm: Annotated[bool, typer.Option("--use-cloud-llm")] = False,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    custom_split: Annotated[Optional[str], typer.Option("--custom-split", help="e.g. train:0.7,val:0.15,test:0.15")] = None,
    ignore_official: Annotated[bool, typer.Option("--ignore-official")] = False,
    stratify_by: Annotated[Optional[str], typer.Option("--stratify-by")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    push: Annotated[bool, typer.Option("--push", help="Publish to HuggingFace (requires HF_TOKEN)")] = False,
    output_json: Annotated[bool, typer.Option("--json")] = False,
    # V2 options
    requirement: Annotated[Optional[str], typer.Option("--requirement", "-r", help="Natural language requirement (V2)")] = None,
    llm_orchestrator: Annotated[Optional[str], typer.Option("--llm-orchestrator", help="Model for orchestrator agent")] = None,
    llm_nl_requirement: Annotated[Optional[str], typer.Option("--llm-nl-requirement", help="Model for nl_requirement agent")] = None,
    llm_discovery: Annotated[Optional[str], typer.Option("--llm-discovery", help="Model for discovery agent")] = None,
    llm_paper: Annotated[Optional[str], typer.Option("--llm-paper", help="Model for paper agent")] = None,
    llm_analysis: Annotated[Optional[str], typer.Option("--llm-analysis", help="Model for analysis agent")] = None,
    llm_splitter: Annotated[Optional[str], typer.Option("--llm-splitter", help="Model for splitter agent")] = None,
    llm_security: Annotated[Optional[str], typer.Option("--llm-security", help="Model for security agent")] = None,
    llm_coder: Annotated[Optional[str], typer.Option("--llm-coder", help="Model for coder agent")] = None,
    llm_publisher: Annotated[Optional[str], typer.Option("--llm-publisher", help="Model for publisher agent")] = None,
    # V4 options
    approve: Annotated[bool, typer.Option("--approve", help="Approve subset validation and proceed to full dataset run (V4)")] = False,
    auto_approve: Annotated[bool, typer.Option("--auto-approve", help="Skip human approval gate and auto-approve subset validation (V4)")] = False,
    v4_subset_size: Annotated[int, typer.Option("--v4-subset-size", help="Number of records for subset validation (V4, default 100)")] = 100,
    skip_ecg_stats: Annotated[bool, typer.Option("--skip-ecg-stats", help="Skip ECG statistical analysis phase (V4)")] = False,
) -> None:
    """Analyze an ECG dataset and save splits locally. Use --push to publish to HuggingFace."""
    from cardiomas.schemas.state import UserOptions
    from cardiomas.graph.workflow import run_pipeline

    # Parse custom split
    custom_split_dict: dict | None = None
    if custom_split:
        try:
            custom_split_dict = {}
            for part in custom_split.split(","):
                name, ratio = part.strip().split(":")
                custom_split_dict[name.strip()] = float(ratio.strip())
        except Exception:
            typer.echo(f"Invalid --custom-split format: {custom_split}", err=True)
            raise typer.Exit(1)

    # Build per-agent LLM map from CLI flags
    agent_llm_map: dict[str, str] = {}
    _agent_flag_map = {
        "orchestrator":   llm_orchestrator,
        "nl_requirement": llm_nl_requirement,
        "discovery":      llm_discovery,
        "paper":          llm_paper,
        "analysis":       llm_analysis,
        "splitter":       llm_splitter,
        "security":       llm_security,
        "coder":          llm_coder,
        "publisher":      llm_publisher,
    }
    for agent, model in _agent_flag_map.items():
        if model:
            agent_llm_map[agent] = model
            # Also apply at config level for env-var-based resolution
            import cardiomas.config as _cfg
            _cfg.set_agent_llm(agent, model)

    # V4 approval status
    v4_approval_status = "pending"
    if approve:
        v4_approval_status = "approved"

    options = UserOptions(
        dataset_source=dataset_source,
        local_path=local_path,
        output_dir=output_dir,
        force_reanalysis=force,
        use_cloud_llm=use_cloud_llm,
        seed=seed,
        custom_split=custom_split_dict,
        ignore_official=ignore_official,
        stratify_by=stratify_by,
        verbose=verbose,
        push_to_hf=push,
        requirement=requirement,
        agent_llm_map=agent_llm_map,
        # V4 options
        v4_auto_approve=auto_approve,
        v4_subset_size=v4_subset_size,
        v4_skip_ecg_stats=skip_ecg_stats,
    )

    from cardiomas.verbose import enable as verbose_enable
    verbose_enable(verbose)

    if verbose:
        if requirement:
            console.print(f"[dim]Requirement:[/dim] {requirement}\n")
        if agent_llm_map:
            console.print(f"[dim]Per-agent LLMs:[/dim] {agent_llm_map}\n")
        if approve:
            console.print("[dim]V4: approval_status=approved[/dim]\n")
        if auto_approve:
            console.print("[dim]V4: auto-approve enabled[/dim]\n")
        console.print("[dim]Verbose mode on — streaming agent output below.[/dim]\n")
        state = run_pipeline(dataset_source, options, v4_approval_status=v4_approval_status)
    else:
        with console.status("[bold green]Running CardioMAS pipeline...", spinner="dots"):
            state = run_pipeline(dataset_source, options, v4_approval_status=v4_approval_status)

    if output_json:
        rprint(json.dumps(state.model_dump(mode="json"), indent=2, default=str))
        return

    # Pretty output
    if state.errors:
        console.print("[bold red]Pipeline completed with errors:[/bold red]")
        for err in state.errors:
            console.print(f"  [red]✗[/red] {err}")
    else:
        console.print("[bold green]Pipeline completed successfully.[/bold green]")

    if state.proposed_splits:
        splits = state.proposed_splits.splits
        table = Table(title=f"Splits for {state.proposed_splits.dataset_name}")
        table.add_column("Split", style="cyan")
        table.add_column("Records", justify="right")
        table.add_column("Ratio", justify="right")
        total = sum(len(v) for v in splits.values())
        for split_name, ids in splits.items():
            ratio = f"{len(ids)/total:.1%}" if total else "0%"
            table.add_row(split_name, str(len(ids)), ratio)
        console.print(table)

    if state.local_output_dir:
        console.print(f"\n[cyan]Saved locally:[/cyan] {state.local_output_dir}/")
        console.print(f"  splits.json, split_metadata.json, analysis_report.md")
        if state.generated_scripts:
            console.print(f"  scripts/  ({', '.join(state.generated_scripts.keys())})")
        if Path(state.local_output_dir, "session_log").exists():
            console.print(f"  session_log/  (session.json, conversation.md, reasoning_trace.md)")
        if not push:
            ds = state.proposed_splits.dataset_name if state.proposed_splits else "<name>"
            console.print(f"\n[dim]To publish to HuggingFace: cardiomas push {ds}[/dim]")

    if state.publish_status == "ok":
        console.print("[green]Published to HuggingFace: vlbthambawita/ECGBench[/green]")
    elif state.publish_status == "already_published":
        console.print("[blue]Dataset already on HF. Use --force-reanalysis to overwrite.[/blue]")

    if state.parsed_requirement:
        pr = state.parsed_requirement
        if hasattr(pr, "notes") and pr.notes:
            console.print(f"\n[yellow]Requirement notes:[/yellow] {pr.notes}")

    if verbose:
        from rich.rule import Rule
        console.print(Rule("[dim]execution log[/dim]"))
        for entry in state.execution_log:
            console.print(
                f"  [dim]{entry.timestamp.strftime('%H:%M:%S')} "
                f"[{entry.agent}] {entry.action}: {entry.detail}[/dim]"
            )
        if state.orchestrator_reasoning:
            console.print(Rule("[dim]orchestrator reasoning[/dim]"))
            for r in state.orchestrator_reasoning:
                console.print(f"  [dim]{r}[/dim]")


@app.command()
def resume(
    checkpoint_file: Annotated[str, typer.Argument(help="Path to session_checkpoint.json")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Resume a pipeline from a saved checkpoint file."""
    from cardiomas.graph.workflow import resume_pipeline
    from cardiomas.verbose import enable as verbose_enable

    if not Path(checkpoint_file).exists():
        console.print(f"[red]Checkpoint not found: {checkpoint_file}[/red]")
        raise typer.Exit(1)

    verbose_enable(verbose)
    if verbose:
        console.print(f"[dim]Resuming from {checkpoint_file}[/dim]\n")
        state = resume_pipeline(checkpoint_file)
    else:
        with console.status("[bold green]Resuming CardioMAS pipeline...", spinner="dots"):
            state = resume_pipeline(checkpoint_file)

    if output_json:
        rprint(json.dumps(state.model_dump(mode="json"), indent=2, default=str))
        return

    if state.errors:
        console.print("[bold red]Pipeline completed with errors:[/bold red]")
        for err in state.errors:
            console.print(f"  [red]✗[/red] {err}")
    else:
        console.print("[bold green]Pipeline resumed and completed.[/bold green]")
    if state.local_output_dir:
        console.print(f"[cyan]Output:[/cyan] {state.local_output_dir}/")


@app.command()
def push(
    dataset_name: Annotated[str, typer.Argument(help="Dataset name matching a local output directory")],
    output_dir: Annotated[str, typer.Option("--output-dir")] = "output",
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Push locally saved splits to HuggingFace vlbthambawita/ECGBench. Requires HF_TOKEN."""
    import json as _json
    from cardiomas import config as cfg
    from cardiomas.tools.security_tools import validate_split_file
    from cardiomas.tools.publishing_tools import push_to_hf
    from cardiomas.publishing.github_updater import update_github_page

    if not cfg.HF_TOKEN:
        console.print("[red]HF_TOKEN is not set. Run: export HF_TOKEN=<your_token>[/red]")
        raise typer.Exit(1)

    dataset_dir = Path(output_dir) / dataset_name
    splits_file = dataset_dir / "splits.json"
    meta_file   = dataset_dir / "split_metadata.json"
    report_file = dataset_dir / "analysis_report.md"

    if not splits_file.exists():
        console.print(f"[red]No local splits found at {splits_file}[/red]")
        console.print(f"Run [cyan]cardiomas analyze {dataset_name}[/cyan] first.")
        raise typer.Exit(1)

    with console.status("Running security audit…"):
        validation = validate_split_file.invoke({"path": str(splits_file)})

    if not validation["valid"]:
        console.print("[red]Security audit failed — not pushing:[/red]")
        for issue in validation["issues"]:
            console.print(f"  [red]✗[/red] {issue}")
        raise typer.Exit(1)

    console.print("[green]✓[/green] Security audit passed")

    files: dict[str, str] = {
        f"datasets/{dataset_name}/splits.json": str(splits_file),
    }
    if meta_file.exists():
        files[f"datasets/{dataset_name}/split_metadata.json"] = str(meta_file)
    if report_file.exists():
        files[f"datasets/{dataset_name}/analysis_report.md"] = str(report_file)

    with console.status(f"Pushing {len(files)} file(s) to {cfg.HF_REPO_ID}…"):
        result = push_to_hf.invoke({
            "repo_id": cfg.HF_REPO_ID,
            "files": files,
            "commit_message": f"add splits for {dataset_name}",
        })

    if output_json:
        rprint(_json.dumps(result, indent=2))
        return

    if result["status"] == "ok":
        hf_url = result.get("url", "")
        console.print(f"[green]Published:[/green] {hf_url}")
        update_github_page(dataset_name, hf_url)
        console.print("[green]GitHub README updated.[/green]")
    else:
        console.print(f"[red]Push failed:[/red] {result.get('error', '?')}")
        raise typer.Exit(1)


@app.command()
def status(
    dataset_name: Annotated[str, typer.Argument(help="Dataset name to check (e.g. ptb-xl)")],
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Check if a dataset already has splits on HuggingFace vlbthambawita/ECGBench."""
    from cardiomas import config as cfg
    from cardiomas.tools.publishing_tools import check_hf_repo

    with console.status(f"[bold]Checking HF for {dataset_name}...", spinner="dots"):
        result = check_hf_repo.invoke({"repo_id": cfg.HF_REPO_ID, "dataset_name": dataset_name})

    if output_json:
        rprint(json.dumps(result, indent=2, default=str))
        return

    if result.get("exists"):
        console.print(f"[green]✓[/green] {dataset_name} found on HuggingFace ({cfg.HF_REPO_ID})")
        meta = result.get("metadata", {})
        if meta:
            for split_name, ids in meta.get("splits", {}).items():
                console.print(f"  {split_name}: {len(ids)} records")
    else:
        console.print(f"[yellow]✗[/yellow] {dataset_name} not yet published to {cfg.HF_REPO_ID}")


@app.command(name="list")
def list_datasets(
    remote: Annotated[bool, typer.Option("--remote")] = False,
    local: Annotated[bool, typer.Option("--local")] = False,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List known ECG datasets (registry) and optionally remote/local ones."""
    from cardiomas.datasets.registry import get_registry
    from cardiomas import config as cfg

    registry = get_registry()
    datasets = registry.all()

    if output_json:
        rprint(json.dumps([d.model_dump(mode="json") for d in datasets], indent=2, default=str))
        return

    table = Table(title="Known ECG Datasets (Registry)")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="dim")
    table.add_column("Records", justify="right")
    table.add_column("Leads", justify="right")
    table.add_column("Fs (Hz)", justify="right")
    for d in datasets:
        table.add_row(
            d.name, d.source_type.value,
            str(d.num_records or "?"), str(d.num_leads or "?"), str(d.sampling_rate or "?"),
        )
    console.print(table)

    if remote:
        from huggingface_hub import HfApi
        console.print(f"\n[bold]Published on HuggingFace ({cfg.HF_REPO_ID}):[/bold]")
        try:
            api = HfApi()
            files = list(api.list_repo_files(cfg.HF_REPO_ID, repo_type="dataset"))
            published = sorted({
                f.split("/")[1]
                for f in files
                if f.startswith("datasets/") and f.endswith("/splits.json")
            })
            if published:
                remote_table = Table()
                remote_table.add_column("Dataset", style="cyan")
                remote_table.add_column("HF URL", style="dim")
                for name in published:
                    url = f"https://huggingface.co/datasets/{cfg.HF_REPO_ID}/tree/main/datasets/{name}"
                    remote_table.add_row(name, url)
                console.print(remote_table)
            else:
                console.print("  [dim]No datasets published yet.[/dim]")
        except Exception as e:
            console.print(f"  [red]Could not reach HuggingFace: {e}[/red]")

    if local:
        local_dir = cfg.DATA_DIR
        console.print(f"\n[dim]Local cache: {local_dir}[/dim]")
        if local_dir.exists():
            for p in sorted(local_dir.iterdir()):
                if p.is_dir():
                    console.print(f"  [green]{p.name}[/green]")
        else:
            console.print("  [dim](empty)[/dim]")


@app.command()
def config(
    show: Annotated[bool, typer.Option("--show")] = False,
    set_key: Annotated[Optional[str], typer.Option("--set", help="KEY=VALUE")] = None,
) -> None:
    """View or update CardioMAS configuration."""
    import cardiomas.config as cfg_module

    if show or not set_key:
        console.print("[bold]Current Configuration:[/bold]")
        console.print(f"  OLLAMA_BASE_URL  = {cfg_module.OLLAMA_BASE_URL}")
        console.print(f"  OLLAMA_MODEL     = {cfg_module.OLLAMA_MODEL}")
        console.print(f"  CLOUD_PROVIDER   = {cfg_module.CLOUD_LLM_PROVIDER}")
        console.print(f"  HF_REPO_ID       = {cfg_module.HF_REPO_ID}")
        console.print(f"  GITHUB_REPO      = {cfg_module.GITHUB_REPO}")
        console.print(f"  DATA_DIR         = {cfg_module.DATA_DIR}")
        console.print(f"  SEED             = {cfg_module.SEED}")
        console.print(f"  COMPRESS_MODEL   = {cfg_module.CONTEXT_COMPRESS_MODEL}")
        console.print(f"  COMPRESS_THRESH  = {cfg_module.CONTEXT_COMPRESS_THRESHOLD}")
        hf_ok = "[green]set[/green]" if cfg_module.HF_TOKEN else "[red]not set[/red]"
        gh_ok = "[green]set[/green]" if cfg_module.GITHUB_TOKEN else "[red]not set[/red]"
        console.print(f"  HF_TOKEN         = {hf_ok}")
        console.print(f"  GITHUB_TOKEN     = {gh_ok}")
        if cfg_module._AGENT_LLM_OVERRIDES:
            console.print("\n  [bold]Per-agent LLM overrides:[/bold]")
            for agent, model in cfg_module._AGENT_LLM_OVERRIDES.items():
                console.print(f"    {agent} = {model}")
        return

    if set_key:
        if "=" not in set_key:
            typer.echo("--set requires KEY=VALUE format", err=True)
            raise typer.Exit(1)
        key, value = set_key.split("=", 1)
        env_file = Path(".env")
        lines = env_file.read_text().splitlines() if env_file.exists() else []
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                updated = True
        if not updated:
            lines.append(f"{key}={value}")
        env_file.write_text("\n".join(lines) + "\n")
        console.print(f"[green]Set {key}={value} in .env[/green]")


@app.command()
def contribute(
    dataset_name: Annotated[str, typer.Argument(help="Dataset name")],
    split_file: Annotated[str, typer.Option("--split-file")] = "",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Package and submit community splits to vlbthambawita/ECGBench."""
    from cardiomas.tools.security_tools import validate_split_file as validate
    from cardiomas.tools.publishing_tools import push_to_hf
    from cardiomas import config as cfg

    if not split_file or not Path(split_file).exists():
        typer.echo(f"Split file not found: {split_file}", err=True)
        raise typer.Exit(1)

    with console.status("Validating split file..."):
        validation = validate.invoke({"path": split_file})

    if not validation["valid"]:
        console.print("[red]Split file failed validation:[/red]")
        for issue in validation["issues"]:
            console.print(f"  [red]✗[/red] {issue}")
        raise typer.Exit(1)

    console.print("[green]✓[/green] Split file validated")

    if dry_run:
        console.print("[yellow]Dry-run mode — not pushing.[/yellow]")
        return

    result = push_to_hf.invoke({
        "repo_id": cfg.HF_REPO_ID,
        "files": {f"datasets/{dataset_name}/community_splits.json": split_file},
        "commit_message": f"community: add splits for {dataset_name}",
    })
    if result["status"] == "ok":
        console.print(f"[green]Contributed splits for {dataset_name}: {result.get('url', '')}[/green]")
    else:
        console.print(f"[red]Failed: {result.get('error')}[/red]")
        raise typer.Exit(1)


@app.command()
def verify(
    dataset_name: Annotated[str, typer.Argument(help="Dataset name")],
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Re-check reproducibility metadata of published splits."""
    from cardiomas import config as cfg
    from cardiomas.tools.publishing_tools import check_hf_repo

    console.print(f"[bold]Verifying reproducibility for {dataset_name}...[/bold]")
    result = check_hf_repo.invoke({"repo_id": cfg.HF_REPO_ID, "dataset_name": dataset_name})

    if not result.get("exists"):
        console.print(f"[yellow]No published splits found for {dataset_name}[/yellow]")
        raise typer.Exit(1)

    published = result.get("metadata", {})
    pub_repro = published.get("reproducibility_config", {})
    pub_seed = pub_repro.get("seed", -1)

    if pub_seed != seed:
        console.print(f"[yellow]Seed mismatch: published={pub_seed}, requested={seed}[/yellow]")

    console.print("[green]✓[/green] Published splits metadata retrieved")
    console.print(f"  Version:   {pub_repro.get('cardiomas_version', '?')}")
    console.print(f"  Strategy:  {pub_repro.get('split_strategy', '?')}")
    console.print(f"  Seed:      {pub_seed}")
    console.print(f"  Timestamp: {pub_repro.get('timestamp', '?')}")
    console.print("\nTo fully verify, re-run `cardiomas analyze` with the same seed and compare split IDs.")


@app.command()
def organize(
    dataset_dir: Annotated[Optional[str], typer.Argument(help="Local dataset directory to analyze")] = None,
    dataset_name: Annotated[Optional[str], typer.Option("--dataset-name")] = None,
    config_path: Annotated[Optional[str], typer.Option("--config", "-c", help="YAML or JSON file containing organize inputs")] = None,
    goal: Annotated[Optional[str], typer.Option("--goal")] = None,
    knowledge_url: Annotated[Optional[list[str]], typer.Option("--knowledge-url", help="Repeat to provide dataset landing pages, docs, or paper links")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir")] = None,
    approve: Annotated[bool, typer.Option("--approve", help="Mark major artifacts as approved in the final report")] = False,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run the organization-style workflow for dataset knowledge, tooling, testing, and ECG review."""
    from cardiomas.organization import build_default_organization, resolve_organization_config

    if not dataset_dir and not config_path:
        console.print("[red]Provide a DATASET_DIR argument or pass --config with local_data_path/dataset_dir.[/red]")
        raise typer.Exit(1)

    try:
        config = resolve_organization_config(
            config_path=config_path,
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            knowledge_urls=knowledge_url,
            goal=goal,
            output_dir=output_dir,
            approve=True if approve else None,
        )
    except Exception as exc:
        console.print(f"[red]Could not load organization config:[/red] {exc}")
        raise typer.Exit(1)

    dataset_path = Path(config.resolved_dataset_dir)
    if not dataset_path.exists():
        console.print(f"[red]Dataset directory not found:[/red] {dataset_path}")
        raise typer.Exit(1)
    if not dataset_path.is_dir():
        console.print(f"[red]Dataset path is not a directory:[/red] {dataset_path}")
        raise typer.Exit(1)

    result = build_default_organization().run(
        goal=config.goal,
        dataset_name=config.dataset_name or dataset_path.name,
        dataset_dir=str(dataset_path),
        knowledge_urls=config.knowledge_urls,
        output_dir=config.output_dir,
        approve=config.approve,
    )

    if output_json:
        rprint(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
        return

    console.print(f"[bold]Organization workflow:[/bold] {config.dataset_name or dataset_path.name}")
    console.print(f"Status: [cyan]{result.status}[/cyan]")
    console.print(f"Output: [cyan]{result.output_dir}[/cyan]")

    for report in result.department_reports:
        console.print(f"\n[bold]{report.department}[/bold] {report.summary}")
        for artifact in report.artifacts:
            console.print(f"  - {artifact.name}: {artifact.path}")
        for note in report.notes:
            console.print(f"  - note: {note}")

    if result.status == "awaiting_approval":
        console.print("\n[dim]Re-run with --approve to mark major artifacts as approved.[/dim]")


@app.command()
def version() -> None:
    """Print the installed CardioMAS version."""
    from cardiomas import __version__
    console.print(f"cardiomas {__version__}")


if __name__ == "__main__":
    app()
