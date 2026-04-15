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
    dataset_source: Annotated[str, typer.Argument(help="URL (HF, PhysioNet, etc.) or local path")],
    local_path: Annotated[Optional[str], typer.Option("--local-path", help="Explicit local data path (skip download)")] = None,
    output_dir: Annotated[str, typer.Option("--output-dir")] = "output",
    force: Annotated[bool, typer.Option("--force-reanalysis", help="Re-run even if HF already has results")] = False,
    use_cloud_llm: Annotated[bool, typer.Option("--use-cloud-llm")] = False,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    custom_split: Annotated[Optional[str], typer.Option("--custom-split", help="e.g. 'train:0.7,val:0.15,test:0.15'")] = None,
    ignore_official: Annotated[bool, typer.Option("--ignore-official")] = False,
    stratify_by: Annotated[Optional[str], typer.Option("--stratify-by")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Analyze an ECG dataset and publish reproducible splits to HuggingFace."""
    from cardiomas.schemas.state import UserOptions
    from cardiomas.graph.workflow import run_pipeline

    # Parse custom split
    custom_split_dict = None
    if custom_split:
        try:
            custom_split_dict = {}
            for part in custom_split.split(","):
                name, ratio = part.strip().split(":")
                custom_split_dict[name.strip()] = float(ratio.strip())
        except Exception:
            typer.echo(f"Invalid --custom-split format: {custom_split}", err=True)
            raise typer.Exit(1)

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
        dry_run=dry_run,
    )

    from cardiomas.verbose import enable as verbose_enable
    verbose_enable(verbose)

    if verbose:
        console.print("[dim]Verbose mode on — streaming agent output below.[/dim]\n")
        state = run_pipeline(dataset_source, options)
    else:
        with console.status("[bold green]Running CardioMAS pipeline...", spinner="dots"):
            state = run_pipeline(dataset_source, options)

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

    if state.publish_status == "ok":
        console.print(f"[green]Published to HuggingFace: vlbthambawita/ECGBench[/green]")
    elif state.publish_status == "dry_run":
        console.print("[yellow]Dry-run mode — nothing published.[/yellow]")
    elif state.publish_status == "already_published":
        console.print("[blue]Dataset already published on HF. Use --force-reanalysis to overwrite.[/blue]")

    if verbose:
        from rich.rule import Rule
        console.print(Rule("[dim]execution log[/dim]"))
        for entry in state.execution_log:
            console.print(f"  [dim]{entry.timestamp.strftime('%H:%M:%S')} [{entry.agent}] {entry.action}: {entry.detail}[/dim]")


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
            splits = meta.get("splits", {})
            for split_name, ids in splits.items():
                console.print(f"  {split_name}: {len(ids)} records")
    else:
        console.print(f"[yellow]✗[/yellow] {dataset_name} not yet published to {cfg.HF_REPO_ID}")


@app.command(name="list")
def list_datasets(
    remote: Annotated[bool, typer.Option("--remote", help="Show datasets on HuggingFace")] = False,
    local: Annotated[bool, typer.Option("--local", help="Show locally cached datasets")] = False,
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
            d.name,
            d.source_type.value,
            str(d.num_records or "?"),
            str(d.num_leads or "?"),
            str(d.sampling_rate or "?"),
        )
    console.print(table)

    if remote:
        from huggingface_hub import HfApi
        console.print(f"\n[bold]Published datasets on HuggingFace ({cfg.HF_REPO_ID}):[/bold]")
        try:
            api = HfApi()
            files = list(api.list_repo_files(cfg.HF_REPO_ID, repo_type="dataset"))
            # Paths look like: datasets/{name}/splits.json
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
    show: Annotated[bool, typer.Option("--show", help="Show current configuration")] = False,
    set_key: Annotated[Optional[str], typer.Option("--set", help="KEY=VALUE to set")] = None,
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
        hf_ok = "[green]set[/green]" if cfg_module.HF_TOKEN else "[red]not set[/red]"
        gh_ok = "[green]set[/green]" if cfg_module.GITHUB_TOKEN else "[red]not set[/red]"
        console.print(f"  HF_TOKEN         = {hf_ok}")
        console.print(f"  GITHUB_TOKEN     = {gh_ok}")
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
    split_file: Annotated[str, typer.Option("--split-file", help="Path to your splits JSON")] = "",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Package and submit community splits to vlbthambawita/ECGBench via PR."""
    from cardiomas.tools.security_tools import validate_split_file as validate
    from cardiomas.tools.publishing_tools import push_to_hf
    from cardiomas import config as cfg

    if not split_file or not Path(split_file).exists():
        typer.echo(f"Split file not found: {split_file}", err=True)
        raise typer.Exit(1)

    # Validate
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

    # Push
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
    dataset_name: Annotated[str, typer.Argument(help="Dataset name to verify")],
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Re-run split generation and compare against published splits to confirm reproducibility."""
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
    console.print(f"  Version: {pub_repro.get('cardiomas_version', '?')}")
    console.print(f"  Strategy: {pub_repro.get('split_strategy', '?')}")
    console.print(f"  Seed: {pub_seed}")
    console.print(f"  Timestamp: {pub_repro.get('timestamp', '?')}")
    console.print("\nTo fully verify, re-run `cardiomas analyze` with the same seed and compare split IDs.")


@app.command()
def version() -> None:
    """Print the installed CardioMAS version."""
    from cardiomas import __version__
    console.print(f"cardiomas {__version__}")


if __name__ == "__main__":
    app()
