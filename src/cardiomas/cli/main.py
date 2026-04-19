from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from cardiomas import CardioMAS
from cardiomas.schemas.config import RuntimeConfig

app = typer.Typer(
    name="cardiomas",
    help="CardioMAS — Fresh Agentic RAG runtime for dataset understanding and grounded Q&A",
    add_completion=False,
)
console = Console()


def _load_api(config_path: str) -> CardioMAS:
    try:
        return CardioMAS(config_path=config_path)
    except Exception as exc:
        console.print(f"[red]Could not load runtime config:[/red] {exc}")
        raise typer.Exit(1)


@app.command("build-corpus")
def build_corpus(
    config_path: Annotated[str, typer.Option("--config", "-c", help="Path to YAML or JSON runtime config")],
    force: Annotated[bool, typer.Option("--force", help="Rebuild the corpus even if it already exists")] = False,
    agent: Annotated[str | None, typer.Option("--agent", "-a", help="Build only this named agent's corpus")] = None,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Build the local knowledge corpus defined by the runtime config."""
    api = _load_api(config_path)
    result = api.build_corpus(force_rebuild=force, agent_name=agent)
    if output_json:
        rprint(json.dumps(result, indent=2, default=str))
        return
    console.print(f"[green]Corpus built[/green] -> {result['corpus_path']}")
    console.print(f"Documents: {result['document_count']}")
    console.print(f"Chunks: {result['chunk_count']}")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Grounded question to answer")],
    config_path: Annotated[str, typer.Option("--config", "-c", help="Path to YAML or JSON runtime config")],
    agent: Annotated[str | None, typer.Option("--agent", "-a", help="Named agent to use for this query")] = None,
    force_rebuild: Annotated[bool, typer.Option("--force-rebuild", help="Rebuild corpus before running the query")] = False,
    live: Annotated[bool, typer.Option("--live", help="Stream step-level events and LLM tokens live")] = False,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run the fresh Agentic RAG runtime against the configured knowledge sources."""
    api = _load_api(config_path)
    if live:
        events = api.query_stream(question, force_rebuild=force_rebuild, agent_name=agent)
        if output_json:
            for event in events:
                rprint(json.dumps(event, default=str))
            return
        _render_live_events(events)
        return

    result = api.query(question, force_rebuild=force_rebuild, agent_name=agent)
    if output_json:
        rprint(json.dumps(result, indent=2, default=str))
        return

    _render_query_result(result)


@app.command("inspect-tools")
def inspect_tools(
    config_path: Annotated[str, typer.Option("--config", "-c", help="Path to YAML or JSON runtime config")],
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List the tools enabled by the current runtime config."""
    api = _load_api(config_path)
    specs = api.inspect_tools()
    if output_json:
        rprint(json.dumps(specs, indent=2, default=str))
        return

    table = Table(title="Enabled Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Category")
    table.add_column("Read only")
    table.add_column("Description")
    for spec in specs:
        table.add_row(spec["name"], spec["category"], str(spec["read_only"]), spec["description"])
    console.print(table)


@app.command("show-config")
def show_config(
    config_path: Annotated[str, typer.Option("--config", "-c", help="Path to YAML or JSON runtime config")],
) -> None:
    """Print the resolved runtime config."""
    config = RuntimeConfig.from_file(config_path)
    rprint(json.dumps(config.model_dump(mode="json"), indent=2, default=str))


@app.command("check-ollama")
def check_ollama(
    config_path: Annotated[str, typer.Option("--config", "-c", help="Path to YAML or JSON runtime config")],
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Check Ollama connectivity for the configured LLM and embedding clients."""
    api = _load_api(config_path)
    status = api.check_ollama()
    if output_json:
        rprint(json.dumps(status, indent=2, default=str))
        return

    table = Table(title="Ollama Status")
    table.add_column("Client", style="cyan")
    table.add_column("Configured")
    table.add_column("OK")
    table.add_column("Models")
    table.add_column("Error")
    for name in ["llm", "embeddings"]:
        item = status.get(name, {})
        configured = item.get("configured", True)
        models = ", ".join(model["name"] for model in item.get("models", [])) if item.get("models") else ""
        table.add_row(name, str(configured), str(item.get("ok", False)), models, item.get("error", ""))
    console.print(table)


@app.command("list-agents")
def list_agents(
    config_path: Annotated[str, typer.Option("--config", "-c", help="Path to YAML or JSON runtime config")],
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List configured named agents and whether their corpus has been built."""
    api = _load_api(config_path)
    agents = api.list_agents()
    if output_json:
        rprint(json.dumps(agents, indent=2, default=str))
        return
    if not agents:
        console.print("[yellow]No named agents configured.[/yellow]")
        config = RuntimeConfig.from_file(config_path)
        if not config.knowledge_scraping.enabled:
            console.print("[dim]Tip: set knowledge_scraping.enabled: true to activate agent knowledge scraping.[/dim]")
        return
    table = Table(title="Named Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Knowledge")
    table.add_column("Sources")
    table.add_column("Corpus built")
    for ag in agents:
        table.add_row(
            ag["name"],
            ag["description"][:60] + ("…" if len(ag["description"]) > 60 else ""),
            "[green]on[/green]" if ag["knowledge_enabled"] else "[dim]off[/dim]",
            str(ag["source_count"]),
            "[green]yes[/green]" if ag["corpus_built"] else "[red]no[/red]",
        )
    console.print(table)
    config = RuntimeConfig.from_file(config_path)
    if not config.knowledge_scraping.enabled:
        console.print("[yellow]Note: knowledge_scraping.enabled is false — run build-corpus to activate.[/yellow]")


@app.command()
def version() -> None:
    """Print the installed CardioMAS version."""
    from cardiomas import __version__

    console.print(f"cardiomas {__version__}")


def _render_query_result(result: dict, answer_already_streamed: bool = False) -> None:
    standalone_scripts = result.get("standalone_scripts", [])
    executed_scripts = [s for s in standalone_scripts if s.get("executed") and s.get("execution_stdout")]

    if standalone_scripts and not executed_scripts:
        # Phase 1: show script locations, answer IS the script report
        console.print("[bold green]Scripts Generated[/bold green]")
        console.rule()
        for script in standalone_scripts:
            console.print(f"  [cyan]{script.get('script_name', '')}[/cyan]")
            console.print(f"  Path:   {script.get('script_path', '')}")
            console.print(f"  Run:    python {script.get('script_path', '')}")
            if script.get("output_dir"):
                console.print(f"  Output: {script['output_dir']}/results.json")
            console.print()
        return

    if not answer_already_streamed:
        console.print("[bold]Answer[/bold]")
        console.print(result["answer"])

    if executed_scripts:
        console.print()
        console.print("[bold green]Script(s) Executed[/bold green]")
        console.rule()
        for script in executed_scripts:
            console.print(f"  [cyan]{script.get('script_name', '')}[/cyan]  →  {script.get('script_path', '')}")

    if result["citations"]:
        table = Table(title="Citations")
        table.add_column("Source", style="cyan")
        table.add_column("Locator")
        table.add_column("Type")
        for citation in result["citations"]:
            table.add_row(citation["source_label"], citation["locator"], citation["source_type"])
        console.print(table)

    if result["llm_traces"]:
        table = Table(title="LLM Traces")
        table.add_column("Stage", style="cyan")
        table.add_column("Model")
        table.add_column("OK")
        table.add_column("Error")
        for trace in result["llm_traces"]:
            table.add_row(trace["stage"], trace["model"], str(trace["ok"]), trace["error"])
        console.print(table)

    if result["repair_traces"]:
        table = Table(title="Repair Traces")
        table.add_column("Tool", style="cyan")
        table.add_column("Action")
        table.add_column("Attempt")
        table.add_column("OK")
        table.add_column("Error")
        for trace in result["repair_traces"]:
            table.add_row(
                trace["tool_name"],
                trace["action"],
                str(trace["attempt"]),
                str(trace["ok"]),
                trace["error"],
            )
        console.print(table)

    react_steps = result.get("react_steps", [])
    if react_steps:
        table = Table(title="ReAct Steps")
        table.add_column("Iter", style="cyan")
        table.add_column("Action")
        table.add_column("OK")
        table.add_column("Observation")
        for step in react_steps:
            obs = step.get("observation", "")
            obs_short = obs[:80] + "..." if len(obs) > 80 else obs
            table.add_row(
                str(step.get("iteration", "")),
                step.get("action", ""),
                str(step.get("ok", True)),
                obs_short,
            )
        console.print(table)

    if result["warnings"]:
        console.print("[yellow]Warnings[/yellow]")
        for warning in result["warnings"]:
            console.print(f"- {warning}")


def _render_live_events(events) -> None:
    active_llm_stage = ""

    for event in events:
        event_type = event.get("type", "")
        if event_type == "status":
            console.print(f"[cyan]{event.get('stage', 'runtime')}[/cyan] {event.get('message', '')}")
        elif event_type == "tool_started":
            console.print(f"[blue]tool[/blue] {event['data'].get('tool_name', '')} started")
        elif event_type == "tool_finished":
            status = "ok" if event["data"].get("ok", False) else "failed"
            console.print(f"[blue]tool[/blue] {event['data'].get('tool_name', '')} {status}")
        elif event_type == "repair_trace":
            trace = event.get("data", {})
            console.print(
                f"[magenta]repair[/magenta] {trace.get('tool_name', '')} {trace.get('action', '')} "
                f"attempt {trace.get('attempt', 0)} ok={trace.get('ok', False)}"
            )
        elif event_type == "llm_stream_start":
            active_llm_stage = event.get("stage", "")
            label = "responder" if active_llm_stage == "responder" else active_llm_stage
            console.print(f"\n[green dim]▶ {label}[/green dim] ", end="")
        elif event_type == "llm_token":
            console.print(event.get("content", ""), end="", highlight=False)
        elif event_type == "llm_stream_end":
            if active_llm_stage:
                console.print()
            active_llm_stage = ""
        elif event_type == "final_result":
            result = event.get("data", {}).get("result", {})
            _render_query_result(result, answer_already_streamed=False)


if __name__ == "__main__":
    app()
