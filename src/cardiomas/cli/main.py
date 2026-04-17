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
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Build the local knowledge corpus defined by the runtime config."""
    api = _load_api(config_path)
    result = api.build_corpus(force_rebuild=force)
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
    force_rebuild: Annotated[bool, typer.Option("--force-rebuild", help="Rebuild corpus before running the query")] = False,
    live: Annotated[bool, typer.Option("--live", help="Stream step-level events and LLM tokens live")] = False,
    output_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run the fresh Agentic RAG runtime against the configured knowledge sources."""
    api = _load_api(config_path)
    if live:
        events = api.query_stream(question, force_rebuild=force_rebuild)
        if output_json:
            for event in events:
                rprint(json.dumps(event, default=str))
            return
        _render_live_events(events)
        return

    result = api.query(question, force_rebuild=force_rebuild)
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


@app.command()
def version() -> None:
    """Print the installed CardioMAS version."""
    from cardiomas import __version__

    console.print(f"cardiomas {__version__}")


def _render_query_result(result: dict, answer_already_streamed: bool = False) -> None:
    if not answer_already_streamed:
        console.print("[bold]Answer[/bold]")
        console.print(result["answer"])

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
            console.print(f"[green]{active_llm_stage} stream[/green]")
        elif event_type == "llm_token":
            console.print(event.get("content", ""), end="")
        elif event_type == "llm_stream_end":
            if active_llm_stage:
                console.print()
            active_llm_stage = ""
        elif event_type == "final_result":
            result = event.get("data", {}).get("result", {})
            _render_query_result(result, answer_already_streamed=False)


if __name__ == "__main__":
    app()
