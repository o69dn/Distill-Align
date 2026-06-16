"""
CLI entry point for Distill-Align.

Uses Typer for command-line interface with Rich for beautiful output.
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from ..core.logging import setup_logging
from ..core.schemas import IngestionConfig, SynthesisConfig, ExportConfig, DataChunk, ConversationSchema
from ..ingestion.pipeline import IngestionPipeline
from ..synthesis.pipeline import SynthesisPipeline
from ..exporter.pipeline import ExportPipeline

# Create Typer app
app = typer.Typer(
    name="distill-align",
    help="The Structured Reasoning Extraction Factory",
    add_completion=False,
)
console = Console()


@app.callback()
def main(
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Log file path"),
):
    """Distill-Align: Generate fine-tuning datasets from raw domain data."""
    setup_logging(log_level=log_level, log_file=log_file)


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Source file or directory path"),
    output: str = typer.Option("./chunks.json", "--output", "-o", help="Output file path"),
    chunk_size: int = typer.Option(1000, "--chunk-size", "-s", help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(200, "--overlap", help="Chunk overlap in characters"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r", help="Search subdirectories"),
):
    """Ingest files and split into semantic chunks."""
    console.print(Panel.fit("📥 Ingestion Pipeline", style="bold blue"))

    config = IngestionConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    pipeline = IngestionPipeline(config)

    source_path = Path(source)
    if not source_path.exists():
        console.print(f"[red]Error: Source path does not exist: {source}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting files...", total=None)

        if source_path.is_file():
            chunks = pipeline.ingest_file(source_path)
        else:
            chunks = pipeline.ingest_directory(source_path, recursive=recursive)

        progress.update(task, completed=True)

    # Save chunks
    import json
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_data = [chunk.model_dump() for chunk in chunks]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    # Print summary
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Chunks", str(len(chunks)))
    table.add_row("Output File", str(output_path))
    table.add_row("File Size", f"{output_path.stat().st_size / 1024:.1f} KB")
    console.print(table)


@app.command()
def synthesize(
    input: str = typer.Argument(..., help="Input chunks JSON file"),
    output: str = typer.Option("./conversations.json", "--output", "-o", help="Output file path"),
    provider: str = typer.Option("openai", "--provider", "-p", help="LLM provider (openai, ollama, vllm)"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="API base URL"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Max concurrent requests"),
    rpm: int = typer.Option(60, "--rpm", help="Max requests per minute"),
):
    """Synthesize chunks into structured conversations."""
    console.print(Panel.fit("🧠 Synthesis Pipeline", style="bold magenta"))

    # Load chunks
    import json
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file does not exist: {input}[/red]")
        raise typer.Exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = [DataChunk(**chunk) for chunk in chunks_data]

    config = SynthesisConfig(
        llm_provider=provider,
        model_name=model,
        base_url=base_url,
        api_key=api_key,
        max_concurrency=concurrency,
        max_rpm=rpm,
    )
    pipeline = SynthesisPipeline(config)

    async def run_synthesis():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Synthesizing {len(chunks)} chunks...", total=len(chunks))

            def update_progress(current, total):
                progress.update(task, completed=current)

            conversations = await pipeline.synthesize_batch(chunks, update_progress)
            return conversations

    conversations = asyncio.run(run_synthesis())

    # Save conversations
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conv_data = [conv.model_dump() for conv in conversations]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(conv_data, f, indent=2, ensure_ascii=False)

    # Print summary
    table = Table(title="Synthesis Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Input Chunks", str(len(chunks)))
    table.add_row("Conversations", str(len(conversations)))
    table.add_row("Success Rate", f"{len(conversations)/len(chunks)*100:.1f}%")
    table.add_row("Output File", str(output_path))
    console.print(table)


@app.command()
def export(
    input: str = typer.Argument(..., help="Input conversations JSON file"),
    formats: str = typer.Option("sharegpt", "--format", "-f", help="Export formats (comma-separated)"),
    output_dir: str = typer.Option("./output", "--output-dir", "-o", help="Output directory"),
    model_name: str = typer.Option("unsloth/Meta-Llama-3.1-8B-Instruct", "--model", help="Unsloth model name"),
    no_unsloth: bool = typer.Option(False, "--no-unsloth", help="Skip Unsloth script generation"),
):
    """Export conversations to training formats."""
    console.print(Panel.fit("📤 Export Pipeline", style="bold green"))

    # Load conversations
    import json
    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file does not exist: {input}[/red]")
        raise typer.Exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        conv_data = json.load(f)

    conversations = [ConversationSchema(**conv) for conv in conv_data]

    format_list = [f.strip() for f in formats.split(",")]
    config = ExportConfig(
        formats=format_list,
        output_dir=output_dir,
        unsloth_model=model_name,
    )
    pipeline = ExportPipeline(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Exporting...", total=None)

        output_files = pipeline.export(
            conversations,
            formats=format_list,
            generate_unsloth=not no_unsloth,
        )

        progress.update(task, completed=True)

    # Print summary
    table = Table(title="Export Summary")
    table.add_column("Format", style="cyan")
    table.add_column("File", style="green")
    table.add_column("Size", style="yellow")

    for format_name, file_path in output_files.items():
        size = f"{file_path.stat().st_size / 1024:.1f} KB" if file_path.exists() else "N/A"
        table.add_row(format_name, str(file_path), size)

    console.print(table)


@app.command()
def tui():
    """Launch the interactive TUI dashboard."""
    console.print(Panel.fit("🖥️ Launching TUI...", style="bold cyan"))
    from ..tui.app import DistillAlignApp
    app = DistillAlignApp()
    app.run()


@app.command()
def version():
    """Show version information."""
    from .. import __version__
    console.print(f"distill-align v{__version__}")


if __name__ == "__main__":
    app()
