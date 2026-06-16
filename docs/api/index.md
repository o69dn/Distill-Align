# API Reference

This section contains auto-generated API reference documentation for the Distill-Align Python package.

## Modules

- `distill_align.core` - Configuration, schemas, logging, exceptions
- `distill_align.ingestion` - File loaders and chunkers
- `distill_align.synthesis` - LLM clients, prompts, conversation builders
- `distill_align.exporter` - Formatters, splitter, validator, Unsloth builder
- `distill_align.tui` - Textual terminal UI
- `distill_align.cli` - Typer CLI entry points

## Building the API Docs

This page is populated by [mkdocstrings](https://mkdocstrings.github.io/). To generate the full reference:

```bash
poetry run mkdocs build
```

Or serve with live reload:

```bash
poetry run mkdocs serve
```
