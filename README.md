# Distill-Align

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Distill-Align: The Structured Reasoning Extraction Factory**
>
> A CLI/Python framework that automates the generation of high-quality, fine-tuning datasets from raw domain data. It utilizes high-context frontier reasoning models as teachers, captures their deep thinking traces, and filters/prunes those traces into highly structured instruction-following formats (ShareGPT/Alpaca JSON) optimized for Unsloth Studio fine-tuning.

## Features

- **Smart Ingestion**: Async chunking pipelines with semantic-aware splitting for Markdown and Code.
- **Multi-Provider Synthesis**: Supports OpenAI, Ollama, and vLLM backends with async worker pools.
- **Socratic Transformer**: Converts raw reasoning into structured, multi-turn conversational Q&A.
- **Scaffold Action Pruner**: Strips conversational filler to extract pure tool-calling or structural output.
- **Unsloth Integration**: Auto-generates optimized `train.py` scripts and exports to ShareGPT/Alpaca formats.
- **Rich TUI**: Interactive terminal dashboard for monitoring pipeline execution.

## Installation

```bash
pip install distill-align
```

## Quick Start

```bash
# Ingest and process data
distill-align ingest --source ./my-docs --output ./chunks.json

# Synthesize conversations
distill-align synthesize --input ./chunks.json --output ./conversations.json

# Export to training format
distill-align export --input ./conversations.json --format sharegpt --output ./dataset

# Launch TUI
distill-align tui
```

## Project Structure

This project follows a **Modular Monolith** architecture.

```text
distill-align/
├── src/distill_align/    # Core application package
│   ├── core/             # Config, schemas, logging
│   ├── ingestion/        # Data loaders and chunkers
│   ├── synthesis/        # LLM client, worker pool, prompts
│   ├── exporter/         # Formatters and Unsloth builder
│   ├── tui/              # Textual terminal UI
│   └── cli/              # Typer CLI entry points
├── tests/                # Pytest suite
└── docs/                 # Documentation
```

## Development

1. Clone the repository
2. Install dependencies with Poetry: `poetry install`
3. Run tests: `poetry run pytest`

## License

MIT License - see [LICENSE](LICENSE) for details.
