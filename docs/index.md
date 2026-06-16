# Distill-Align Documentation

Welcome to the Distill-Align documentation!

## Overview

**Distill-Align** is a CLI/Python framework that automates the generation of high-quality, fine-tuning datasets from raw domain data (PDFs, Markdown, Codebases). It utilizes high-context frontier reasoning models as teachers, captures their deep thinking traces, and filters/prunes those traces into highly structured instruction-following formats (ShareGPT/Alpaca JSON) optimized for Unsloth Studio fine-tuning.

## Quick Start

### Installation

```bash
pip install distill-align
```

### Basic Usage

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

## Architecture

Distill-Align follows a **Modular Monolith** architecture:

```
src/distill_align/
├── core/               # Configuration, schemas, logging
├── ingestion/          # Data loaders and chunkers
├── synthesis/          # LLM client, worker pool, prompts
├── exporter/           # Formatters and Unsloth builder
├── tui/                # Textual terminal UI
└── cli/                # Typer CLI entry points
```

## Core Modules

### Ingestion Module

The ingestion module handles loading and chunking raw data:

- **Loaders**: Support for Markdown, PDF, and code files
- **Chunkers**: Semantic-aware splitting based on headers and code definitions
- **Pipeline**: Orchestration for batch processing

### Synthesis Module

The synthesis module transforms chunks into structured conversations:

- **LLM Clients**: Support for OpenAI, Ollama, and vLLM
- **Worker Pool**: Async batch processing with rate limiting
- **Prompts**: Socratic Transformer and Scaffold Action pipelines

### Export Module

The export module formats and exports datasets:

- **Formatters**: ShareGPT and Alpaca JSON formats
- **Unsloth Builder**: Auto-generated training scripts

## Configuration

Distill-Align uses environment variables for configuration:

```bash
# General
DISTILL_LOG_LEVEL=INFO
DISTILL_CACHE_DIR=.cache

# Synthesis
DISTILL_LLM_PROVIDER=openai
DISTILL_LLM_MODEL=gpt-4o
DISTILL_LLM_API_KEY=your-api-key

# Export
DISTILL_EXPORT_FORMATS=sharegpt,alpaca
DISTILL_EXPORT_OUTPUT_DIR=./output
```

## Examples

### Ingest a Markdown Directory

```python
from distill_align.ingestion.pipeline import IngestionPipeline
from distill_align.core.schemas import IngestionConfig

config = IngestionConfig(chunk_size=1000, chunk_overlap=200)
pipeline = IngestionPipeline(config)

chunks = pipeline.ingest_directory("./my-docs", recursive=True)
print(f"Created {len(chunks)} chunks")
```

### Synthesize Conversations

```python
import asyncio
from distill_align.synthesis.pipeline import SynthesisPipeline
from distill_align.core.schemas import SynthesisConfig

config = SynthesisConfig(
    llm_provider="openai",
    model_name="gpt-4o",
    api_key="your-api-key",
)
pipeline = SynthesisPipeline(config)

conversations = asyncio.run(pipeline.synthesize_batch(chunks))
print(f"Created {len(conversations)} conversations")
```

### Export Dataset

```python
from distill_align.exporter.pipeline import ExportPipeline
from distill_align.core.schemas import ExportConfig

config = ExportConfig(
    formats=["sharegpt", "alpaca"],
    output_dir="./output",
)
pipeline = ExportPipeline(config)

output_files = pipeline.export(conversations)
print(f"Exported to: {output_files}")
```

## API Reference

### Core Schemas

- `SourceMetadata`: Metadata for data chunks
- `DataChunk`: Single unit of raw data
- `ConversationSchema`: Structured conversation
- `ShareGPTMessage`: ShareGPT format message
- `AlpacaEntry`: Alpaca format entry

### Pipelines

- `IngestionPipeline`: Data ingestion orchestration
- `SynthesisPipeline`: Conversation synthesis
- `ExportPipeline`: Dataset export

### Workers

- `BatchWorker`: Async batch processing with rate limiting

## Troubleshooting

### Common Issues

**Import Error**: Make sure you've installed the package:
```bash
pip install -e .
```

**Rate Limiting**: Adjust concurrency and RPM settings:
```bash
distill-align synthesize --concurrency 3 --rpm 30
```

**Memory Issues**: Reduce chunk size:
```bash
distill-align ingest --chunk-size 500
```

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/distill-align/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/distill-align/discussions)

## License

MIT License - see [LICENSE](LICENSE) for details.
