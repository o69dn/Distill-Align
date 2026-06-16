# Getting Started

This guide will walk you through installing Distill-Align and running your first end-to-end pipeline.

## Installation

### From PyPI

```bash
pip install distill-align
```

### From Source (Development)

```bash
git clone https://github.com/o69dn/Distill-Align.git
cd Distill-Align
poetry install
```

### Using Docker

```bash
docker build -t distill-align .
docker run -it --rm -v $(pwd)/data:/app/data distill-align
```

## Quick Start

### 1. Initialize a Project

```bash
distill-align init
```

This creates a `distill-align.yaml` config file. Edit it to match your needs:

```yaml
project:
  name: "my-dataset"

ingestion:
  sources:
    - path: ./data
      type: auto

synthesis:
  provider: openai
  model: gpt-4o
  max_concurrency: 5

export:
  formats: [sharegpt, alpaca]
  output_dir: ./output
```

### 2. Set Your API Key

```bash
export OPENAI_API_KEY=sk-...
```

For Ollama (local):
```bash
# Start Ollama first
ollama serve

# Then run
distill-align synthesize ... --provider ollama --base-url http://localhost:11434
```

### 3. Ingest Your Data

```bash
# Ingest a directory
distill-align ingest --source ./data --output chunks.json

# Or use auto-detection
distill-align ingest --source ./my-docs
```

### 4. Synthesize Conversations

```bash
# With OpenAI
distill-align synthesize \
    --input chunks.json \
    --provider openai \
    --model gpt-4o \
    --output conversations.json
```

### 5. Export to Training Format

```bash
distill-align export \
    --input conversations.json \
    --format sharegpt,alpaca,chatml \
    --output-dir ./output \
    --split  # Generate train/val/test
```

## Using the TUI

Launch the interactive terminal dashboard:

```bash
distill-align tui
```

Features:
- 📊 Dashboard with real-time stats
- 📋 Job management (list, resume, delete)
- 💾 Cache inspector (prune, clear)
- 🔧 Configuration viewer
- 📝 Live logs

## Advanced Usage

### Resume a Failed Job

```bash
distill-align synthesize --input chunks.json --job-id my-job-20240101 --resume
```

### Validate a Dataset

```bash
distill-align validate --input conversations.json
```

Output:
```
Validation Report
==================================================
Valid: True
Quality Score: 0.95
Errors: 0, Warnings: 2

Dataset Statistics:
  Conversations: 100
  Total Turns: 312
  Avg Turns/Conv: 3.1
  Est. Total Tokens: 45,230
  Duplicates Found: 0
```

### Custom Prompts

Create a `my_prompts/` directory with custom `.j2` files:

```
my_prompts/
├── socratic/
│   ├── system.j2
│   └── code.j2
└── scaffold/
    └── system.j2
```

### Programmatic Usage

```python
import asyncio
from distill_align.ingestion.auto import AutoIngestionPipeline
from distill_align.synthesis.pipeline import SynthesisPipeline
from distill_align.exporter.pipeline import ExportPipeline


async def main():
    # Ingest
    pipeline = AutoIngestionPipeline()
    chunks = pipeline.ingest_directory("./data")

    # Synthesize
    synth = SynthesisPipeline()
    conversations = await synth.synthesize_batch(chunks)

    # Export
    export = ExportPipeline()
    files = export.export(conversations, formats=["sharegpt", "alpaca"])

    for name, path in files.items():
        print(f"{name}: {path}")


asyncio.run(main())
```

## Next Steps

- Read the [Configuration Guide](configuration.md) for advanced options
- See [CLI Reference](cli-reference.md) for all commands
- Check [Examples](../examples/) for more use cases
