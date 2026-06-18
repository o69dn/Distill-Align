# Distill-Align

**The Structured Reasoning Extraction Factory**

Generate high-quality fine-tuning datasets from raw domain data using frontier reasoning models as teachers.

<div class="grid cards" markdown>

- :material-download: **Install** — `pip install distill-align`
- :material-github: **Source** — [github.com/o69dn/Distill-Align](https://github.com/o69dn/Distill-Align)
- :material-tag: **Version** — 0.1.1
- :material-license: **License** — MIT

</div>

---

## What It Does

Distill-Align transforms raw content (documents, code, PDFs, web pages) into structured conversation datasets optimized for fine-tuning LLMs. It captures the reasoning traces of frontier models and refines them into clean, multi-turn Q&A formats.

```
Raw Files  ──▶  Ingestion  ──▶  Synthesis  ──▶  Export
(PDF, MD,       (chunking)     (LLM teacher)   (ShareGPT,
 Code, HTML)                                   Alpaca, ...)
```

## Quick Example

```bash
# 1. Install
pip install distill-align

# 2. Set API key
export OPENAI_API_KEY=sk-...

# 3. Ingest your data
distill-align ingest --source ./docs --output chunks.json

# 4. Generate conversations
distill-align synthesize --input chunks.json --provider openai --model gpt-4o

# 5. Export for training
distill-align export --input conversations.json --format sharegpt --split
```

## Core Features

| Feature | Description |
|---------|-------------|
| **9 File Loaders** | Markdown, PDF, DOCX, HTML, Jupyter, JSON, CSV, Code (20+ languages), Text |
| **6 LLM Providers** | OpenAI, Anthropic, Gemini, Azure, Ollama, vLLM |
| **Socratic Transformer** | Converts raw content into guided multi-turn Q&A |
| **Scaffold Action** | Strips filler, extracts clean structured output |
| **LLM-as-Judge** | Automated quality scoring on 5 criteria |
| **7 Export Formats** | ShareGPT, Alpaca, ChatML, HuggingFace, JSONL, Parquet, DPO Pairs |
| **Job Checkpoints** | Resume failed synthesis jobs from last checkpoint |
| **Cost Tracking** | Estimate costs across all providers |
| **TUI Dashboard** | Real-time interactive monitoring |

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting Started](getting-started.md)**

    ---

    Installation, quick start, and first pipeline run.

-   :material-cog: **[Configuration](configuration.md)**

    ---

    Config file, environment variables, and advanced options.

-   :material-console: **[CLI Reference](cli-reference.md)**

    ---

    All commands, flags, and examples.

-   :material-pipe: **Pipelines**

    ---

    [Ingestion](pipelines/ingestion.md) · [Synthesis](pipelines/synthesis.md) · [Export](pipelines/export.md)

-   :material-book-open-variant: **Guides**

    ---

    [Best Practices](guides/best-practices.md) · [Contributing](guides/contributing.md) · [Troubleshooting](guides/troubleshooting.md)

-   :material-history: **[Changelog](changelog.md)**

    ---

    Release history and changes.

</div>

## Supported Providers

| Provider | API Key | Structured Output | Local |
|----------|---------|-------------------|-------|
| OpenAI | `OPENAI_API_KEY` | ✓ | — |
| Anthropic | `ANTHROPIC_API_KEY` | ✓ (JSON mode) | — |
| Google Gemini | `GEMINI_API_KEY` | ✓ (MIME type) | — |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | ✓ | — |
| Ollama | None | — | ✓ |
| vLLM | None (or API key) | ✓ (OpenAI-compatible) | ✓ |

## License

MIT License — see [LICENSE](https://github.com/o69dn/Distill-Align/blob/main/LICENSE) for details.
