# Synthesis Pipeline

The synthesis pipeline transforms raw text chunks into structured, multi-turn conversations suitable for fine-tuning LLMs. It orchestrates LLM calls, applies pedagogical strategies, and manages caching and checkpointing.

## Overview

```
DataChunks (from ingestion)
        |
        v
+-----------------------+
|   LLM Client Layer     |  <- OpenAI, Ollama, vLLM
+--------+--------------+
         |
         v
+-----------------------+
|  Conversation Builder  |  <- Mode-specific strategies
|  (Socratic/Scaffold)   |
+--------+--------------+
         |
         v
+-----------------------+
|   Content Pruner       |  <- Quality and format validation
+--------+--------------+
         |
         v
+-----------------------+
| ConversationSchemas   |  <- Output JSON file
+-----------------------+
```

## LLM Providers

Distill-Align supports **6 LLM backends**:

| Provider | CLI Value | Default Base URL | Description |
|----------|-----------|-----------------|-------------|
| OpenAI | `openai` | `https://api.openai.com/v1` | GPT-4o, GPT-4, GPT-3.5, etc. |
| Ollama | `ollama` | `http://localhost:11434` | Local LLMs (Llama 3, Mistral, etc.) |
| vLLM | `vllm` | `http://localhost:8000/v1` | High-throughput serving |
| Anthropic | `anthropic` | `https://api.anthropic.com/v1` | Claude 3.5 Sonnet, Haiku, etc. |
| Gemini | `gemini` | `https://generativelanguage.googleapis.com/v1beta` | Gemini 1.5 Pro, Flash |
| Azure | `azure` | `https://api.openai.azure.com` | Azure OpenAI (GPT-4o, etc.) |

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
distill-align synthesize --input chunks.json --provider openai --model gpt-4o
```

### Ollama (Local)

```bash
ollama serve
distill-align synthesize --input chunks.json --provider ollama --model llama3.1 --base-url http://localhost:11434
```

### vLLM

```bash
distill-align synthesize --input chunks.json --provider vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct --base-url http://localhost:8000/v1
```

## Conversation Modes

Distill-Align provides **6 modes** for conversation generation:

| Mode | CLI Value | Description |
|------|-----------|-------------|
| Default | `default` | Standard Socratic + Scaffold pipeline |
| Teach | `teach` | Progressive multi-turn teaching (5 questions) |
| Debug | `debug` | Debugging session with error reproduction and fix |
| Review | `review` | Code review conversation with improvement suggestions |
| QA | `qa` | Simple question-answer pairs (3-5 pairs) |
| Explain | `explain` | Structured explanation with overview, concepts, examples, pitfalls |

### Default Mode

Uses the two-stage Socratic Transformer + Scaffold Action pipeline:
1. **Socratic Transformer**: Converts raw content into a multi-turn Q&A conversation
2. **Scaffold Action**: Strips conversational filler, extracts clean, structured output

### Custom Modes

```bash
distill-align synthesize --input chunks.json --mode teach
distill-align synthesize --input chunks.json --mode debug
distill-align synthesize --input chunks.json --mode review
distill-align synthesize --input chunks.json --mode qa
distill-align synthesize --input chunks.json --mode explain
```

## Caching

Caching avoids re-synthesizing chunks that have already been processed:

```bash
# Enable caching (default)
distill-align synthesize --input chunks.json

# Disable caching
distill-align synthesize --input chunks.json --no-cache
```

Cache stats are displayed after synthesis:

```
Cache Hit Rate: 73.5%
Cache Entries: 142
```

## Checkpointing

Checkpointing enables resuming failed or interrupted synthesis jobs:

```bash
# Start a named job
distill-align synthesize --input chunks.json --job-id my-dataset-1

# Resume if interrupted
distill-align synthesize --input chunks.json --job-id my-dataset-1 --resume
```

## Rate Limiting

Control API request rates to avoid hitting provider limits:

```bash
distill-align synthesize --input chunks.json --concurrency 3 --rpm 30
```

- `--concurrency`: Maximum concurrent requests (default: 5)
- `--rpm`: Maximum requests per minute (default: 60)

## CLI Usage

### Basic synthesis

```bash
distill-align synthesize chunks.json --output conversations.json
```

### With custom provider and model

```bash
distill-align synthesize chunks.json --provider ollama --model llama3.1
```

### With job tracking

```bash
distill-align synthesize chunks.json --job-id project-alpha --resume
```

### With LLM-as-judge evaluation

```bash
distill-align synthesize chunks.json --judge --judge-model gpt-4o-mini
```

### With Anthropic Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
distill-align synthesize chunks.json --provider anthropic --model claude-sonnet-4-20250514
```

### With Gemini

```bash
export GOOGLE_API_KEY=...
distill-align synthesize chunks.json --provider gemini --model gemini-2.0-flash
```

### With Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY=...
distill-align synthesize chunks.json --provider azure --model gpt-4o-deployment --base-url https://my-resource.openai.azure.com
```

## Cost Tracking

After synthesis completes, a cost summary panel displays estimated token usage and API cost:

```
── Cost Report ──────────────────────
  Model:               gpt-4o
  Requests:             42
  Input tokens:         85,432
  Output tokens:        12,678
  Total tokens:         98,110
  Avg tokens/request:   2,336.0
  Estimated cost (USD): $0.4321
─────────────────────────────────────
```

Costs are tracked automatically using tiktoken-based token counting and model-specific pricing tables. Pricing is resolved for all supported providers, including Azure deployment name stripping.
