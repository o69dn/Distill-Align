# Best Practices

This guide covers recommended practices for getting the most out of Distill-Align.

## Data Preparation

### Source Quality

The quality of your generated dataset depends heavily on your source data quality:

- **Clean your sources**: Remove irrelevant content, boilerplate, and navigation elements from HTML sources before ingestion.
- **Prefer structured formats**: Markdown and code files produce better chunks than unstructured text or PDFs.
- **Split large documents**: Break very large files (100+ pages) into logical sections before ingestion.
- **Remove duplicates**: Deduplicate your source files to avoid repeated conversations in the final dataset.

### Directory Organization

Organize your source data by type:

```
./data/
├── docs/
│   ├── guide.md
│   └── reference.md
├── code/
│   ├── api.py
│   └── utils.js
└── articles/
    └── introduction.pdf
```

## Choosing the Right Conversation Mode

### When to Use Each Mode

| Mode | Best For | Expected Output |
|------|----------|----------------|
| `default` | General-purpose fine-tuning | Balanced multi-turn conversations |
| `teach` | Educational/tutorial content | Progressive 5-question teaching sessions |
| `debug` | Code and error documentation | Debugging walkthroughs |
| `review` | Code quality improvement | Code review dialogues |
| `qa` | Factual/encyclopedic content | Direct Q&A pairs |
| `explain` | Complex concepts | Structured explanations |

### Mixing Modes

For the best results, consider running different modes on different source types:

```bash
# Code files -> debug mode
distill-align synthesize --input code_chunks.json --mode debug

# Documentation -> teach mode
distill-align synthesize --input docs_chunks.json --mode teach
```

## Cost Optimization

### Caching

Always enable caching to avoid re-processing chunks:

```bash
distill-align synthesize --input chunks.json  # Cache enabled by default
```

Cache hit rates improve significantly on re-runs after fixing prompt or mode changes.

### Rate Limiting

Set conservative rate limits to avoid API errors and unnecessary retries:

```bash
# For OpenAI GPT-4o (Tier 1: 500 RPM)
distill-align synthesize --input chunks.json --concurrency 5 --rpm 60

# For local Ollama
distill-align synthesize --input chunks.json --concurrency 2 --rpm 30
```

### Checkpointing

Use job IDs for long-running synthesis jobs:

```bash
distill-align synthesize --input chunks.json --job-id large-dataset-1
```

If the job fails, resume from the last checkpoint:

```bash
distill-align synthesize --input chunks.json --job-id large-dataset-1 --resume
```

## Security Best Practices

### API Keys

**Never** pass API keys via CLI arguments. Use environment variables instead:

```bash
# Recommended
export OPENAI_API_KEY=sk-...
distill-align synthesize --input chunks.json

# Avoid (key visible in process listings)
distill-align synthesize --input chunks.json --api-key sk-...
```

Supported environment variables:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key |
| `DISTILL_LLM_API_KEY` | Custom LLM API key |
| `DISTILL_LLM_BASE_URL` | Custom base URL |
| `DISTILL_LOG_LEVEL` | Log level (DEBUG, INFO, etc.) |

## Performance Tuning

### Chunk Size

Adjust chunk size based on your content and model context window:

| Content Type | Recommended Chunk Size | Reasoning |
|-------------|----------------------|-----------|
| Short docs, readmes | 500-1000 chars | Dense information, fewer tokens |
| Long articles | 1500-2000 chars | Balance context and precision |
| Code files | 1000-1500 chars | Function-level granularity |
| PDFs | 800-1200 chars | Noisy OCR text needs smaller chunks |

### Concurrency

Match concurrency to your provider:

| Provider | Recommended Concurrency |
|----------|----------------------|
| OpenAI GPT-4o | 3-5 |
| OpenAI GPT-4o-mini | 5-10 |
| Ollama (local) | 1-3 |
| vLLM | 5-20 |

### Logging

Use DEBUG logging during development:

```bash
distill-align --log-level DEBUG ingest --source ./data
```

For production, use INFO or WARNING:

```bash
distill-align --log-level WARNING synthesize --input chunks.json
```
