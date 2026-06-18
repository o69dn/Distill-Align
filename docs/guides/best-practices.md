# Best Practices

This guide covers recommended practices for getting the most out of Distill-Align.

## Data Preparation

### Source Quality

The quality of your generated dataset depends heavily on your source data:

- **Clean your sources:** Remove irrelevant content, boilerplate, and navigation elements before ingestion.
- **Prefer structured formats:** Markdown and code files produce better chunks than unstructured text or PDFs.
- **Split large documents:** Break very large files (100+ pages) into logical sections.
- **Remove duplicates:** Deduplicate source files to avoid repeated conversations.

### Directory Organization

Organize your source data by type for best results:

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

| Mode | Best For | Expected Output |
|------|----------|----------------|
| `default` | General-purpose fine-tuning | Balanced multi-turn conversations |
| `teach` | Educational/tutorial content | Progressive 5-question teaching sessions |
| `debug` | Code and error documentation | Debugging walkthroughs |
| `review` | Code quality improvement | Code review dialogues |
| `qa` | Factual/encyclopedic content | Direct Q&A pairs |
| `explain` | Complex concepts | Structured explanations |

### Mixing Modes

Run different modes on different source types for best results:

```bash
# Code files → debug mode
distill-align synthesize --input code_chunks.json --mode debug

# Documentation → teach mode
distill-align synthesize --input docs_chunks.json --mode teach

# Reference material → qa mode
distill-align synthesize --input ref_chunks.json --mode qa
```

## Quality Control with LLM-as-Judge

The `--judge` feature evaluates every conversation on 5 criteria:

| Criterion | What It Measures | Score Range |
|-----------|-----------------|-------------|
| Relevance | How well the conversation relates to source content | 0–10 |
| Coherence | Is the conversation logical and well-structured | 0–10 |
| Correctness | Is the information factually accurate | 0–10 |
| Completeness | Does it cover the key points from the source | 0–10 |
| Safety | Is the content safe and appropriate | 0–10 |

### Using the Judge

```bash
# Use a cheaper model as judge to save costs
distill-align synthesize \
    --input chunks.json \
    --provider openai --model gpt-4o \
    --judge --judge-model gpt-4o-mini \
    --output conversations.json
```

### Judge Output

Each conversation gets a `confidence_score` (0–1) and individual `judge_scores`. Use validation to filter:

```bash
# Validate and see quality scores
distill-align validate --input conversations.json

# Low-scoring conversations are flagged with warnings
```

### When to Use the Judge

- **Always** for production datasets
- When experimenting with different modes or prompts
- When the source content is critical (medical, legal, financial)
- When you need to guarantee quality before expensive training runs

## Cost Optimization

### Caching

Caching is enabled by default. The cache key is based on chunk content + model + mode:

```bash
# First run: processes all chunks
distill-align synthesize --input chunks.json --provider openai --model gpt-4o

# Re-run: skips already-processed chunks (instant)
distill-align synthesize --input chunks.json --provider openai --model gpt-4o

# Force re-processing
distill-align synthesize --input chunks.json --no-cache
```

### Rate Limiting

Match rate limits to your provider tier:

| Provider | Tier 1 | Tier 2 | Tier 3 | Local |
|----------|--------|--------|--------|-------|
| OpenAI GPT-4o | 500 RPM | 5,000 RPM | 10,000 RPM | — |
| OpenAI GPT-4o-mini | 500 RPM | 5,000 RPM | 10,000 RPM | — |
| Anthropic Claude | 50 RPM | 1,000 RPM | 4,000 RPM | — |
| Ollama | — | — | — | 1–5 RPM |
| vLLM | — | — | — | 10–50 RPM |

```bash
# Conservative settings (Tier 1)
distill-align synthesize --input chunks.json --concurrency 3 --rpm 30

# Aggressive settings (Tier 3)
distill-align synthesize --input chunks.json --concurrency 10 --rpm 200

# Local Ollama
distill-align synthesize --input chunks.json --concurrency 2 --rpm 10
```

### Checkpointing

Use job IDs for long-running synthesis jobs:

```bash
# Start with a job ID
distill-align synthesize --input chunks.json --job-id production-run-1

# If it fails, resume
distill-align synthesize --input chunks.json --job-id production-run-1 --resume

# List all jobs
distill-align jobs list
```

### Cost Estimation

Estimate costs before running synthesis:

```bash
distill-align cost-report --input chunks.json --provider openai --model gpt-4o
```

## Security Best Practices

### API Keys

**Never** pass API keys via CLI arguments. Use environment variables:

```bash
# ✓ Recommended
export OPENAI_API_KEY=sk-...
distill-align synthesize --input chunks.json

# ✗ Avoid (visible in process listings)
distill-align synthesize --input chunks.json --api-key sk-...
```

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI |
| `DISTILL_LLM_API_KEY` | Custom provider |

### PII Filtering

Enable automatic PII detection and redaction:

```yaml
# distill-align.yaml
ingestion:
  scan_pii: true
```

This detects and redacts: API keys, passwords, email addresses, phone numbers, SSNs, credit card numbers, and IP addresses.

## Performance Tuning

### Chunk Size

Adjust based on content type and model context window:

| Content Type | Recommended Chunk Size | Reasoning |
|-------------|----------------------|-----------|
| Short docs, READMEs | 500–1,000 chars | Dense information |
| Long articles | 1,500–2,000 chars | Balance context and precision |
| Code files | 1,000–1,500 chars | Function-level granularity |
| PDFs | 800–1,200 chars | Noisy OCR text needs smaller chunks |

### Concurrency

Match concurrency to your provider's capacity:

| Provider | Recommended Concurrency |
|----------|----------------------|
| OpenAI GPT-4o | 3–5 |
| OpenAI GPT-4o-mini | 5–10 |
| Anthropic Claude | 2–5 |
| Gemini | 5–10 |
| Ollama (local) | 1–3 |
| vLLM | 5–20 |

### Context Window

Set `--max-tokens` to prevent server-side truncation:

```bash
# For models with 4K output limit
distill-align synthesize --input chunks.json --max-tokens 4096

# For models with 8K+ output limit
distill-align synthesize --input chunks.json --max-tokens 8192
```

## Dataset Size Recommendations

| Use Case | Recommended Conversations | Notes |
|----------|--------------------------|-------|
| Quick prototype | 50–100 | Enough to test the pipeline |
| Fine-tuning LoRA | 500–2,000 | Good balance of quality and cost |
| Full fine-tuning | 5,000–50,000 | High quality, diverse coverage |
| Domain-specific expert | 1,000–5,000 | Focus on depth over breadth |

## Iterative Refinement

1. **Start small:** Ingest a subset, synthesize, evaluate quality
2. **Experiment with modes:** Try `teach`, `qa`, `debug` on the same data
3. **Review samples:** Manually inspect 10–20 conversations
4. **Adjust prompts:** Use custom `.j2` templates for domain-specific needs
5. **Scale up:** Once quality is good, process the full dataset
6. **Validate:** Run `distill-align validate` before training

## Logging

```bash
# Development: verbose output
distill-align --log-level DEBUG synthesize --input chunks.json

# Production: minimal output
distill-align --log-level WARNING synthesize --input chunks.json

# Quiet: errors only
distill-align --log-level ERROR synthesize --input chunks.json
```
