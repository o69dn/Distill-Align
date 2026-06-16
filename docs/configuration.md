# Configuration Reference

Distill-Align can be configured via a YAML or TOML file, or via environment variables.

## Config File

Create a `distill-align.yaml` in your project root:

```bash
distill-align init
```

### Full Example

```yaml
# Project metadata
project:
  name: "my-dataset"
  version: "1.0"
  description: "Fine-tuning data for our domain"

# Ingestion configuration
ingestion:
  chunk_size: 1000
  chunk_overlap: 200
  respect_headers: true
  max_chunk_tokens: 4000
  sources:
    - path: ./data
      type: auto        # auto-detect or specify: markdown, code, pdf, docx, html, jupyter, json, csv, text
      recursive: true
      patterns: []      # Optional glob patterns

# Synthesis configuration
synthesis:
  provider: openai      # openai, ollama, vllm
  model: gpt-4o
  # base_url: http://localhost:11434  # For Ollama/vLLM
  # api_key: sk-...                    # Or use env var
  max_concurrency: 5
  max_rpm: 60
  temperature: 0.7
  socratic: true
  scaffold: true
  retry_attempts: 5

# Export configuration
export:
  formats:
    - sharegpt
    - alpaca
    - chatml
  output_dir: ./output
  generate_unsloth_script: true
  train_split: 0.9
  val_split: 0.05
  test_split: 0.05
  unsloth:
    model: unsloth/Meta-Llama-3.1-8B-Instruct
    max_seq_length: 2048
    lora_rank: 16
    lora_alpha: 16
    load_in_4bit: true
    batch_size: 2
    gradient_accumulation_steps: 4
    num_epochs: 3
    learning_rate: 0.0002

# Global settings
log_level: INFO
cache_dir: .cache
```

## Environment Variables

All config options can be overridden via environment variables with the `DISTILL_` prefix:

```bash
export DISTILL_LLM_PROVIDER=ollama
export DISTILL_LLM_MODEL=llama3.1
export DISTILL_LLM_BASE_URL=http://localhost:11434
export DISTILL_LLM_API_KEY=sk-...
export DISTILL_LOG_LEVEL=DEBUG
export DISTILL_CACHE_DIR=/var/cache/distill-align
```

## CLI Override

CLI arguments override config file settings:

```bash
distill-align synthesize \
    --provider ollama \
    --model llama3.1 \
    --base-url http://localhost:11434 \
    --concurrency 3
```

## Priority Order

Settings are applied in this order (highest priority first):

1. CLI arguments
2. Environment variables
3. Config file (`distill-align.yaml`)
4. Default values

## Supported File Types

| Extension | Loader |
|-----------|--------|
| `.md`, `.markdown` | MarkdownLoader |
| `.pdf` | PDFLoader |
| `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs` | CodeLoader |
| `.ipynb` | JupyterLoader |
| `.docx` | DOCXLoader |
| `.html`, `.htm` | HTMLLoader |
| `.json`, `.jsonl` | JSONLoader |
| `.csv` | CSVLoader |
| `.txt`, `.log`, `.yaml`, `.toml` | TextLoader |
