# Export Pipeline

The export pipeline converts synthesized conversations into standard training dataset formats, validates quality, splits into train/val/test sets, and optionally generates Unsloth training scripts and dataset cards.

## Overview

```
ConversationSchemas (from synthesis)
        |
        v
+-----------------+
|  Validator       |  <- Dedup, quality checks, stats
+--------+---------+
         |
         v
+-----------------+
|   Splitter       |  <- Optional train/val/test split
+--------+---------+
         |
         v
+--------------------------+
|    Formatters             |  <- ShareGPT, Alpaca, ChatML, etc.
+--------+-----------------+
         |
         v
+--------------------------+
|  Unsloth Builder         |  <- Optional train.py generation
|  Dataset Card Generator  |  <- Optional HuggingFace-style README
+--------------------------+
```

## CLI Usage

### Basic Export

```bash
# Single format
distill-align export --input conversations.json --format sharegpt

# Multiple formats
distill-align export --input conversations.json --format sharegpt,alpaca,chatml

# With train/val/test split
distill-align export --input conversations.json --format sharegpt --split

# With dataset card
distill-align export --input conversations.json --format sharegpt --split --card

# Custom output directory
distill-align export --input conversations.json --format sharegpt --output-dir ./dataset
```

### Parquet Export

```bash
# Requires pyarrow
pip install distill-align[parquet]

distill-align export --input conversations.json --format parquet --output-dir ./output
```

### DPO Preference Pairs

```bash
# Requires --judge from synthesis step
distill-align export --input conversations.json --format preference --output-dir ./dpo
```

## Export Formats

### ShareGPT

Standard multi-turn conversation format used by many fine-tuning frameworks (LLaMA-Factory, Axolotl):

```json
{
  "conversations": [
    {"from": "human", "value": "What is overfitting in machine learning?"},
    {"from": "gpt", "value": "Overfitting occurs when a model learns the training data too well, including noise and patterns that don't generalize to new data..."}
  ]
}
```

### Alpaca

Instruction-following format, compatible with Stanford Alpaca and Alpaca-LoRA:

```json
{
  "instruction": "Explain the concept of overfitting in machine learning.",
  "input": "",
  "output": "Overfitting occurs when a model learns the training data too well, including noise and patterns that don't generalize to new data...",
  "system": "You are a helpful AI assistant."
}
```

### ChatML

Chat Markup Language format, compatible with Qwen, OpenHermes, and Nous Research models:

```json
{
  "messages": [
    {"role": "user", "content": "What is overfitting?"},
    {"role": "assistant", "content": "Overfitting occurs when..."}
  ]
}
```

### HuggingFace Messages

HuggingFace `messages` format, compatible with `transformers` tokenizer's `apply_chat_template`:

```json
{"messages": [{"role": "user", "content": "What is overfitting?"}, {"role": "assistant", "content": "Overfitting occurs when..."}]}
```

One JSON object per line (JSONL) for easy streaming and large dataset support.

### JSONL

Newline-delimited JSON with `instruction`/`output` fields:

```jsonl
{"instruction": "What is overfitting?", "output": "Overfitting occurs when..."}
{"instruction": "Explain regularization", "output": "Regularization is a technique..."}
```

Ideal for high-volume datasets and streaming pipelines.

### Parquet

Apache Parquet columnar format for efficient storage and processing:

```bash
pip install distill-align[parquet]
distill-align export --input conversations.json --format parquet
```

Schema includes `id`, `instruction`, `output`, `system`, and a JSON-encoded `messages` column. Supports streaming for memory-efficient export of large datasets.

### Conversation (Raw)

Generic conversation format preserving the original schema:

```json
{
  "turns": [
    {"role": "user", "content": "What is overfitting?"},
    {"role": "assistant", "content": "Overfitting occurs when..."}
  ],
  "metadata": {
    "source": "ml-guide.md",
    "mode": "teach",
    "confidence_score": 0.92
  }
}
```

### DPO Preference Pairs

Generates chosen/rejected pairs for Direct Preference Optimization training. Requires the `--judge` flag during synthesis to provide quality scores:

```json
{
  "chosen": {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "High quality response..."}]},
  "rejected": {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "Lower quality response..."}]}
}
```

## Dataset Splitting

Automatically split into train/validation/test sets:

```bash
distill-align export --input conversations.json --format sharegpt --split
```

Default ratios:

| Split | Default Ratio | Purpose |
|-------|---------------|---------|
| Train | 90% | Model training |
| Validation | 5% | Hyperparameter tuning |
| Test | 5% | Final evaluation |

Output files follow the naming convention: `{format}_{split}.json` (e.g., `sharegpt_train.json`).

## Unsloth Training Scripts

Generate optimized `train.py` scripts for Unsloth:

```bash
distill-align export --input conversations.json --format sharegpt --unsloth
```

This creates a `train.py` file configured for your dataset, ready to run with Unsloth for efficient fine-tuning.

## Dataset Cards

Generate a HuggingFace-style dataset card:

```bash
distill-align export --input conversations.json --format sharegpt --split --card
```

This creates a `README.md` with dataset description, statistics, supported formats, and usage instructions.

## Validation

The export pipeline automatically validates your dataset:

1. **Structural checks** — minimum turns, user/assistant alternation
2. **Quality scoring** — filler phrase detection, short turn warnings
3. **Deduplication** — SHA256-based exact duplicate removal
4. **Statistics** — token estimates, role distribution, length histogram

Validation errors block the export. Warnings are reported but don't prevent output.

## Streaming Export

For very large datasets, use streaming to avoid loading everything into memory:

```python
from distill_align.exporter.pipeline import ExportPipeline

pipeline = ExportPipeline()

# Process conversations from an iterator
async for batch in conversation_generator():
    pipeline.export_stream(batch, formats=["jsonl"], output_dir="./output")
```

## Format Selection Guide

| Use Case | Recommended Format | Notes |
|----------|-------------------|-------|
| LLaMA-Factory / Axolotl | `sharegpt` | Most widely supported |
| Alpaca-LoRA / Stanford Alpaca | `alpaca` | Simple instruction format |
| Qwen / OpenHermes | `chatml` | Model-specific format |
| HuggingFace transformers | `hf_messages` | Direct `apply_chat_template` support |
| Large datasets / BigQuery | `parquet` | Columnar, efficient, queryable |
| Custom pipelines | `jsonl` | Easy to process line by line |
| DPO training | `preference` | Chosen/rejected pairs |
