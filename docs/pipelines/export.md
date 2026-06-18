# Export Pipeline

The export pipeline converts synthesized conversations into standard training dataset formats, optionally splits them into train/val/test sets, validates quality, generates Unsloth fine-tuning scripts, and creates dataset cards.

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
|    Formatters             |  <- ShareGPT, Alpaca, ChatML, Conversation
+--------+-----------------+
         |
         v
+--------------------------+
|  Unsloth Builder         |  <- Optional train.py generation
|  Dataset Card Generator  |  <- Optional HuggingFace-style README
+--------------------------+
```

## Export Formats

Distill-Align supports **6 export formats**:

### ShareGPT

Standard multi-turn conversation format used by many fine-tuning frameworks:

```json
{
  "conversations": [
    {"from": "human", "value": "What is X?"},
    {"from": "gpt", "value": "X is..."}
  ]
}
```

### Alpaca

Instruction-following format, compatible with Stanford Alpaca:

```json
{
  "instruction": "Explain the concept of X.",
  "input": "Optional context",
  "output": "X is a concept that..."
}
```

### ChatML

Chat Markup Language format, compatible with Qwen and OpenHermes models:

```json
{
  "messages": [
    {"role": "user", "content": "What is X?"},
    {"role": "assistant", "content": "X is..."}
  ]
}
```

### Conversation

Generic conversation format:

```json
{
  "turns": [
    {"role": "user", "content": "What is X?"},
    {"role": "assistant", "content": "X is..."}
  ]
}
```

### HF Messages (HuggingFace)

HuggingFace `messages` format, compatible with `transformers` tokenizer's `apply_chat_template`:

```json
{"messages": [{"role": "user", "content": "What is X?"}, {"role": "assistant", "content": "X is..."}]}
```

One JSON object per line (JSONL) for easy streaming.

### JSONL

Newline-delimited JSON with `instruction`/`output` fields, one conversation per line:

```jsonl
{"instruction": "What is X?", "output": "X is a concept that..."}
{"instruction": "Explain Y", "output": "Y refers to..."}
```

Ideal for high-volume datasets and streaming pipelines.

### Parquet

Apache Parquet export for efficient columnar storage. Requires `pyarrow`:

```bash
pip install distill-align[parquet]
# or
pip install pyarrow

distill-align export --input conversations.json --format parquet --output-dir ./output
```

Parquet supports streaming for memory-efficient export of large datasets. Output schema includes `id`, `instruction`, `output`, `system`, and a JSON-encoded `messages` column.

## Dataset Splitting

Automatically split into train/validation/test sets:

```bash
distill-align export --input conversations.json --formats sharegpt --split
```

Default split ratios:
- **Train**: 90%
- **Validation**: 5%
- **Test**: 5%
