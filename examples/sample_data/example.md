# Sample Markdown Document for Testing

This is a sample document for testing the Distill-Align pipeline.

## Introduction

Distill-Align is a powerful tool for generating fine-tuning datasets from raw domain data. It uses frontier reasoning models as teachers to transform your content into structured, instruction-following formats.

## Key Features

### Smart Ingestion

The pipeline supports multiple file formats including:
- Markdown files
- PDF documents
- Source code files (Python, JavaScript, etc.)
- Jupyter notebooks
- HTML pages
- JSON/CSV data

### Synthesis

The synthesis module uses the "Socratic Transformer" pattern to:
1. Convert raw content into guided multi-turn conversations
2. Apply the "Scaffold Action" to extract clean, structured output
3. Cache results for resume support
4. Track progress with checkpoints

### Export

The export module generates training-ready files in multiple formats:
- ShareGPT
- Alpaca
- ChatML
- Generic Conversation format

## Usage Example

```python
from distill_align.ingestion.auto import AutoIngestionPipeline
from distill_align.synthesis.pipeline import SynthesisPipeline
from distill_align.exporter.pipeline import ExportPipeline

# Ingest
pipeline = AutoIngestionPipeline()
chunks = pipeline.ingest_directory("./data")

# Synthesize
synth = SynthesisPipeline()
conversations = await synth.synthesize_batch(chunks)

# Export
export = ExportPipeline()
files = export.export(conversations, formats=["sharegpt", "alpaca"])
```

## Conclusion

Distill-Align makes it easy to convert your domain knowledge into high-quality training data for fine-tuning language models.
