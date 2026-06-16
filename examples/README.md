# Examples

This directory contains example scripts showing how to use Distill-Align.

## Available Examples

| Script | Description |
|--------|-------------|
| `01_basic_ingest.py` | Basic file ingestion and chunking |
| `02_synthesize_openai.py` | Synthesize conversations with OpenAI |
| `03_export_formats.py` | Export to multiple training formats |
| `04_custom_prompts.py` | Use custom prompt templates |
| `05_full_pipeline.py` | Complete end-to-end pipeline |

## Running Examples

```bash
# Set your OpenAI API key (required for synthesis)
export OPENAI_API_KEY=sk-...

# Run individual examples
python examples/01_basic_ingest.py
python examples/02_synthesize_openai.py
python examples/03_export_formats.py
python examples/04_custom_prompts.py
python examples/05_full_pipeline.py
```

## Sample Data

The `sample_data/` directory contains example files for testing.
You can replace these with your own data to generate custom datasets.

## Output

Examples generate outputs in the project root:
- `chunks.json` — Ingested chunks
- `conversations.json` — Synthesized conversations
- `./output/` — Exported training files
- `.cache/` — Synthesis cache
- `.distill-align/jobs/` — Job checkpoints
