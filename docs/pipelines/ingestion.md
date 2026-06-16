# Ingestion Pipeline

The ingestion pipeline is the first stage of the Distill-Align workflow. It loads raw files from disk, extracts their content, and splits them into semantic chunks that can be processed by the synthesis pipeline.

## Overview

```
Raw Files (Markdown, PDF, Code, etc.)
        |
        v
+-----------------+
|  File Loaders    |  <- Auto-detect or manual format selection
+--------+---------+
         |
         v
+-----------------+
|   Chunkers       |  <- Semantic-aware splitting
+--------+---------+
         |
         v
+-----------------+
|   DataChunks     |  <- Output JSON file
+-----------------+
```

## File Loaders

Distill-Align supports **9 file loader types** that handle different file formats:

| Loader | File Extensions | Description |
|--------|----------------|-------------|
| MarkdownLoader | `.md`, `.markdown`, `.mdown`, `.mkd` | Parses Markdown with metadata support |
| TextLoader | `.txt`, `.log`, `.cfg`, `.ini`, `.yaml`, `.yml`, `.toml`, `.rst`, `.org` | Plain text and config files |
| PDFLoader | `.pdf` | Extracts text from PDF documents |
| DOCXLoader | `.docx` | Extracts text from Word documents |
| CodeLoader | `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs`, and 20+ more | Language-aware code extraction |
| HTMLLoader | `.html`, `.htm` | Parses web pages |
| JSONLoader | `.json`, `.jsonl` | Structured data loading |
| CSVLoader | `.csv` | Tabular data loading |
| JupyterLoader | `.ipynb` | Extracts code, markdown, and outputs from notebooks |

## Chunkers

Chunkers split loaded content into manageable pieces for LLM processing:

### BaseChunker

The default chunker splits content by token count with configurable overlap.

- **Parameters**: `chunk_size` (default: 1000 characters), `chunk_overlap` (default: 200 characters)
- **Behavior**: Splits at sentence boundaries when possible

### MarkdownChunker

Header-aware chunker that respects document structure:

- Splits on `#`, `##`, `###` headers
- Preserves section context in chunk metadata
- Configurable via `respect_headers: true`

### CodeChunker

Language-aware chunker for source code:

- Splits at function/class boundaries
- Preserves imports and module-level docstrings
- Tracks function and class names in metadata

## Auto-Detection

The `AutoIngestionPipeline` automatically detects file types by extension:

```python
from distill_align.ingestion.auto import AutoIngestionPipeline

pipeline = AutoIngestionPipeline()
chunks = pipeline.ingest_directory("./data", recursive=True)
```

To disable auto-detection and use a restricted set of loaders:

```python
from distill_align.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()
chunks = pipeline.ingest_directory("./data")
```

## CLI Usage

### Ingest a directory

```bash
distill-align ingest --source ./my-docs --output chunks.json
```

### Ingest with custom chunk size

```bash
distill-align ingest --source ./data --chunk-size 2000 --overlap 300
```

### Ingest a single file

```bash
distill-align ingest ./my-docs/README.md --output chunks.json
```

### Disable auto-detection

```bash
distill-align ingest --source ./data --no-auto
```

## Output Format

Each chunk is a `DataChunk` object with the following structure:

```json
{
  "id": "uuid-v4",
  "content": "Extracted text content...",
  "metadata": {
    "source_path": "path/to/file.md",
    "source_type": "markdown",
    "title": "Document Title",
    "language": "python",
    "section_headers": ["Section 1", "Subsection A"],
    "module_path": "module.submodule",
    "functions": ["func_name"],
    "classes": ["ClassName"],
    "custom_tags": {}
  }
}
```

## Async Support

The pipeline supports async directory ingestion for large codebases:

```python
chunks = await pipeline.ingest_directory_async(
    "./large-codebase",
    recursive=True,
    max_concurrency=10,
)
```
