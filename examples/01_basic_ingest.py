#!/usr/bin/env python3
"""
Example 1: Basic ingestion of Markdown files.

This example shows how to use the AutoIngestionPipeline to ingest
a directory of Markdown files and split them into semantic chunks.
"""

from pathlib import Path
from distill_align.ingestion.auto import AutoIngestionPipeline
from distill_align.core.schemas import IngestionConfig


def main():
    """Basic ingestion example."""
    print("=" * 60)
    print("Distill-Align Example: Basic Ingestion")
    print("=" * 60)

    # Configure the pipeline
    config = IngestionConfig(
        chunk_size=1000,
        chunk_overlap=200,
        respect_headers=True,
    )

    # Create the auto-detection pipeline
    pipeline = AutoIngestionPipeline(config)

    # Ingest a directory (use the sample_data directory in this repo)
    source_dir = Path(__file__).parent / "sample_data"
    if not source_dir.exists():
        print(f"Creating sample data directory: {source_dir}")
        source_dir.mkdir()

    print(f"\nScanning: {source_dir}")

    # Show what files would be ingested
    files = pipeline.scan_directory(source_dir, recursive=True)
    print(f"\nFound {len(files)} supported files:")
    for f in files[:5]:
        print(f"  - {f.relative_to(source_dir)}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

    # Ingest all files
    chunks = pipeline.ingest_directory(source_dir, recursive=True)

    print(f"\n✓ Created {len(chunks)} chunks from {len(files)} files")

    # Show a sample chunk
    if chunks:
        print("\n--- Sample chunk ---")
        print(f"ID: {chunks[0].id}")
        print(f"Source: {chunks[0].metadata.file_name}")
        print(f"Type: {chunks[0].metadata.source_type}")
        print(f"Content preview: {chunks[0].content[:200]}...")

    # Save chunks to JSON
    import json
    output_file = Path(__file__).parent / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved chunks to {output_file}")


if __name__ == "__main__":
    main()
