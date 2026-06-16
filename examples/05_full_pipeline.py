#!/usr/bin/env python3
"""
Example 5: Full end-to-end pipeline.

This example shows the complete Distill-Align workflow:
1. Initialize project (generate config)
2. Ingest files
3. Synthesize conversations
4. Validate dataset
5. Export to multiple formats
6. Generate dataset card
"""

import asyncio
import json
import os
from pathlib import Path

from distill_align.core.cache import CacheManager
from distill_align.core.checkpoint import CheckpointManager
from distill_align.core.config_file import generate_default_config
from distill_align.core.schemas import (
    ConversationSchema,
    DataChunk,
    ExportConfig,
    IngestionConfig,
    SynthesisConfig,
)
from distill_align.exporter.pipeline import ExportPipeline
from distill_align.exporter.validator import DatasetValidator
from distill_align.ingestion.auto import AutoIngestionPipeline
from distill_align.synthesis.pipeline import SynthesisPipeline


async def main():
    """Run the full pipeline end-to-end."""
    print("=" * 60)
    print("Distill-Align: Full End-to-End Pipeline")
    print("=" * 60)

    # ============================================================
    # Step 1: Initialize project
    # ============================================================
    print("\n[1/6] Initializing project...")
    config_path = Path("distill-align.yaml")
    if not config_path.exists():
        generate_default_config("my-dataset", str(config_path))
        print(f"  ✓ Created {config_path}")
    else:
        print(f"  - {config_path} already exists")

    # ============================================================
    # Step 2: Ingest files
    # ============================================================
    print("\n[2/6] Ingesting files...")

    source_dir = Path(__file__).parent / "sample_data"
    if not (source_dir / "example.md").exists():
        print(f"  ! Sample data not found at {source_dir}")
        return

    config = IngestionConfig(chunk_size=1000, chunk_overlap=200)
    pipeline = AutoIngestionPipeline(config)
    chunks = pipeline.ingest_directory(source_dir)
    print(f"  ✓ Created {len(chunks)} chunks")

    # ============================================================
    # Step 3: Synthesize conversations
    # ============================================================
    print("\n[3/6] Synthesizing conversations...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ! OPENAI_API_KEY not set, skipping synthesis")
        return

    synth_config = SynthesisConfig(
        llm_provider="openai",
        model_name="gpt-4o-mini",
        api_key=api_key,
        max_concurrency=3,
        max_rpm=30,
    )

    cache = CacheManager(cache_dir=".cache")
    checkpoint = CheckpointManager()
    synth_pipeline = SynthesisPipeline(
        config=synth_config,
        cache_manager=cache,
        checkpoint_manager=checkpoint,
    )

    job = checkpoint.create_job("synthesize", total_items=len(chunks))
    print(f"  - Job: {job.job_id}")

    conversations = await synth_pipeline.synthesize_batch(
        chunks, job_id=job.job_id,
    )
    print(f"  ✓ Generated {len(conversations)} conversations")

    # ============================================================
    # Step 4: Validate
    # ============================================================
    print("\n[4/6] Validating dataset...")

    validator = DatasetValidator()
    conversations = validator.deduplicate(conversations)
    report = validator.validate(conversations)
    print(f"  ✓ Quality score: {report.quality_score:.2f}")
    print(f"  - Total turns: {report.stats.total_turns}")
    print(f"  - Estimated tokens: {report.stats.estimated_total_tokens:,}")
    print(f"  - Duplicates removed: {report.stats.duplicate_count}")

    # ============================================================
    # Step 5: Export
    # ============================================================
    print("\n[5/6] Exporting to training formats...")

    export_config = ExportConfig(
        formats=["sharegpt", "alpaca", "chatml"],
        output_dir="./output",
        generate_unsloth_script=True,
    )
    export_pipeline = ExportPipeline(export_config)

    output_files = export_pipeline.export(
        conversations,
        formats=export_config.formats,
        generate_unsloth=True,
        split=True,
        generate_card=True,
    )

    for name, path in output_files.items():
        if path.exists():
            print(f"  ✓ {name}: {path.name}")

    # ============================================================
    # Step 6: Summary
    # ============================================================
    print("\n[6/6] Summary")
    print("=" * 60)
    print(f"  Chunks: {len(chunks)}")
    print(f"  Conversations: {len(conversations)}")
    print(f"  Quality: {report.quality_score:.1%}")
    print(f"  Output files: {len(output_files)}")
    print(f"  Output dir: ./output")
    print(f"  Cache: {cache.stats().total_entries} entries")
    print("\n✓ Pipeline complete! Ready for training with Unsloth.")


if __name__ == "__main__":
    asyncio.run(main())
