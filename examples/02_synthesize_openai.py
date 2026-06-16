#!/usr/bin/env python3
"""
Example 2: Synthesize conversations with OpenAI.

This example shows how to use the SynthesisPipeline to generate
structured conversations from chunks using OpenAI's API.
"""

import asyncio
import json
import os
from pathlib import Path

from distill_align.core.cache import CacheManager
from distill_align.core.checkpoint import CheckpointManager
from distill_align.core.schemas import DataChunk, SynthesisConfig
from distill_align.synthesis.pipeline import SynthesisPipeline


async def main():
    """Synthesize conversations using OpenAI."""
    print("=" * 60)
    print("Distill-Align Example: Synthesize with OpenAI")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠  OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        return

    # Load chunks (or generate sample ones)
    chunks_file = Path(__file__).parent / "chunks.json"
    if not chunks_file.exists():
        print(f"\n⚠  {chunks_file} not found. Run 01_basic_ingest.py first!")
        return

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [DataChunk(**c) for c in chunks_data]
    print(f"\nLoaded {len(chunks)} chunks")

    # Configure synthesis
    config = SynthesisConfig(
        llm_provider="openai",
        model_name="gpt-4o-mini",  # Use mini for cost
        api_key=api_key,
        max_concurrency=3,
        max_rpm=30,
        temperature=0.7,
        socratic_enabled=True,
        scaffold_enabled=True,
    )

    # Create pipeline with cache and checkpoint support
    cache = CacheManager(cache_dir=".cache")
    checkpoint = CheckpointManager()

    pipeline = SynthesisPipeline(
        config=config,
        cache_manager=cache,
        checkpoint_manager=checkpoint,
    )

    # Create or resume job
    job = checkpoint.create_job("synthesize", total_items=len(chunks))
    print(f"Created job: {job.job_id}")

    # Synthesize
    print(f"\nSynthesizing {len(chunks)} chunks...")

    def progress_cb(current, total):
        pct = (current / total) * 100
        print(f"  Progress: {current}/{total} ({pct:.0f}%)")

    conversations = await pipeline.synthesize_batch(
        chunks,
        progress_callback=progress_cb,
        job_id=job.job_id,
    )

    # Save results
    output_file = Path(__file__).parent / "conversations.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in conversations], f, indent=2, ensure_ascii=False)

    print(f"\n✓ Generated {len(conversations)} conversations")
    print(f"✓ Saved to {output_file}")

    # Print cache stats
    stats = cache.stats()
    print(f"\n📊 Cache Statistics:")
    print(f"  - Entries: {stats.total_entries}")
    print(f"  - Hit rate: {stats.hit_rate:.1%}")
    print(f"  - DB size: {stats.db_size_mb} MB")

    # Print sample conversation
    if conversations:
        conv = conversations[0]
        print(f"\n--- Sample conversation ---")
        for turn in conv.turns:
            print(f"  [{turn.role}]: {turn.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
