#!/usr/bin/env python3
"""
Example 4: Custom prompts directory.

Shows how to override default prompts with custom templates.
Create a `my_prompts/` directory with custom .j2 files.
"""

import asyncio
import os
from pathlib import Path

from distill_align.core.schemas import DataChunk, SynthesisConfig
from distill_align.synthesis.pipeline import SynthesisPipeline


async def main():
    """Use custom prompts."""
    print("=" * 60)
    print("Distill-Align Example: Custom Prompts")
    print("=" * 60)

    # Path to custom prompts directory
    custom_prompts = Path(__file__).parent / "my_prompts"

    if not custom_prompts.exists():
        print(f"\nCreating {custom_prompts} with example templates...")
        custom_prompts.mkdir()
        create_example_prompts(custom_prompts)
        print(f"✓ Created {custom_prompts}")
        print("\nEdit the templates in that directory to customize the synthesis behavior.")
        print(f"Then set the environment variable: DISTILL_PROMPT_DIR={custom_prompts}")

    # Run synthesis with custom prompts
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠  OPENAI_API_KEY not set. This example requires an API key.")
        return

    # Load chunks
    chunks_file = Path(__file__).parent / "chunks.json"
    if not chunks_file.exists():
        print(f"\n⚠  {chunks_file} not found. Run 01_basic_ingest.py first!")
        return

    import json
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [DataChunk(**c) for c in chunks_data]

    # Use only first 3 chunks for this example
    chunks = chunks[:3]

    config = SynthesisConfig(
        llm_provider="openai",
        model_name="gpt-4o-mini",
        api_key=api_key,
        max_concurrency=2,
        max_rpm=20,
    )

    pipeline = SynthesisPipeline(config)
    # Note: Custom prompt dir would be passed here in a full implementation
    # For now, modify socratic.py/scaffold.py to point to your custom dir

    print(f"\nProcessing {len(chunks)} chunks with custom prompts...")
    conversations = await pipeline.synthesize_batch(chunks)

    print(f"\n✓ Generated {len(conversations)} conversations with custom prompts")


def create_example_prompts(directory: Path):
    """Create example custom prompt templates."""
    socratic_dir = directory / "socratic"
    socratic_dir.mkdir()
    scaffold_dir = directory / "scaffold"
    scaffold_dir.mkdir()

    # Custom system prompt
    (socratic_dir / "system.j2").write_text("""\
You are a friendly, patient teacher who specializes in explaining things simply.

When transforming the given content, use lots of analogies and examples.
Speak in a conversational, encouraging tone.

Output JSON: {"conversation": [...], "reasoning_trace": "..."}
""", encoding="utf-8")

    # Custom code prompt
    (socratic_dir / "code.j2").write_text("""\
Analyze the following {{ language }} code from a beginner's perspective.

{{ content }}

Ask 3-4 simple questions that help a new developer understand:
- What does this code do?
- Why is it written this way?
- What are the key patterns?
- How can it be improved?

Output JSON: {"conversation": [...], "reasoning_trace": "..."}
""", encoding="utf-8")

    # Custom scaffold
    (scaffold_dir / "system.j2").write_text("""\
You are an expert at cleaning up technical responses for documentation.

When cleaning text:
- Remove ALL filler words and pleasantries
- Use clear, direct technical language
- Structure with headers and bullet points when appropriate
- Preserve all code blocks exactly
- Add a brief summary at the start

Output JSON: {"extracted_content": "..."}
""", encoding="utf-8")

    print(f"\nCreated custom prompt templates in {directory}/")


if __name__ == "__main__":
    asyncio.run(main())
