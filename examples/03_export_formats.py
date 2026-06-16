#!/usr/bin/env python3
"""
Example 3: Export to multiple training formats.

Shows how to use the ExportPipeline to generate ShareGPT, Alpaca,
ChatML, and Conversation format files from synthesized conversations.
"""

import json
from pathlib import Path

from distill_align.core.schemas import ConversationSchema, ExportConfig
from distill_align.exporter.pipeline import ExportPipeline


def main():
    """Export conversations to multiple formats."""
    print("=" * 60)
    print("Distill-Align Example: Export to Multiple Formats")
    print("=" * 60)

    # Load conversations
    conv_file = Path(__file__).parent / "conversations.json"
    if not conv_file.exists():
        print(f"\n⚠  {conv_file} not found. Run 02_synthesize_openai.py first!")
        return

    with open(conv_file, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    conversations = [ConversationSchema(**c) for c in conv_data]
    print(f"\nLoaded {len(conversations)} conversations")

    # Configure export
    config = ExportConfig(
        formats=["sharegpt", "alpaca", "chatml", "conversation"],
        output_dir="./output",
        generate_unsloth_script=True,
    )

    # Create pipeline
    pipeline = ExportPipeline(config)

    # Export with validation, splitting, and card generation
    print("\nExporting with validation, splitting, and dataset card...")
    output_files = pipeline.export(
        conversations,
        formats=config.formats,
        generate_unsloth=True,
        split=True,        # Generate train/val/test splits
        generate_card=True,  # Generate HuggingFace dataset card
    )

    # Print results
    print(f"\n✓ Generated {len(output_files)} output files:")
    for name, path in output_files.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  - {name}: {path} ({size_kb:.1f} KB)")

    # Validate exports
    print("\n🔍 Validating exports...")
    validation = pipeline.validate_export(output_files)
    for name, is_valid in validation.items():
        icon = "✓" if is_valid else "✗"
        print(f"  {icon} {name}: {'valid' if is_valid else 'invalid'}")

    # Get stats
    print("\n📊 Export Statistics:")
    stats = pipeline.get_export_stats(output_files)
    for name, stat in stats.items():
        if "entries" in stat:
            print(f"  - {name}: {stat['entries']} entries, {stat['size_kb']:.1f} KB")


if __name__ == "__main__":
    main()
