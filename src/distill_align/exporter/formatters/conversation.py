"""
Generic Conversation format exporter.

Exports in a flexible generic format that can be converted to most other formats.
"""

import json
from pathlib import Path
from typing import List

from loguru import logger

from .base import BaseFormatter
from ...core.schemas import ConversationSchema
from ...core.exceptions import FormatError


class ConversationFormatter(BaseFormatter):
    """Formatter for generic Conversation JSON format."""

    def format(
        self,
        conversations: List[ConversationSchema],
        filename: str = "dataset_conversation.json",
    ) -> Path:
        """
        Format conversations into a generic JSON format.

        Format:
        {
            "id": "...",
            "source_chunk_id": "...",
            "turns": [
                {"role": "system|user|assistant|tool", "content": "..."}
            ],
            "reasoning_trace": "...",
            "metadata": {...}
        }

        Args:
            conversations: List of conversations.
            filename: Output filename.

        Returns:
            Path to output file.

        Raises:
            FormatError: If formatting fails.
        """
        filename = self._ensure_json_extension(filename)
        output_path = self.output_dir / filename

        try:
            formatted = []
            for conv in conversations:
                formatted.append({
                    "id": conv.id,
                    "source_chunk_id": conv.source_chunk_id,
                    "turns": [
                        {"role": t.role, "content": t.content}
                        for t in conv.turns
                    ],
                    "reasoning_trace": conv.reasoning_trace,
                    "metadata": {
                        "turn_count": len(conv.turns),
                        "confidence_score": conv.confidence_score,
                    },
                })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(formatted, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(formatted)} conversations: {output_path}")
            return output_path

        except Exception as e:
            raise FormatError(f"Failed to format Conversation: {e}")

    def validate(self, data: List[dict]) -> bool:
        """Validate generic conversation format data."""
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, dict):
                return False
            if "id" not in item or "turns" not in item:
                return False
            if not isinstance(item["turns"], list):
                return False
        return True
