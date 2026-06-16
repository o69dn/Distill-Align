"""
ChatML format exporter.

Exports conversations in ChatML format used by Qwen, OpenHermes, and others.
"""

import json
from pathlib import Path
from typing import List

from loguru import logger

from .base import BaseFormatter
from ...core.schemas import ConversationSchema
from ...core.exceptions import FormatError


class ChatMLFormatter(BaseFormatter):
    """Formatter for ChatML JSON format."""

    # ChatML uses different role names
    ROLE_MAP = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }

    def format(
        self,
        conversations: List[ConversationSchema],
        filename: str = "dataset_chatml.json",
    ) -> Path:
        """
        Format conversations into ChatML JSON.

        ChatML format:
        {
            "id": "...",
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
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
                messages = [
                    {"role": self.ROLE_MAP.get(t.role, "assistant"), "content": t.content}
                    for t in conv.turns
                ]
                formatted.append({
                    "id": conv.id,
                    "messages": messages,
                })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(formatted, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(formatted)} conversations to ChatML: {output_path}")
            return output_path

        except Exception as e:
            raise FormatError(f"Failed to format ChatML: {e}")

    def validate(self, data: List[dict]) -> bool:
        """Validate ChatML format data."""
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, dict):
                return False
            if "id" not in item or "messages" not in item:
                return False
            if not isinstance(item["messages"], list):
                return False
            for msg in item["messages"]:
                if "role" not in msg or "content" not in msg:
                    return False
        return True
