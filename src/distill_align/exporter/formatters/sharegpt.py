"""
ShareGPT formatter for dataset export.

Exports conversations in the ShareGPT format used by many fine-tuning tools.
"""

import json
from pathlib import Path
from typing import List, Any

from loguru import logger

from .base import BaseFormatter
from ...core.schemas import ConversationSchema, ShareGPTMessage, ShareGPTItem
from ...core.exceptions import FormatError


class ShareGPTFormatter(BaseFormatter):
    """Formatter for ShareGPT JSON format."""

    # Role mapping from internal format to ShareGPT format
    ROLE_MAP = {
        "system": "system",
        "user": "human",
        "assistant": "gpt",
        "tool": "tool",
    }

    def format(self, conversations: List[ConversationSchema], filename: str = "dataset_sharegpt.json") -> Path:
        """
        Format conversations into ShareGPT JSON.

        Args:
            conversations: List of conversations to format.
            filename: Output filename.

        Returns:
            Path to the output file.

        Raises:
            FormatError: If formatting fails.
        """
        filename = self._ensure_json_extension(filename)
        output_path = self.output_dir / filename

        try:
            formatted_items = []

            for conv in conversations:
                messages = []
                for turn in conv.turns:
                    role = self.ROLE_MAP.get(turn.role, "gpt")
                    messages.append(ShareGPTMessage(from_=role, value=turn.content))

                item = ShareGPTItem(
                    id=conv.id,
                    conversations=messages,
                )
                formatted_items.append(item.model_dump(by_alias=True))

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(formatted_items, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(formatted_items)} conversations to {output_path}")
            return output_path

        except Exception as e:
            raise FormatError(f"Failed to format ShareGPT: {e}")

    def validate(self, data: Any) -> bool:
        """
        Validate ShareGPT format data.

        Args:
            data: Data to validate.

        Returns:
            True if valid.
        """
        if not isinstance(data, list):
            return False

        for item in data:
            if not isinstance(item, dict):
                return False
            if "id" not in item or "conversations" not in item:
                return False
            if not isinstance(item["conversations"], list):
                return False

            for msg in item["conversations"]:
                if "from" not in msg or "value" not in msg:
                    return False
                if msg["from"] not in ("human", "gpt", "system", "tool"):
                    return False

        return True

    def load(self, file_path: str | Path) -> List[dict]:
        """
        Load ShareGPT format data from file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            List of ShareGPT items.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not self.validate(data):
            raise FormatError(f"Invalid ShareGPT format in {file_path}")

        return data
