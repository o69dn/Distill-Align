"""
Alpaca formatter for dataset export.

Exports conversations in the Alpaca format used by Stanford Alpaca and similar projects.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import FormatError
from ...core.schemas import AlpacaEntry, ConversationSchema
from .base import BaseFormatter


class AlpacaFormatter(BaseFormatter):
    """Formatter for Alpaca JSON format."""

    def format(self, conversations: list[ConversationSchema], filename: str = "dataset_alpaca.json") -> Path:
        """
        Format conversations into Alpaca JSON.

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
                entry = self._conversation_to_alpaca(conv)
                if entry:
                    formatted_items.append(entry.model_dump())

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(formatted_items, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(formatted_items)} entries to {output_path}")
            return output_path

        except Exception as e:
            raise FormatError(f"Failed to format Alpaca: {e}") from e

    def _conversation_to_alpaca(self, conversation: ConversationSchema) -> AlpacaEntry | None:
        """
        Convert a conversation to an Alpaca entry.

        Args:
            conversation: Input conversation.

        Returns:
            AlpacaEntry or None if conversion fails.
        """
        # Extract system prompt
        system_prompt = None
        for turn in conversation.turns:
            if turn.role == "system":
                system_prompt = turn.content
                break

        # Find first user turn (instruction)
        instruction = None
        for turn in conversation.turns:
            if turn.role == "user":
                instruction = turn.content
                break

        if not instruction:
            return None

        # Find first assistant turn (output)
        output = None
        for turn in conversation.turns:
            if turn.role == "assistant":
                output = turn.content
                break

        if not output:
            return None

        # For multi-turn, combine remaining turns into output
        assistant_turns = [t for t in conversation.turns if t.role == "assistant"]
        if len(assistant_turns) > 1:
            output = "\n\n".join(t.content for t in assistant_turns)

        return AlpacaEntry(
            instruction=instruction,
            input="",  # Alpaca format has separate input field
            output=output,
            system=system_prompt,
        )

    def validate(self, data: Any) -> bool:
        """
        Validate Alpaca format data.

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
            if "instruction" not in item or "output" not in item:
                return False
            if not isinstance(item["instruction"], str) or not isinstance(item["output"], str):
                return False

        return True

    def load(self, file_path: str | Path) -> list[dict]:
        """
        Load Alpaca format data from file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            List of Alpaca entries.
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if not self.validate(data):
            raise FormatError(f"Invalid Alpaca format in {file_path}")

        return data  # type: ignore[no-any-return]
