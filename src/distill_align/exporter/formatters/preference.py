"""
Preference/DPO format exporter.

Exports conversations as preference pairs (chosen/rejected) for DPO/RLHF training.
Also supports pure preference datasets with scored responses.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import FormatError
from ...core.schemas import ConversationSchema
from .base import BaseFormatter


class PreferenceFormatter(BaseFormatter):
    """Formatter for DPO/Preference dataset format.

    DPO format:
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
    }

    Or with multiple responses:
    {
        "prompt": "...",
        "responses": [
            {"response": "...", "score": 0.9, "label": "chosen"},
            {"response": "...", "score": 0.3, "label": "rejected"}
        ]
    }
    """

    def __init__(
        self,
        output_dir: str | Path,
        format_type: str = "dpo",
    ):
        super().__init__(output_dir)
        self.format_type = format_type

    def format(
        self,
        conversations: list[ConversationSchema],
        filename: str = "dataset_preference.json",
    ) -> Path:
        """Format conversations into DPO/preference format.

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
            formatted = []
            for conv in conversations:
                user_turns = [t for t in conv.turns if t.role == "user"]
                assistant_turns = [t for t in conv.turns if t.role == "assistant"]

                if not user_turns or not assistant_turns:
                    continue

                prompt = user_turns[0].content

                entry: dict[str, Any]
                if self.format_type == "dpo":
                    if len(assistant_turns) >= 2:
                        logger.warning(
                            "DPO chosen/rejected assignment is order-based (first assistant turn = chosen, "
                            "second = rejected). This may not reflect actual quality differences. "
                            "Consider using a scoring mechanism for meaningful preference pairs."
                        )
                        entry = {
                            "prompt": prompt,
                            "chosen": assistant_turns[0].content,
                            "rejected": assistant_turns[1].content,
                        }
                    else:
                        entry = {
                            "prompt": prompt,
                            "chosen": assistant_turns[0].content,
                            "rejected": "",
                        }
                else:
                    responses = [
                        {
                            "response": t.content,
                            "score": 1.0 - i * 0.5,
                            "label": "chosen" if i == 0 else "rejected",
                        }
                        for i, t in enumerate(assistant_turns)
                    ]
                    entry = {
                        "prompt": prompt,
                        "responses": responses,
                    }
                formatted.append(entry)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(formatted, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(formatted)} preference pairs to {output_path}")
            return output_path
        except Exception as e:
            raise FormatError(f"Failed to format preference data: {e}") from e

    def validate(self, data: list[dict]) -> bool:
        """Validate DPO/preference format data.

        Args:
            data: Formatted data to validate.

        Returns:
            True if valid.
        """
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, dict):
                return False
            if "prompt" not in item:
                return False
            if self.format_type == "dpo":
                if "chosen" not in item or "rejected" not in item:
                    return False
            else:
                if "responses" not in item or not isinstance(item["responses"], list):
                    return False
        return True
