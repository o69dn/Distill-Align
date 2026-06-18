"""
HuggingFace messages format exporter.

Exports conversations in the standard HuggingFace messages format used by
TRL, the transformers library, and the datasets library for fine-tuning.

Format:
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import FormatError
from ...core.schemas import ConversationSchema
from .base import BaseFormatter

VALID_ROLES = {"system", "user", "assistant", "tool"}


class HFMessagesFormatter(BaseFormatter):
    """Formatter for HuggingFace messages JSON format.

    Produces the standard ``{"messages": [{"role": …, "content": …}, …]}``
    format accepted by ``datasets.Dataset``, TRL trainers, and
    ``transformers`` chat templates.

    The output is a JSONL file (one JSON object per line) for efficient
    streaming, or a standard JSON array depending on the *jsonl* parameter.
    """

    def __init__(
        self,
        output_dir: str | Path,
        jsonl: bool = True,
    ):
        """
        Args:
            output_dir: Directory for output files.
            jsonl: If True, write one JSON object per line (JSONL).
                If False, write a JSON array. JSONL is the recommended
                format for HuggingFace ``datasets``.
        """
        super().__init__(output_dir)
        self.jsonl = jsonl

    def format(
        self,
        conversations: list[ConversationSchema],
        filename: str = "dataset_hf_messages.jsonl",
    ) -> Path:
        """Format conversations into HuggingFace messages format.

        Each conversation becomes::

            {"messages": [{"role": …, "content": …}, …]}

        Args:
            conversations: List of conversations to format.
            filename: Output filename (``.jsonl`` recommended).

        Returns:
            Path to the output file.

        Raises:
            FormatError: If formatting fails.
        """
        if not filename.endswith(".jsonl") and not filename.endswith(".json"):
            filename += ".jsonl" if self.jsonl else ".json"
        output_path = self.output_dir / filename

        try:
            records: list[dict[str, Any]] = []
            for conv in conversations:
                messages = [{"role": t.role, "content": t.content} for t in conv.turns]
                records.append({"messages": messages})

            with open(output_path, "w", encoding="utf-8") as f:
                if self.jsonl:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    json.dump(records, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Exported {len(records)} conversations to HF messages format: "
                f"{output_path} ({'JSONL' if self.jsonl else 'JSON'})"
            )
            return output_path

        except Exception as e:
            raise FormatError(f"Failed to format HF messages: {e}") from e

    def validate(self, data: Any) -> bool:
        """Validate HuggingFace messages format data.

        Args:
            data: Parsed data to validate (list of dicts for JSON array,
                  or a single dict for JSONL).

        Returns:
            True if valid.
        """
        # Normalise to a list, regardless of JSONL vs JSON array input
        items: list[dict[str, Any]] = data if isinstance(data, list) else [data]

        if not items:
            return False

        for item in items:
            if not isinstance(item, dict):
                return False
            if "messages" not in item:
                return False
            msgs = item["messages"]
            if not isinstance(msgs, list) or len(msgs) == 0:
                return False
            for msg in msgs:
                if not isinstance(msg, dict):
                    return False
                if "role" not in msg or "content" not in msg:
                    return False
                if msg["role"] not in VALID_ROLES:
                    return False
                if not isinstance(msg["content"], str):
                    return False
        return True
