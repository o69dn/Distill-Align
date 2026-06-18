"""
JSONL (JSON Lines) formatter for streaming dataset export.

Writes one JSON object per line, following the JSON Lines convention.
This is ideal for large datasets as it supports streaming writes and
line-by-line reads without loading the entire dataset into memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import FormatError
from ...core.schemas import ConversationSchema
from .base import BaseFormatter


class JsonlFormatter(BaseFormatter):
    """Formatter for JSON Lines (JSONL) format.

    Each conversation is serialised as a single JSON object on one line::

        {"id": "...", "source_chunk_id": "...", "turns": [...], ...}
        {"id": "...", "source_chunk_id": "...", "turns": [...], ...}

    This formatter supports **streaming** via :meth:`format_stream`, which
    writes each conversation as it is received from an iterable without
    buffering the full dataset in memory.
    """

    def __init__(
        self,
        output_dir: str | Path,
        flatten: bool = False,
        include_metadata: bool = True,
    ):
        """
        Args:
            output_dir: Directory for output files.
            flatten: If True, include conversation fields at the top level
                alongside ``"messages"`` instead of nesting.
            include_metadata: If True, include source_chunk_id, confidence_score,
                and judge_scores in the output.
        """
        super().__init__(output_dir)
        self.flatten = flatten
        self.include_metadata = include_metadata

    def _conversation_to_dict(self, conv: ConversationSchema) -> dict[str, Any]:
        """Convert a single conversation to a JSON-serialisable dict."""
        record: dict[str, Any] = {
            "messages": [
                {"role": t.role, "content": t.content}
                for t in conv.turns
            ],
        }

        if self.flatten:
            record["id"] = conv.id
            record["source_chunk_id"] = conv.source_chunk_id
            if conv.confidence_score is not None:
                record["confidence_score"] = conv.confidence_score
            if self.include_metadata and conv.judge_scores:
                record["judge_scores"] = conv.judge_scores
        elif self.include_metadata:
            record["id"] = conv.id
            record["source_chunk_id"] = conv.source_chunk_id
            if conv.confidence_score is not None:
                record["confidence_score"] = conv.confidence_score
            if conv.judge_scores:
                record["judge_scores"] = conv.judge_scores

        return record

    def format(
        self,
        conversations: list[ConversationSchema],
        filename: str = "dataset.jsonl",
    ) -> Path:
        """Format conversations into JSONL format.

        Args:
            conversations: List of conversations to format.
            filename: Output filename (``.jsonl`` recommended).

        Returns:
            Path to the output file.

        Raises:
            FormatError: If formatting fails.
        """
        if not filename.endswith(".jsonl"):
            filename += ".jsonl"
        output_path = self.output_dir / filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for conv in conversations:
                    record = self._conversation_to_dict(conv)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(f"Exported {len(conversations)} conversations to JSONL: {output_path}")
            return output_path

        except Exception as e:
            raise FormatError(f"Failed to write JSONL file: {e}") from e

    def format_stream(
        self,
        conversations: Iterable[ConversationSchema],
        filename: str = "dataset.jsonl",
    ) -> Path:
        """Stream conversations to JSONL from an iterable.

        Writes each conversation as it arrives, making this suitable for
        large datasets that do not fit in memory.

        Args:
            conversations: Iterable of conversations (streaming).
            filename: Output filename.

        Returns:
            Path to the output file.
        """
        if not filename.endswith(".jsonl"):
            filename += ".jsonl"
        output_path = self.output_dir / filename

        try:
            count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for conv in conversations:
                    record = self._conversation_to_dict(conv)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

            logger.info(f"Streamed {count} conversations to JSONL: {output_path}")
            return output_path

        except Exception as e:
            raise FormatError(f"Failed to stream JSONL file: {e}") from e

    def validate(self, data: Any) -> bool:
        """Validate JSONL data.

        Args:
            data: Parsed data to validate (list of dicts for JSON array,
                  or a single dict for JSONL line).

        Returns:
            True if valid.
        """
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
                if msg["role"] not in {"system", "user", "assistant", "tool"}:
                    return False
                if not isinstance(msg["content"], str):
                    return False
        return True
