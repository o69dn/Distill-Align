"""
Parquet formatter for columnar dataset export.

Writes conversations to Apache Parquet format using PyArrow.
Parquet is a columnar storage format optimised for large-scale ML workflows:
it is space-efficient, supports predicate pushdown, and integrates natively
with PyTorch ``DataLoader``, HuggingFace ``datasets``, and Polars.

Requires ``pyarrow`` (optional dependency).
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger

from ...core.exceptions import FormatError
from ...core.schemas import ConversationSchema
from .base import BaseFormatter

# ---------------------------------------------------------------------------
# Optional dependency — pyarrow
# ---------------------------------------------------------------------------

try:
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    _HAS_PYARROW = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MESSAGE_FIELD_NAMES = ("role", "content")
_VALID_ROLES = {"system", "user", "assistant", "tool"}


def _conversation_to_struct(conv: ConversationSchema) -> dict[str, Any]:
    """Convert a ConversationSchema to a flat dict suitable for PyArrow."""
    messages = [{"role": t.role, "content": t.content} for t in conv.turns]
    row: dict[str, Any] = {
        "id": conv.id,
        "source_chunk_id": conv.source_chunk_id,
        "messages": messages,
    }
    if conv.confidence_score is not None:
        row["confidence_score"] = conv.confidence_score
    if conv.judge_scores:
        row["judge_scores"] = conv.judge_scores
    return row


def _build_schema(
    sample: dict[str, Any],
) -> pa.Schema:
    """Infer a PyArrow schema from a sample row.

    This helper builds a schema that handles nested ``messages`` structs
    and optional ``confidence_score`` / ``judge_scores`` columns.
    """
    fields: list[pa.Field] = [
        pa.field("id", pa.string()),
        pa.field("source_chunk_id", pa.string()),
        pa.field(
            "messages",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string()),
                        pa.field("content", pa.string()),
                    ]
                ),
            ),
        ),
    ]
    if "confidence_score" in sample:
        fields.append(pa.field("confidence_score", pa.float64()))
    if "judge_scores" in sample:
        # Store as JSON string for simplicity — Parquet can store arbitrary
        # dicts via JSON string columns or structs, but structs require
        # uniform keys across all rows.
        fields.append(pa.field("judge_scores", pa.string()))
    return pa.schema(fields)


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class ParquetFormatter(BaseFormatter):
    """Formatter for Apache Parquet columnar format.

    Converts conversations to a Parquet table where each row represents
    one conversation. The ``messages`` field is stored as a nested list
    of structs.

    .. note::

       This formatter requires the ``pyarrow`` package. Install it with::

           pip install distill-align[parquet]

       or manually::

           pip install pyarrow
    """

    def __init__(self, output_dir: str | Path):
        """
        Args:
            output_dir: Directory for output files.

        Raises:
            FormatError: If ``pyarrow`` is not installed.
        """
        super().__init__(output_dir)
        if not _HAS_PYARROW:
            raise FormatError(
                "pyarrow is required for Parquet export. Install it with: pip install distill-align[parquet]"
            )

    def format(
        self,
        conversations: list[ConversationSchema],
        filename: str = "dataset.parquet",
    ) -> Path:
        """Format conversations into a Parquet file.

        Args:
            conversations: List of conversations to format.
            filename: Output filename (``.parquet`` recommended).

        Returns:
            Path to the output file.

        Raises:
            FormatError: If formatting fails or pyarrow is unavailable.
        """
        if not _HAS_PYARROW:
            raise FormatError("pyarrow is not installed")
        if not filename.endswith(".parquet"):
            filename += ".parquet"
        output_path = self.output_dir / filename

        try:
            rows = [_conversation_to_struct(c) for c in conversations]
            if not rows:
                raise FormatError("Cannot export empty conversation list to Parquet")

            schema = _build_schema(rows[0])

            # Convert judge_scores dicts to JSON strings if present
            for row in rows:
                if "judge_scores" in row and isinstance(row["judge_scores"], dict):
                    import json

                    row["judge_scores"] = json.dumps(row["judge_scores"])

            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, output_path)

            logger.info(
                f"Exported {len(conversations)} conversations to Parquet: "
                f"{output_path} ({output_path.stat().st_size / 1024:.1f} KB)"
            )
            return output_path

        except FormatError:
            raise
        except Exception as e:
            raise FormatError(f"Failed to write Parquet file: {e}") from e

    def format_stream(
        self,
        conversations: Iterable[ConversationSchema],
        filename: str = "dataset.parquet",
    ) -> Path:
        """Stream conversations to Parquet using a record batch writer.

        Writes data in batches, avoiding a full in-memory copy of the
        entire dataset. Each batch is flushed to disk incrementally.

        Args:
            conversations: Iterable of conversations (streaming).
            filename: Output filename.

        Returns:
            Path to the output file.

        Raises:
            FormatError: If formatting fails.
        """
        if not _HAS_PYARROW:
            raise FormatError("pyarrow is not installed")
        if not filename.endswith(".parquet"):
            filename += ".parquet"
        output_path = self.output_dir / filename

        try:
            writer: pq.ParquetWriter | None = None
            count = 0
            schema: pa.Schema | None = None

            for conv in conversations:
                row = _conversation_to_struct(conv)
                if "judge_scores" in row and isinstance(row["judge_scores"], dict):
                    import json

                    row["judge_scores"] = json.dumps(row["judge_scores"])

                # Infer schema from first row
                if schema is None:
                    schema = _build_schema(row)

                table = pa.Table.from_pylist([row], schema=schema)

                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema)
                writer.write_table(table)
                count += 1

            if writer is not None:
                writer.close()

            if count == 0:
                raise FormatError("Cannot export empty conversation stream to Parquet")

            file_size = output_path.stat().st_size if output_path.exists() else 0
            logger.info(f"Streamed {count} conversations to Parquet: {output_path} ({file_size / 1024:.1f} KB)")
            return output_path

        except FormatError:
            raise
        except Exception as e:
            # Ensure writer is closed on error
            if writer is not None:
                with suppress(Exception):
                    writer.close()
            raise FormatError(f"Failed to stream Parquet file: {e}") from e

    def validate(self, data: Any) -> bool:
        """Validate Parquet file contents.

        Args:
            data: A ``pyarrow.Table`` instance or a path to a parquet file.

        Returns:
            True if valid.
        """
        try:
            if isinstance(data, str | Path):
                if not _HAS_PYARROW:
                    return False
                table = pq.read_table(str(data))
            elif _HAS_PYARROW and isinstance(data, pa.Table):
                table = data
            else:
                return False

            required_cols = {"id", "source_chunk_id", "messages"}
            missing = required_cols - set(table.column_names)
            if missing:
                return False

            if table.num_rows == 0:
                return False

            # Validate messages column structure
            messages_col = table.column("messages")
            num_chunks: int = messages_col.num_chunks
            return num_chunks != 0

        except Exception:
            return False
