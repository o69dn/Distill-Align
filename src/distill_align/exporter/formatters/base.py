"""
Base formatter class for dataset export.

All formatters should inherit from this base class.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ...core.schemas import ConversationSchema


class BaseFormatter(ABC):
    """Abstract base class for dataset formatters."""

    def __init__(self, output_dir: str | Path):
        """
        Initialize the formatter.

        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def format(self, conversations: list[ConversationSchema], filename: str) -> Path:
        """
        Format conversations into the target format.

        Args:
            conversations: List of conversations to format.
            filename: Output filename.

        Returns:
            Path to the output file.

        Raises:
            FormatError: If formatting fails.
        """
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate formatted data.

        Args:
            data: Formatted data to validate.

        Returns:
            True if valid.
        """
        pass

    def format_stream(
        self,
        conversations: Iterable[ConversationSchema],
        filename: str,
    ) -> Path:
        """
        Format conversations in a streaming fashion from an iterable.

        The default implementation materialises the iterable into a list and
        delegates to :meth:`format`. Subclasses that support true streaming
        (e.g. JSONL) should override this method.

        Args:
            conversations: Iterable of conversations to format (streaming).
            filename: Output filename.

        Returns:
            Path to the output file.
        """
        return self.format(list(conversations), filename)

    def _ensure_json_extension(self, filename: str) -> str:
        """
        Ensure filename has .json extension.

        Args:
            filename: Input filename.

        Returns:
            Filename with .json extension.
        """
        if not filename.endswith(".json"):
            filename += ".json"
        return filename
