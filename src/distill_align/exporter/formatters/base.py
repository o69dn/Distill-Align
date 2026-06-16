"""
Base formatter class for dataset export.

All formatters should inherit from this base class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any

from ...core.schemas import ConversationSchema
from ...core.exceptions import FormatError


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
    def format(self, conversations: List[ConversationSchema], filename: str) -> Path:
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
