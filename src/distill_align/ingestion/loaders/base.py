"""
Base loader class for file ingestion.

All file loaders should inherit from this base class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ...core.schemas import DataChunk, SourceMetadata
from ...core.exceptions import LoaderError


class BaseLoader(ABC):
    """Abstract base class for file loaders."""

    def __init__(self, file_path: str | Path):
        """
        Initialize the loader.

        Args:
            file_path: Path to the file to load.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise LoaderError(f"File not found: {self.file_path}")

    @abstractmethod
    def load(self) -> str:
        """
        Load the file content.

        Returns:
            Raw file content as string.

        Raises:
            LoaderError: If loading fails.
        """
        pass

    @abstractmethod
    def extract_metadata(self) -> SourceMetadata:
        """
        Extract metadata from the file.

        Returns:
            SourceMetadata object with file information.
        """
        pass

    def to_chunks(self, chunker: "BaseChunker") -> List[DataChunk]:
        """
        Load file and split into chunks using the provided chunker.

        Args:
            chunker: Chunker instance to split the content.

        Returns:
            List of DataChunk objects.
        """
        content = self.load()
        metadata = self.extract_metadata()
        return chunker.chunk(content, metadata)
