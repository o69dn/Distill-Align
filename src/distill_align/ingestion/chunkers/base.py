"""
Base chunker class for semantic content splitting.

All chunkers should inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List

from ...core.schemas import DataChunk, SourceMetadata
from ...core.exceptions import ChunkerError


class BaseChunker(ABC):
    """Abstract base class for content chunkers."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for each chunk (in characters or tokens).
            chunk_overlap: Overlap between consecutive chunks.
        """
        if chunk_size <= 0:
            raise ChunkerError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ChunkerError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ChunkerError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, content: str, metadata: SourceMetadata) -> List[DataChunk]:
        """
        Split content into chunks.

        Args:
            content: Raw content to split.
            metadata: Source metadata to attach to each chunk.

        Returns:
            List of DataChunk objects.

        Raises:
            ChunkerError: If chunking fails.
        """
        pass

    def _create_chunk(self, content: str, metadata: SourceMetadata, **kwargs) -> DataChunk:
        """
        Helper to create a DataChunk with updated metadata.

        Args:
            content: Chunk content.
            metadata: Base metadata.
            **kwargs: Additional metadata fields to update.

        Returns:
            DataChunk object.
        """
        # Merge any additional metadata fields
        updated_metadata = metadata.model_copy(update=kwargs)
        return DataChunk(content=content, metadata=updated_metadata)
