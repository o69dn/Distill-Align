"""
Markdown chunker with header-aware semantic splitting.

Splits Markdown content based on header structure rather than static token lengths.
"""

import re

from ...core.schemas import DataChunk, SourceMetadata
from .base import BaseChunker


class MarkdownChunker(BaseChunker):
    """Chunker for Markdown content with header-aware splitting."""

    # Regex pattern for Markdown headers
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        respect_headers: bool = True,
        max_header_level: int = 3,
    ):
        """
        Initialize the Markdown chunker.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Overlap between consecutive chunks.
            respect_headers: Whether to split on headers.
            max_header_level: Maximum header level to split on (1-6).
        """
        super().__init__(chunk_size, chunk_overlap)
        self.respect_headers = respect_headers
        self.max_header_level = min(max(max_header_level, 1), 6)

    def chunk(self, content: str, metadata: SourceMetadata) -> list[DataChunk]:
        """
        Split Markdown content into chunks.

        Args:
            content: Raw Markdown content.
            metadata: Source metadata.

        Returns:
            List of DataChunk objects.
        """
        if not content.strip():
            return []

        if self.respect_headers:
            return self._chunk_by_headers(content, metadata)
        else:
            return self._chunk_by_size(content, metadata)

    def _chunk_by_headers(self, content: str, metadata: SourceMetadata) -> list[DataChunk]:
        """
        Split content by Markdown headers.

        Args:
            content: Raw Markdown content.
            metadata: Source metadata.

        Returns:
            List of DataChunk objects.
        """
        chunks = []
        sections = self._split_by_headers(content)

        for header, section_content in sections:
            if not section_content.strip():
                continue

            # If section is small enough, create a single chunk
            if len(section_content) <= self.chunk_size:
                updated_metadata = metadata.model_copy(
                    update={
                        "section_headers": metadata.section_headers + [header] if header else metadata.section_headers
                    }
                )
                chunk = DataChunk(content=section_content.strip(), metadata=updated_metadata)
                chunks.append(chunk)
            else:
                # Section is too large, split by size with overlap
                sub_chunks = self._chunk_by_size(section_content, metadata)
                # Update section headers for sub-chunks
                for sub_chunk in sub_chunks:
                    if header:
                        sub_chunk.metadata.section_headers = sub_chunk.metadata.section_headers + [header]
                chunks.extend(sub_chunks)

        return chunks

    def _split_by_headers(self, content: str) -> list[tuple[str, str]]:
        """
        Split content into (header, content) pairs.

        Args:
            content: Raw Markdown content.

        Returns:
            List of (header, section_content) tuples.
        """
        sections = []
        current_header = ""
        current_content = []

        for line in content.split("\n"):
            match = self.HEADER_PATTERN.match(line)
            if match:
                level = len(match.group(1))
                header_text = match.group(2)

                # Only split on headers up to max_header_level
                if level <= self.max_header_level:
                    # Save previous section
                    if current_content:
                        sections.append((current_header, "\n".join(current_content)))

                    current_header = header_text
                    current_content = []
                    continue

            current_content.append(line)

        # Save last section
        if current_content:
            sections.append((current_header, "\n".join(current_content)))

        return sections

    def _chunk_by_size(self, content: str, metadata: SourceMetadata) -> list[DataChunk]:
        """
        Split content by size with overlap.

        Args:
            content: Raw content.
            metadata: Source metadata.

        Returns:
            List of DataChunk objects.
        """
        chunks = []
        start = 0

        while start < len(content):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence end
                for sep in [". ", ".\n", "!\n", "?\n", "\n\n"]:
                    sep_pos = content.rfind(sep, start, end)
                    if sep_pos > start:
                        end = sep_pos + len(sep)
                        break

            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk = DataChunk(content=chunk_content, metadata=metadata)
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(content):
                break

        return chunks
