"""
Unit tests for ingestion module.
"""

import pytest

from distill_align.core.exceptions import ChunkerError
from distill_align.core.schemas import DataChunk, SourceMetadata
from distill_align.ingestion.chunkers.code import CodeChunker
from distill_align.ingestion.chunkers.markdown import MarkdownChunker


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    def test_chunk_by_headers(self, sample_markdown_content):
        metadata = SourceMetadata(
            source_type="markdown",
            file_path="/test/file.md",
            file_name="file.md",
        )
        chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50, respect_headers=True)
        chunks = chunker.chunk(sample_markdown_content, metadata)

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DataChunk)
            assert chunk.metadata.source_type == "markdown"

    def test_chunk_by_size(self, sample_markdown_content):
        metadata = SourceMetadata(
            source_type="markdown",
            file_path="/test/file.md",
            file_name="file.md",
        )
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=20, respect_headers=False)
        chunks = chunker.chunk(sample_markdown_content, metadata)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 150  # Allow some flexibility

    def test_empty_content(self):
        metadata = SourceMetadata(
            source_type="markdown",
            file_path="/test/file.md",
            file_name="file.md",
        )
        chunker = MarkdownChunker()
        chunks = chunker.chunk("", metadata)
        assert len(chunks) == 0

    def test_invalid_config(self):
        with pytest.raises((ValueError, TypeError, ChunkerError)):
            MarkdownChunker(chunk_size=-1)

        with pytest.raises((ValueError, TypeError, ChunkerError)):
            MarkdownChunker(chunk_size=100, chunk_overlap=200)


class TestCodeChunker:
    """Tests for CodeChunker."""

    def test_chunk_python_code(self, sample_code_content):
        metadata = SourceMetadata(
            source_type="code",
            file_path="/test/file.py",
            file_name="file.py",
            language="python",
        )
        chunker = CodeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(sample_code_content, metadata)

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DataChunk)
            assert chunk.metadata.source_type == "code"

    def test_chunk_preserves_definitions(self, sample_code_content):
        metadata = SourceMetadata(
            source_type="code",
            file_path="/test/file.py",
            file_name="file.py",
            language="python",
        )
        chunker = CodeChunker(chunk_size=1000, chunk_overlap=100)
        chunks = chunker.chunk(sample_code_content, metadata)

        # Should have at least one chunk
        assert len(chunks) > 0

        # Check that definitions are preserved
        all_content = " ".join(c.content for c in chunks)
        assert "def hello_world" in all_content
        assert "def add_numbers" in all_content
        assert "class Calculator" in all_content

    def test_empty_code(self):
        metadata = SourceMetadata(
            source_type="code",
            file_path="/test/file.py",
            file_name="file.py",
            language="python",
        )
        chunker = CodeChunker()
        chunks = chunker.chunk("", metadata)
        assert len(chunks) == 0
