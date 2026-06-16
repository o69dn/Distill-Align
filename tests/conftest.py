"""
Pytest configuration and fixtures for Distill-Align tests.
"""

import pytest
from pathlib import Path


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for testing."""
    return """# Introduction

This is a sample markdown document for testing purposes.

## Section 1

Here is some content in section 1.

### Subsection 1.1

More detailed content here.

## Section 2

Another section with different content.
"""


@pytest.fixture
def sample_code_content():
    """Sample Python code content for testing."""
    return '''def hello_world():
    """Print hello world."""
    print("Hello, World!")


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a: int, b: int) -> int:
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
'''


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_chunks():
    """Sample data chunks for testing."""
    from distill_align.core.schemas import DataChunk, SourceMetadata

    chunks = []
    for i in range(3):
        metadata = SourceMetadata(
            source_type="markdown",
            file_path="/test/file.md",
            file_name="file.md",
            title="Test Document",
            section_headers=[f"Section {i+1}"],
        )
        chunk = DataChunk(
            content=f"This is test chunk {i+1} with some sample content.",
            metadata=metadata,
        )
        chunks.append(chunk)
    return chunks
