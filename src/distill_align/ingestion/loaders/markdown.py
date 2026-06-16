"""
Markdown file loader.

Handles loading and metadata extraction from Markdown files.
"""

from ...core.exceptions import LoaderError
from ...core.schemas import SourceMetadata
from .base import BaseLoader


class MarkdownLoader(BaseLoader):
    """Loader for Markdown (.md) files."""

    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd"}

    def load(self) -> str:
        """
        Load Markdown file content.

        Returns:
            Raw Markdown content as string.

        Raises:
            LoaderError: If file cannot be read.
        """
        try:
            return self.file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise LoaderError(f"Failed to read Markdown file: {e}") from e

    def extract_metadata(self) -> SourceMetadata:
        """
        Extract metadata from Markdown file.

        Returns:
            SourceMetadata with file information and extracted headers.
        """
        content = self.load()
        headers = self._extract_headers(content)
        title = headers[0] if headers else self.file_path.stem

        return SourceMetadata(
            source_type="markdown",
            file_path=str(self.file_path),
            file_name=self.file_path.name,
            title=title,
            section_headers=headers,
            custom_tags={"format": "markdown"},
        )

    def _extract_headers(self, content: str) -> list[str]:
        """
        Extract all headers from Markdown content.

        Args:
            content: Raw Markdown content.

        Returns:
            List of header strings (without # prefixes).
        """
        headers = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                # Remove # prefix and whitespace
                header = line.lstrip("#").strip()
                if header:
                    headers.append(header)
        return headers
