"""
Generic text file loader.

Handles loading and metadata extraction from plain text files.
"""

from pathlib import Path
from typing import List, Optional

from .base import BaseLoader
from ...core.schemas import SourceMetadata
from ...core.exceptions import LoaderError


class TextLoader(BaseLoader):
    """Loader for plain text (.txt, .log, .cfg, .ini, .yaml, .yml, .toml) files."""

    SUPPORTED_EXTENSIONS = {".txt", ".log", ".cfg", ".ini", ".yaml", ".yml", ".toml", ".rst", ".org"}

    def load(self) -> str:
        """
        Load text file content.

        Returns:
            File content as string.

        Raises:
            LoaderError: If file cannot be read.
        """
        try:
            return self.file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                return self.file_path.read_text(encoding="latin-1")
            except Exception as e:
                raise LoaderError(f"Failed to read text file with fallback encoding: {e}")
        except Exception as e:
            raise LoaderError(f"Failed to read text file: {e}")

    def extract_metadata(self) -> SourceMetadata:
        """
        Extract metadata from text file.

        Returns:
            SourceMetadata with file information.
        """
        try:
            content = self.load()
            lines = content.split("\n")

            return SourceMetadata(
                source_type="text",
                file_path=str(self.file_path),
                file_name=self.file_path.name,
                title=self.file_path.stem,
                custom_tags={
                    "format": "text",
                    "extension": self.file_path.suffix,
                    "lines": len(lines),
                    "characters": len(content),
                },
            )
        except Exception as e:
            raise LoaderError(f"Failed to extract text metadata: {e}")
