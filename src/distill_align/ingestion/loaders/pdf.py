"""
PDF file loader.

Handles loading and metadata extraction from PDF files.
"""

from pathlib import Path
from typing import List, Optional

from .base import BaseLoader
from ...core.schemas import DataChunk, SourceMetadata
from ...core.exceptions import LoaderError


class PDFLoader(BaseLoader):
    """Loader for PDF (.pdf) files."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, file_path: str | Path, password: Optional[str] = None):
        """
        Initialize the PDF loader.

        Args:
            file_path: Path to the PDF file.
            password: Optional password for encrypted PDFs.
        """
        super().__init__(file_path)
        self.password = password

    def load(self) -> str:
        """
        Load PDF file content.

        Returns:
            Extracted text content as string.

        Raises:
            LoaderError: If file cannot be read.
        """
        try:
            import pypdf

            reader = pypdf.PdfReader(self.file_path, password=self.password)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            raise LoaderError("pypdf is required for PDF loading. Install with: pip install pypdf")
        except Exception as e:
            raise LoaderError(f"Failed to read PDF file: {e}")

    def extract_metadata(self) -> SourceMetadata:
        """
        Extract metadata from PDF file.

        Returns:
            SourceMetadata with file information and PDF metadata.
        """
        try:
            import pypdf

            reader = pypdf.PdfReader(self.file_path, password=self.password)
            info = reader.metadata if reader.metadata else {}

            return SourceMetadata(
                source_type="pdf",
                file_path=str(self.file_path),
                file_name=self.file_path.name,
                title=getattr(info, "title", None) or self.file_path.stem,
                author=getattr(info, "author", None),
                custom_tags={
                    "format": "pdf",
                    "pages": len(reader.pages),
                    "creator": getattr(info, "creator", None),
                    "producer": getattr(info, "producer", None),
                },
            )
        except ImportError:
            # Fallback if pypdf not installed
            return SourceMetadata(
                source_type="pdf",
                file_path=str(self.file_path),
                file_name=self.file_path.name,
                title=self.file_path.stem,
                custom_tags={"format": "pdf"},
            )
        except Exception as e:
            raise LoaderError(f"Failed to extract PDF metadata: {e}")
