"""
HTML file loader.

Handles loading and metadata extraction from HTML files.
"""

from ...core.exceptions import LoaderError
from ...core.schemas import SourceMetadata
from .base import BaseLoader


class HTMLLoader(BaseLoader):
    """Loader for HTML (.html, .htm) files."""

    SUPPORTED_EXTENSIONS = {".html", ".htm"}

    def load(self) -> str:
        """
        Load HTML file content and extract text.

        Returns:
            Extracted text content as string.

        Raises:
            LoaderError: If file cannot be read.
        """
        try:
            from bs4 import BeautifulSoup

            with open(self.file_path, encoding="utf-8") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Extract text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            return "\n\n".join(lines)

        except ImportError:
            raise LoaderError(
                "beautifulsoup4 is required for HTML loading. Install with: pip install beautifulsoup4"
            ) from None
        except Exception as e:
            raise LoaderError(f"Failed to read HTML file: {e}") from e

    def extract_metadata(self) -> SourceMetadata:
        """
        Extract metadata from HTML file.

        Returns:
            SourceMetadata with file information.
        """
        try:
            from bs4 import BeautifulSoup

            with open(self.file_path, encoding="utf-8") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Extract title
            title = None
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)

            # Extract meta tags
            meta_tags = {}
            for meta in soup.find_all("meta"):
                name = meta.get("name") or meta.get("property", "")
                content = meta.get("content", "")
                if name and content:
                    meta_tags[name] = content

            # Extract headers
            headers = []
            for tag in soup.find_all(["h1", "h2", "h3"]):
                headers.append(tag.get_text(strip=True))

            return SourceMetadata(
                source_type="text",
                file_path=str(self.file_path),
                file_name=self.file_path.name,
                title=title or self.file_path.stem,
                author=meta_tags.get("author"),
                section_headers=headers,
                custom_tags={
                    "format": "html",
                    "description": meta_tags.get("description", ""),
                    "keywords": meta_tags.get("keywords", ""),
                },
            )
        except ImportError:
            return SourceMetadata(
                source_type="text",
                file_path=str(self.file_path),
                file_name=self.file_path.name,
                title=self.file_path.stem,
                custom_tags={"format": "html"},
            )
        except Exception as e:
            raise LoaderError(f"Failed to extract HTML metadata: {e}") from e
