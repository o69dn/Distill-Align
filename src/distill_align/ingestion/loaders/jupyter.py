"""
Jupyter Notebook loader.

Handles loading and metadata extraction from .ipynb files.
"""

import json

from ...core.exceptions import LoaderError
from ...core.schemas import SourceMetadata
from .base import BaseLoader


class JupyterLoader(BaseLoader):
    """Loader for Jupyter Notebook (.ipynb) files."""

    SUPPORTED_EXTENSIONS = {".ipynb"}

    def load(self) -> str:
        """
        Load Jupyter notebook content.

        Extracts code cells and markdown cells as structured text.

        Returns:
            Extracted content as string.

        Raises:
            LoaderError: If file cannot be read.
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                notebook = json.load(f)

            parts = []

            for i, cell in enumerate(notebook.get("cells", [])):
                cell_type = cell.get("cell_type", "code")
                source = "".join(cell.get("source", []))

                if not source.strip():
                    continue

                if cell_type == "markdown":
                    parts.append(f"[Markdown Cell {i + 1}]\n{source}")
                elif cell_type == "code":
                    parts.append(f"[Code Cell {i + 1}]\n```python\n{source}\n```")

                    # Include outputs if they contain text
                    for output in cell.get("outputs", []):
                        if output.get("output_type") == "stream":
                            text = "".join(output.get("text", []))
                            if text.strip():
                                parts.append(f"[Output]\n{text}")
                        elif output.get("output_type") == "execute_result":
                            text = "".join(output.get("data", {}).get("text/plain", []))
                            if text.strip():
                                parts.append(f"[Output]\n{text}")

            return "\n\n".join(parts)

        except Exception as e:
            raise LoaderError(f"Failed to read Jupyter notebook: {e}") from e

    def extract_metadata(self) -> SourceMetadata:
        """
        Extract metadata from Jupyter notebook.

        Returns:
            SourceMetadata with file information.
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                notebook = json.load(f)

            metadata = notebook.get("metadata", {})
            kernelspec = metadata.get("kernelspec", {})
            language_info = metadata.get("language_info", {})

            cells = notebook.get("cells", [])
            code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
            markdown_cells = sum(1 for c in cells if c.get("cell_type") == "markdown")

            # Try to get title from first markdown cell
            title = self.file_path.stem
            for cell in cells:
                if cell.get("cell_type") == "markdown":
                    source = "".join(cell.get("source", []))
                    # Look for first heading
                    for line in source.split("\n"):
                        if line.startswith("#"):
                            title = line.lstrip("#").strip()
                            break
                    break

            return SourceMetadata(
                source_type="code",
                file_path=str(self.file_path),
                file_name=self.file_path.name,
                title=title,
                language=language_info.get("name", "python"),
                custom_tags={
                    "format": "jupyter",
                    "kernel": kernelspec.get("display_name", "Python"),
                    "language_version": language_info.get("version", ""),
                    "total_cells": len(cells),
                    "code_cells": code_cells,
                    "markdown_cells": markdown_cells,
                },
            )
        except Exception as e:
            raise LoaderError(f"Failed to extract Jupyter metadata: {e}") from e
