"""
CSV file loader.

Handles loading and metadata extraction from CSV files.
"""

import csv
from pathlib import Path
from typing import List, Optional

from .base import BaseLoader
from ...core.schemas import SourceMetadata
from ...core.exceptions import LoaderError


class CSVLoader(BaseLoader):
    """Loader for CSV (.csv) files."""

    SUPPORTED_EXTENSIONS = {".csv"}

    def __init__(self, file_path: str | Path, delimiter: str = ",", text_column: Optional[str] = None):
        """
        Initialize the CSV loader.

        Args:
            file_path: Path to the CSV file.
            delimiter: CSV delimiter character.
            text_column: Specific column to extract as text. If None, combines all columns.
        """
        super().__init__(file_path)
        self.delimiter = delimiter
        self.text_column = text_column

    def load(self) -> str:
        """
        Load CSV file content.

        Returns:
            Formatted CSV content as string.

        Raises:
            LoaderError: If file cannot be read.
        """
        try:
            parts = []

            with open(self.file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter=self.delimiter)

                if not reader.fieldnames:
                    raise LoaderError("CSV file has no headers")

                for i, row in enumerate(reader):
                    if self.text_column:
                        # Extract specific column
                        text = row.get(self.text_column, "")
                        if text.strip():
                            parts.append(f"[Row {i + 1}] {text}")
                    else:
                        # Combine all columns
                        row_parts = []
                        for key, value in row.items():
                            if value and value.strip():
                                row_parts.append(f"{key}: {value}")
                        if row_parts:
                            parts.append(f"[Row {i + 1}]\n" + "\n".join(row_parts))

            return "\n\n".join(parts)

        except LoaderError:
            raise
        except Exception as e:
            raise LoaderError(f"Failed to read CSV file: {e}")

    def extract_metadata(self) -> SourceMetadata:
        """
        Extract metadata from CSV file.

        Returns:
            SourceMetadata with file information.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                headers = next(reader, [])
                row_count = sum(1 for _ in reader)

            return SourceMetadata(
                source_type="text",
                file_path=str(self.file_path),
                file_name=self.file_path.name,
                title=self.file_path.stem,
                custom_tags={
                    "format": "csv",
                    "columns": headers,
                    "column_count": len(headers),
                    "row_count": row_count,
                    "delimiter": self.delimiter,
                },
            )
        except Exception as e:
            raise LoaderError(f"Failed to extract CSV metadata: {e}")
