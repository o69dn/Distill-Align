"""
JSON/JSONL file loader.

Handles loading and metadata extraction from JSON and JSONL files.
"""

import contextlib
import json

from ...core.exceptions import LoaderError
from ...core.schemas import SourceMetadata
from .base import BaseLoader


class JSONLoader(BaseLoader):
    """Loader for JSON and JSONL (.json, .jsonl) files."""

    SUPPORTED_EXTENSIONS = {".json", ".jsonl"}

    def load(self) -> str:
        """
        Load JSON/JSONL file content.

        Returns:
            Formatted JSON content as string.

        Raises:
            LoaderError: If file cannot be read.
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                content = f.read()

            if self.file_path.suffix == ".jsonl":
                return self._load_jsonl(content)
            else:
                return self._load_json(content)

        except Exception as e:
            raise LoaderError(f"Failed to read JSON file: {e}") from e

    def _load_json(self, content: str) -> str:
        """Load and format a JSON file."""
        data = json.loads(content)

        if isinstance(data, list):
            parts = []
            for i, item in enumerate(data):
                parts.append(f"[Entry {i + 1}]\n{json.dumps(item, indent=2, ensure_ascii=False)}")
            return "\n\n".join(parts)
        else:
            return json.dumps(data, indent=2, ensure_ascii=False)

    def _load_jsonl(self, content: str) -> str:
        """Load and format a JSONL file."""
        parts = []
        for i, line in enumerate(content.strip().split("\n")):
            if line.strip():
                try:
                    item = json.loads(line)
                    parts.append(f"[Entry {i + 1}]\n{json.dumps(item, indent=2, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    parts.append(f"[Line {i + 1}] {line}")
        return "\n\n".join(parts)

    def extract_metadata(self) -> SourceMetadata:
        """
        Extract metadata from JSON/JSONL file.

        Returns:
            SourceMetadata with file information.
        """
        try:
            with open(self.file_path, encoding="utf-8") as f:
                content = f.read()

            if self.file_path.suffix == ".jsonl":
                lines = [ln for ln in content.strip().split("\n") if ln.strip()]
                entry_count = len(lines)
                # Try to get keys from first entry
                sample_keys = []
                if lines:
                    with contextlib.suppress(json.JSONDecodeError):
                        sample_keys = list(json.loads(lines[0]).keys())
            else:
                data = json.loads(content)
                if isinstance(data, list):
                    entry_count = len(data)
                    sample_keys = list(data[0].keys()) if data else []
                else:
                    entry_count = 1
                    sample_keys = list(data.keys())

            return SourceMetadata(
                source_type="text",
                file_path=str(self.file_path),
                file_name=self.file_path.name,
                title=self.file_path.stem,
                custom_tags={
                    "format": "json" if self.file_path.suffix == ".json" else "jsonl",
                    "entries": entry_count,
                    "sample_keys": sample_keys[:10],
                },
            )
        except Exception as e:
            raise LoaderError(f"Failed to extract JSON metadata: {e}") from e
