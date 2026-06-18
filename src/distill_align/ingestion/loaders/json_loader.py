"""
JSON/JSONL file loader.

Handles loading and metadata extraction from JSON and JSONL files.
All JSON parsing is size-bounded via :mod:`distill_align.core.json_utils`.
"""

import contextlib
import json

from ...core.exceptions import LoaderError
from ...core.json_utils import safe_json_loads
from ...core.schemas import SourceMetadata
from .base import BaseLoader


class JSONLoader(BaseLoader):
    """Loader for JSON and JSONL (.json, .jsonl) files."""

    SUPPORTED_EXTENSIONS = {".json", ".jsonl"}

    # Max file size: 200 MB (same as core.json_utils)
    MAX_FILE_BYTES = 200 * 1024 * 1024

    def load(self) -> str:
        """
        Load JSON/JSONL file content.

        Returns:
            Formatted JSON content as string.

        Raises:
            LoaderError: If file cannot be read or is too large.
        """
        try:
            # Guard against unbounded file reads
            file_size = self.file_path.stat().st_size
            if file_size > self.MAX_FILE_BYTES:
                raise LoaderError(
                    f"JSON file too large: {file_size / (1024 * 1024):.1f} MB "
                    f"(max {self.MAX_FILE_BYTES / (1024 * 1024):.0f} MB)"
                )

            with open(self.file_path, encoding="utf-8") as f:
                content = f.read()

            if self.file_path.suffix == ".jsonl":
                return self._load_jsonl(content)
            else:
                return self._load_json(content)

        except LoaderError:
            raise
        except Exception as e:
            raise LoaderError(f"Failed to read JSON file: {e}") from e

    def _load_json(self, content: str) -> str:
        """Load and format a JSON file."""
        data = safe_json_loads(content, label=self.file_path.name)

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
                    item = safe_json_loads(line, label=f"{self.file_path.name}:L{i + 1}")
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
            # Guard against unbounded file reads
            file_size = self.file_path.stat().st_size
            if file_size > self.MAX_FILE_BYTES:
                return SourceMetadata(
                    source_type="text",
                    file_path=str(self.file_path),
                    file_name=self.file_path.name,
                    title=self.file_path.stem,
                    custom_tags={
                        "format": "json" if self.file_path.suffix == ".json" else "jsonl",
                        "error": f"File too large ({file_size / (1024 * 1024):.1f} MB)",
                    },
                )

            with open(self.file_path, encoding="utf-8") as f:
                content = f.read()

            if self.file_path.suffix == ".jsonl":
                lines = [ln for ln in content.strip().split("\n") if ln.strip()]
                entry_count = len(lines)
                # Try to get keys from first entry
                sample_keys = []
                if lines:
                    with contextlib.suppress(json.JSONDecodeError):
                        _entry = safe_json_loads(lines[0], label=f"{self.file_path.name}:L1")
                        sample_keys = list(_entry.keys()) if isinstance(_entry, dict) else []
            else:
                data = safe_json_loads(content, label=self.file_path.name)
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
