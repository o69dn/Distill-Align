"""
Code file loader.

Handles loading and metadata extraction from source code files.
"""

from pathlib import Path

from ...core.exceptions import LoaderError
from ...core.schemas import SourceMetadata
from .base import BaseLoader

# Language detection by extension
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".cs": "csharp",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".r": "r",
    ".R": "r",
    ".m": "matlab",
    ".lua": "lua",
    ".pl": "perl",
}


class CodeLoader(BaseLoader):
    """Loader for source code files."""

    SUPPORTED_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys())

    def __init__(self, file_path: str | Path, language: str | None = None):
        """
        Initialize the code loader.

        Args:
            file_path: Path to the code file.
            language: Optional language override (auto-detected from extension if not provided).
        """
        super().__init__(file_path)
        self.language = language or self._detect_language()

    def _detect_language(self) -> str:
        """
        Detect programming language from file extension.

        Returns:
            Detected language string.
        """
        ext = self.file_path.suffix.lower()
        return EXTENSION_TO_LANGUAGE.get(ext, "unknown")

    def load(self) -> str:
        """
        Load code file content.

        Returns:
            Raw code content as string.

        Raises:
            LoaderError: If file cannot be read.
        """
        try:
            return self.file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise LoaderError(f"Failed to read code file: {e}") from e

    def extract_metadata(self) -> SourceMetadata:
        """
        Extract metadata from code file.

        Returns:
            SourceMetadata with file information and language.
        """
        content = self.load()
        functions, classes = self._extract_definitions(content)

        return SourceMetadata(
            source_type="code",
            file_path=str(self.file_path),
            file_name=self.file_path.name,
            language=self.language,
            module_path=self._get_module_path(),
            custom_tags={
                "format": "code",
                "language": self.language,
                "functions": functions,
                "classes": classes,
                "lines": len(content.split("\n")),
            },
        )

    def _extract_definitions(self, content: str) -> tuple[list[str], list[str]]:
        """
        Extract function and class definitions from code.

        Args:
            content: Raw code content.

        Returns:
            Tuple of (function names, class names).
        """
        functions = []
        classes = []

        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("def "):
                func_name = stripped.split("(")[0].replace("def ", "").strip()
                if func_name:
                    functions.append(func_name)
            elif stripped.startswith("class "):
                class_name = stripped.split("(")[0].split(":")[0].replace("class ", "").strip()
                if class_name:
                    classes.append(class_name)

        return functions, classes

    def _get_module_path(self) -> str:
        """
        Get the module path from the file path.

        Returns:
            Module path string (e.g., "package.module").
        """
        parts = []
        current = self.file_path.parent
        while current != current.parent:
            if (current / "__init__.py").exists():
                parts.append(current.name)
            current = current.parent
        parts.reverse()
        if parts:
            return ".".join(parts) + "." + self.file_path.stem
        return self.file_path.stem
