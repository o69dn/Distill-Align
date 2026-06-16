"""
Ingestion pipeline orchestrator.

Handles the full ingestion workflow: loading files, chunking content, and outputting DataChunks.
This pipeline wraps AutoIngestionPipeline for manual file-type selection (non-auto-detect mode).
"""

import asyncio
from pathlib import Path

from loguru import logger

from ..core.exceptions import IngestionError, UnsupportedFormatError
from ..core.schemas import DataChunk, IngestionConfig
from .auto import AutoIngestionPipeline
from .loaders.base import BaseLoader
from .loaders.code import CodeLoader
from .loaders.markdown import MarkdownLoader
from .loaders.pdf import PDFLoader

# Map file extensions to loader classes (non-auto subset)
LOADER_MAP: dict[str, type[BaseLoader]] = {
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".mdown": MarkdownLoader,
    ".mkd": MarkdownLoader,
    ".pdf": PDFLoader,
}

# Add code extensions
for ext in CodeLoader.SUPPORTED_EXTENSIONS if hasattr(CodeLoader, "SUPPORTED_EXTENSIONS") else []:
    LOADER_MAP[ext] = CodeLoader


class IngestionPipeline:
    """
    Orchestrates the ingestion of files into DataChunks.

    This is a wrapper around AutoIngestionPipeline that restricts loaders
    to a manually-configured subset for non-auto-detect mode.
    """

    def __init__(self, config: IngestionConfig | None = None):
        """
        Initialize the ingestion pipeline.

        Args:
            config: Optional ingestion configuration. Uses defaults if not provided.
        """
        self.config = config or IngestionConfig()
        self._auto = AutoIngestionPipeline(config)

    def _get_loader(self, file_path: Path) -> BaseLoader:
        """
        Get the appropriate loader for a file (restricted subset).

        Args:
            file_path: Path to the file.

        Returns:
            Loader instance.

        Raises:
            UnsupportedFormatError: If file format is not supported.
        """
        ext = file_path.suffix.lower()

        if ext in LOADER_MAP:
            return LOADER_MAP[ext](file_path)

        # Try to detect if it's a code file
        code_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".rs", ".go", ".rb"}
        if ext in code_extensions:
            return CodeLoader(file_path)

        raise UnsupportedFormatError(f"Unsupported file format: {ext}")

    def ingest_file(self, file_path: str | Path) -> list[DataChunk]:
        """
        Ingest a single file.

        Args:
            file_path: Path to the file.

        Returns:
            List of DataChunk objects.

        Raises:
            IngestionError: If ingestion fails.
        """
        file_path = Path(file_path)
        logger.info(f"Ingesting file: {file_path}")

        try:
            loader = self._get_loader(file_path)
            metadata = loader.extract_metadata()
            chunker = self._auto.get_chunker(metadata.source_type)
            chunks = loader.to_chunks(chunker)

            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            raise IngestionError(f"Failed to ingest {file_path}: {e}") from e

    def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        file_patterns: list[str] | None = None,
    ) -> list[DataChunk]:
        """
        Ingest all supported files in a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to search subdirectories.
            file_patterns: Optional list of glob patterns to filter files.

        Returns:
            List of DataChunk objects from all files.
        """
        directory = Path(directory)
        logger.info(f"Ingesting directory: {directory}")

        if not directory.is_dir():
            raise IngestionError(f"Not a directory: {directory}")

        # Collect files
        files = []
        if recursive:
            for pattern in file_patterns or ["*"]:
                files.extend(directory.rglob(pattern))
        else:
            for pattern in file_patterns or ["*"]:
                files.extend(directory.glob(pattern))

        # Filter to supported extensions
        supported_extensions = set(LOADER_MAP.keys()) | {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"}
        files = [f for f in files if f.is_file() and f.suffix.lower() in supported_extensions]

        logger.info(f"Found {len(files)} files to ingest")

        # Ingest each file
        all_chunks = []
        for file_path in files:
            try:
                chunks = self.ingest_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
                continue

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    async def ingest_file_async(self, file_path: str | Path) -> list[DataChunk]:
        """
        Asynchronously ingest a single file.

        Args:
            file_path: Path to the file.

        Returns:
            List of DataChunk objects.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.ingest_file, file_path)

    async def ingest_directory_async(
        self,
        directory: str | Path,
        recursive: bool = True,
        file_patterns: list[str] | None = None,
        max_concurrency: int = 5,
    ) -> list[DataChunk]:
        """
        Asynchronously ingest all files in a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to search subdirectories.
            file_patterns: Optional list of glob patterns.
            max_concurrency: Maximum concurrent file ingestions.

        Returns:
            List of DataChunk objects from all files.
        """
        directory = Path(directory)
        logger.info(f"Ingesting directory asynchronously: {directory}")

        # Collect files
        files = []
        if recursive:
            for pattern in file_patterns or ["*"]:
                files.extend(directory.rglob(pattern))
        else:
            for pattern in file_patterns or ["*"]:
                files.extend(directory.glob(pattern))

        # Filter to supported extensions
        supported_extensions = set(LOADER_MAP.keys()) | {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"}
        files = [f for f in files if f.is_file() and f.suffix.lower() in supported_extensions]

        logger.info(f"Found {len(files)} files to ingest")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)

        async def ingest_with_semaphore(file_path: Path) -> list[DataChunk]:
            async with semaphore:
                return await self.ingest_file_async(file_path)

        # Run all ingestions
        tasks = [ingest_with_semaphore(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        all_chunks = []
        for file_path, result in zip(files, results, strict=False):
            if isinstance(result, Exception):
                logger.warning(f"Failed to ingest {file_path}: {result}")
            else:
                all_chunks.extend(result)

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
