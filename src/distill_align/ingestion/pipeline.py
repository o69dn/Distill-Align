"""
Ingestion pipeline orchestrator.

Handles the full ingestion workflow: loading files, chunking content, and outputting DataChunks.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

from ..core.schemas import DataChunk, IngestionConfig
from ..core.exceptions import IngestionError, UnsupportedFormatError
from .loaders.base import BaseLoader
from .loaders.markdown import MarkdownLoader
from .loaders.pdf import PDFLoader
from .loaders.code import CodeLoader
from .chunkers.base import BaseChunker
from .chunkers.markdown import MarkdownChunker
from .chunkers.code import CodeChunker


# Map file extensions to loader classes
LOADER_MAP: Dict[str, type[BaseLoader]] = {
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
    """Orchestrates the ingestion of files into DataChunks."""

    def __init__(self, config: Optional[IngestionConfig] = None):
        """
        Initialize the ingestion pipeline.

        Args:
            config: Optional ingestion configuration. Uses defaults if not provided.
        """
        self.config = config or IngestionConfig()
        self._chunkers: Dict[str, BaseChunker] = {}

    def _get_loader(self, file_path: Path) -> BaseLoader:
        """
        Get the appropriate loader for a file.

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

    def _get_chunker(self, source_type: str) -> BaseChunker:
        """
        Get the appropriate chunker for a source type.

        Args:
            source_type: Type of source content.

        Returns:
            Chunker instance.
        """
        if source_type in self._chunkers:
            return self._chunkers[source_type]

        if source_type == "markdown":
            chunker = MarkdownChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                respect_headers=self.config.respect_headers,
            )
        elif source_type == "code":
            chunker = CodeChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        else:
            # Default to markdown chunker for plain text
            chunker = MarkdownChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                respect_headers=False,
            )

        self._chunkers[source_type] = chunker
        return chunker

    def ingest_file(self, file_path: str | Path) -> List[DataChunk]:
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
            chunker = self._get_chunker(metadata.source_type)
            chunks = loader.to_chunks(chunker)

            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            raise IngestionError(f"Failed to ingest {file_path}: {e}")

    def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
    ) -> List[DataChunk]:
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
            for pattern in (file_patterns or ["*"]):
                files.extend(directory.rglob(pattern))
        else:
            for pattern in (file_patterns or ["*"]):
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

    async def ingest_file_async(self, file_path: str | Path) -> List[DataChunk]:
        """
        Asynchronously ingest a single file.

        Args:
            file_path: Path to the file.

        Returns:
            List of DataChunk objects.
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.ingest_file, file_path)

    async def ingest_directory_async(
        self,
        directory: str | Path,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        max_concurrency: int = 5,
    ) -> List[DataChunk]:
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
            for pattern in (file_patterns or ["*"]):
                files.extend(directory.rglob(pattern))
        else:
            for pattern in (file_patterns or ["*"]):
                files.extend(directory.glob(pattern))

        # Filter to supported extensions
        supported_extensions = set(LOADER_MAP.keys()) | {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"}
        files = [f for f in files if f.is_file() and f.suffix.lower() in supported_extensions]

        logger.info(f"Found {len(files)} files to ingest")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)

        async def ingest_with_semaphore(file_path: Path) -> List[DataChunk]:
            async with semaphore:
                return await self.ingest_file_async(file_path)

        # Run all ingestions
        tasks = [ingest_with_semaphore(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        all_chunks = []
        for file_path, result in zip(files, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to ingest {file_path}: {result}")
            else:
                all_chunks.extend(result)

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
