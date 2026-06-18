"""
Export pipeline orchestrator.

Handles the full export workflow: validation, splitting, formatting, streaming,
and Unsloth config.
"""

from collections.abc import Iterable
from itertools import tee
from pathlib import Path
from typing import Any, cast

from loguru import logger

from ..core.exceptions import ExportError
from ..core.schemas import ConversationSchema, ExportConfig
from .dataset_card import DatasetCardGenerator
from .formatters.alpaca import AlpacaFormatter
from .formatters.base import BaseFormatter
from .formatters.chatml import ChatMLFormatter
from .formatters.conversation import ConversationFormatter
from .formatters.hf_messages import HFMessagesFormatter
from .formatters.jsonl import JsonlFormatter
from .formatters.parquet import ParquetFormatter
from .formatters.sharegpt import ShareGPTFormatter
from .splitter import DatasetSplitter
from .unsloth_builder import UnslothConfigBuilder
from .validator import DatasetValidator, ValidationReport

# Map format names to formatter classes
FORMATTER_MAP: dict[str, type[BaseFormatter]] = {
    "sharegpt": ShareGPTFormatter,
    "alpaca": AlpacaFormatter,
    "chatml": ChatMLFormatter,
    "conversation": ConversationFormatter,
    "hf_messages": HFMessagesFormatter,
    "jsonl": JsonlFormatter,
    "parquet": ParquetFormatter,
}


class ExportPipeline:
    """Orchestrates the export of conversations to various formats."""

    def __init__(self, config: ExportConfig | None = None):
        """
        Initialize the export pipeline.

        Args:
            config: Optional export configuration. Uses defaults if not provided.
        """
        self.config = config or ExportConfig()
        self._formatters: dict[str, BaseFormatter] = {}
        self._unsloth_builder: UnslothConfigBuilder | None = None
        self._validator = DatasetValidator()
        self._splitter = DatasetSplitter()
        self._card_generator = DatasetCardGenerator()

    def _get_formatter(self, format_name: str) -> BaseFormatter:
        """
        Get or create a formatter for the specified format.

        Args:
            format_name: Name of the format (e.g., "sharegpt", "alpaca", "chatml").

        Returns:
            Formatter instance.

        Raises:
            ExportError: If format is not supported.
        """
        if format_name not in FORMATTER_MAP:
            raise ExportError(f"Unsupported format: {format_name}. Supported: {', '.join(FORMATTER_MAP.keys())}")

        if format_name not in self._formatters:
            self._formatters[format_name] = FORMATTER_MAP[format_name](self.config.output_dir)
        return self._formatters[format_name]

    def _get_unsloth_builder(self) -> UnslothConfigBuilder:
        """Get or create the Unsloth config builder."""
        if self._unsloth_builder is None:
            self._unsloth_builder = UnslothConfigBuilder(self.config)
        return self._unsloth_builder

    def validate(
        self,
        conversations: list[ConversationSchema],
        dedupe: bool = True,
    ) -> tuple[list[ConversationSchema], ValidationReport]:
        """
        Validate and optionally deduplicate conversations.

        Args:
            conversations: List of conversations.
            dedupe: Whether to remove duplicates.

        Returns:
            Tuple of (cleaned conversations, validation report).
        """
        if dedupe:
            conversations = self._validator.deduplicate(conversations)

        report = self._validator.validate(conversations)
        logger.info(f"Validation: {len(conversations)} conversations, score={report.quality_score:.2f}")
        return conversations, report

    def export(
        self,
        conversations: list[ConversationSchema],
        formats: list[str] | None = None,
        dataset_filename: str = "dataset",
        generate_unsloth: bool = True,
        split: bool = False,
        generate_card: bool = False,
        **unsloth_kwargs,
    ) -> dict[str, Path]:
        """
        Export conversations to specified formats.

        Args:
            conversations: List of conversations.
            formats: List of format names (defaults to config).
            dataset_filename: Base filename for datasets.
            generate_unsloth: Whether to generate Unsloth script.
            split: Whether to split into train/val/test.
            generate_card: Whether to generate dataset card.
            **unsloth_kwargs: Additional Unsloth config.

        Returns:
            Dictionary mapping format names to output file paths.
        """
        export_formats = cast("list[str]", self.config.formats if formats is None else formats)

        # Validate first
        conversations, validation_report = self.validate(conversations)

        # Optional split
        if split:
            split_result = self._splitter.split(
                conversations,
                train_ratio=0.9,
                val_ratio=0.05,
                test_ratio=0.05,
            )
            split_paths = self._splitter.save_split(split_result, self.config.output_dir, dataset_filename)
            # Use train split as the primary dataset for training
            conversations = split_result.train
        else:
            split_paths = {}

        output_files = {}

        # Export to each format
        for format_name in export_formats:
            try:
                formatter = self._get_formatter(format_name)
                filename = f"{dataset_filename}_{format_name}.json"
                output_path = formatter.format(conversations, filename)
                output_files[format_name] = output_path
                logger.info(f"Exported to {format_name}: {output_path}")
            except Exception as e:
                logger.error(f"Failed to export to {format_name}: {e}")
                raise ExportError(f"Export to {format_name} failed: {e}") from e

        # Add split files to output
        output_files.update(split_paths)

        # Generate Unsloth script
        if generate_unsloth and self.config.generate_unsloth_script:
            try:
                builder = self._get_unsloth_builder()
                dataset_path = str(list(output_files.values())[0])
                script_path = builder.generate_script(
                    dataset_path=dataset_path,
                    output_dir=str(Path(self.config.output_dir) / "model"),
                    **unsloth_kwargs,
                )
                output_files["unsloth_script"] = script_path
                logger.info(f"Generated Unsloth script: {script_path}")
            except Exception as e:
                logger.warning(f"Failed to generate Unsloth script: {e}")

        # Generate dataset card
        if generate_card:
            try:
                card_path = Path(self.config.output_dir) / f"{dataset_filename}_README.md"
                self._card_generator.generate(
                    conversations=conversations,
                    validation_report=validation_report,
                    config=unsloth_kwargs.get("synthesis_config", {}),
                    output_path=card_path,
                )
                output_files["dataset_card"] = card_path
            except Exception as e:
                logger.warning(f"Failed to generate dataset card: {e}")

        return output_files

    def export_stream(
        self,
        conversations: Iterable[ConversationSchema],
        formats: list[str] | None = None,
        dataset_filename: str = "dataset",
    ) -> dict[str, Path]:
        """Export conversations in a streaming fashion.

        Unlike :meth:`export`, this method processes conversations from an
        iterable without materialising the full list in memory. This is
        useful for large datasets or when conversations are produced by a
        live synthesis pipeline.

        Only formatters that support streaming (``jsonl``, ``parquet``,
        ``hf_messages`` via JSONL mode) benefit from this. Other formatters
        will buffer the entire iterable internally.

        When multiple formats are requested, the iterable is fanned out using
        :func:`itertools.tee`, so each formatter receives its own iterator.

        Args:
            conversations: Iterable of conversations (streaming source).
            formats: List of format names (defaults to config).
            dataset_filename: Base filename for datasets.

        Returns:
            Dictionary mapping format names to output file paths.
        """
        export_formats = cast("list[str]", self.config.formats if formats is None else formats)

        # Fan out the iterable for each format
        iterators = tee(conversations, len(export_formats))

        output_files: dict[str, Path] = {}

        for format_name, conv_iter in zip(export_formats, iterators, strict=False):
            try:
                formatter = self._get_formatter(format_name)
                # Determine file extension based on formatter
                if format_name == "parquet":
                    ext = ".parquet"
                elif format_name in ("jsonl", "hf_messages"):
                    ext = ".jsonl"
                else:
                    ext = ".json"
                filename = f"{dataset_filename}_{format_name}{ext}"
                output_path = formatter.format_stream(conv_iter, filename)
                output_files[format_name] = output_path
                logger.info(f"Streamed export to {format_name}: {output_path}")
            except Exception as e:
                logger.error(f"Failed to stream export to {format_name}: {e}")
                raise ExportError(f"Streaming export to {format_name} failed: {e}") from e

        return output_files

    def validate_export(self, output_files: dict[str, Path]) -> dict[str, bool]:
        """Validate exported files."""
        results = {}

        for format_name, file_path in output_files.items():
            if format_name in {"unsloth_script", "dataset_card"} or "_" not in format_name:
                # Validate Python syntax or just check existence
                if file_path.suffix == ".py":
                    try:
                        with open(file_path) as f:
                            compile(f.read(), file_path, "exec")
                        results[format_name] = True
                    except SyntaxError:
                        results[format_name] = False
                else:
                    results[format_name] = file_path.exists()
            elif format_name in FORMATTER_MAP:
                try:
                    formatter = self._get_formatter(format_name)
                    data = formatter.load(file_path)  # type: ignore[attr-defined]
                    results[format_name] = formatter.validate(data)
                except Exception:
                    results[format_name] = False
            else:
                results[format_name] = file_path.exists()

        return results

    def get_export_stats(self, output_files: dict[str, Path]) -> dict[str, Any]:
        """Get statistics about exported files."""
        stats = {}

        for format_name, file_path in output_files.items():
            if file_path.exists():
                stats[format_name] = {
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                }

                # Count entries for JSON files
                if format_name in FORMATTER_MAP:
                    try:
                        import json

                        with open(file_path) as f:
                            data = json.load(f)
                        stats[format_name]["entries"] = len(data)
                    except Exception:
                        pass

        return stats
