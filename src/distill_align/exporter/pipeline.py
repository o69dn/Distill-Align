"""
Export pipeline orchestrator.

Handles the full export workflow: formatting conversations and generating Unsloth configs.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

from ..core.schemas import ConversationSchema, ExportConfig
from ..core.exceptions import ExportError
from .formatters.base import BaseFormatter
from .formatters.sharegpt import ShareGPTFormatter
from .formatters.alpaca import AlpacaFormatter
from .unsloth_builder import UnslothConfigBuilder


# Map format names to formatter classes
FORMATTER_MAP: Dict[str, type[BaseFormatter]] = {
    "sharegpt": ShareGPTFormatter,
    "alpaca": AlpacaFormatter,
}


class ExportPipeline:
    """Orchestrates the export of conversations to various formats."""

    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize the export pipeline.

        Args:
            config: Optional export configuration. Uses defaults if not provided.
        """
        self.config = config or ExportConfig()
        self._formatters: Dict[str, BaseFormatter] = {}
        self._unsloth_builder: Optional[UnslothConfigBuilder] = None

    def _get_formatter(self, format_name: str) -> BaseFormatter:
        """
        Get or create a formatter for the specified format.

        Args:
            format_name: Name of the format (e.g., "sharegpt", "alpaca").

        Returns:
            Formatter instance.

        Raises:
            ExportError: If format is not supported.
        """
        if format_name not in FORMATTER_MAP:
            raise ExportError(f"Unsupported format: {format_name}")

        if format_name not in self._formatters:
            self._formatters[format_name] = FORMATTER_MAP[format_name](self.config.output_dir)

        return self._formatters[format_name]

    def _get_unsloth_builder(self) -> UnslothConfigBuilder:
        """
        Get or create the Unsloth config builder.

        Returns:
            UnslothConfigBuilder instance.
        """
        if self._unsloth_builder is None:
            self._unsloth_builder = UnslothConfigBuilder(self.config)
        return self._unsloth_builder

    def export(
        self,
        conversations: List[ConversationSchema],
        formats: Optional[List[str]] = None,
        dataset_filename: str = "dataset",
        generate_unsloth: bool = True,
        **unsloth_kwargs,
    ) -> Dict[str, Path]:
        """
        Export conversations to specified formats.

        Args:
            conversations: List of conversations to export.
            formats: List of format names (defaults to config).
            dataset_filename: Base filename for datasets.
            generate_unsloth: Whether to generate Unsloth training script.
            **unsloth_kwargs: Additional kwargs for Unsloth config.

        Returns:
            Dictionary mapping format names to output file paths.
        """
        if formats is None:
            formats = self.config.export_formats

        output_files = {}

        # Export to each format
        for format_name in formats:
            try:
                formatter = self._get_formatter(format_name)
                filename = f"{dataset_filename}_{format_name}.json"
                output_path = formatter.format(conversations, filename)
                output_files[format_name] = output_path
                logger.info(f"Exported to {format_name}: {output_path}")
            except Exception as e:
                logger.error(f"Failed to export to {format_name}: {e}")
                raise ExportError(f"Export to {format_name} failed: {e}")

        # Generate Unsloth script
        if generate_unsloth and self.config.generate_unsloth_script:
            try:
                builder = self._get_unsloth_builder()
                # Use the first format's output as the dataset path
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

        return output_files

    def validate_export(self, output_files: Dict[str, Path]) -> Dict[str, bool]:
        """
        Validate exported files.

        Args:
            output_files: Dictionary of format names to file paths.

        Returns:
            Dictionary of format names to validation results.
        """
        results = {}

        for format_name, file_path in output_files.items():
            if format_name == "unsloth_script":
                # Validate Python syntax
                try:
                    with open(file_path, "r") as f:
                        compile(f.read(), file_path, "exec")
                    results[format_name] = True
                except SyntaxError:
                    results[format_name] = False
            elif format_name in FORMATTER_MAP:
                try:
                    formatter = self._get_formatter(format_name)
                    data = formatter.load(file_path)
                    results[format_name] = formatter.validate(data)
                except Exception:
                    results[format_name] = False
            else:
                results[format_name] = file_path.exists()

        return results

    def get_export_stats(self, output_files: Dict[str, Path]) -> Dict[str, Any]:
        """
        Get statistics about exported files.

        Args:
            output_files: Dictionary of format names to file paths.

        Returns:
            Dictionary of statistics.
        """
        stats = {}

        for format_name, file_path in output_files.items():
            if file_path.exists():
                stats[format_name] = {
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                }

                # Count entries for JSON files
                if format_name in FORMATTER_MAP and format_name != "unsloth_script":
                    try:
                        import json
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        stats[format_name]["entries"] = len(data)
                    except Exception:
                        pass

        return stats
