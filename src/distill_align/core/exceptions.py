"""
Custom exceptions for the Distill-Align framework.
"""


class DistillAlignError(Exception):
    """Base exception for all Distill-Align errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Ingestion Errors
# =============================================================================


class IngestionError(DistillAlignError):
    """Base exception for ingestion-related errors."""

    pass


class LoaderError(IngestionError):
    """Error during file loading."""

    pass


class ChunkerError(IngestionError):
    """Error during chunking."""

    pass


class UnsupportedFormatError(IngestionError):
    """Unsupported file format."""

    pass


# =============================================================================
# Synthesis Errors
# =============================================================================


class SynthesisError(DistillAlignError):
    """Base exception for synthesis-related errors."""

    pass


class LLMClientError(SynthesisError):
    """Error communicating with LLM backend."""

    pass


class RateLimitError(LLMClientError):
    """Rate limit exceeded."""

    pass


class ModelNotFoundError(LLMClientError):
    """Requested model not found."""

    pass


class PromptError(SynthesisError):
    """Error in prompt template rendering."""

    pass


# =============================================================================
# Export Errors
# =============================================================================


class ExportError(DistillAlignError):
    """Base exception for export-related errors."""

    pass


class FormatError(ExportError):
    """Error formatting dataset."""

    pass


class UnslothConfigError(ExportError):
    """Error generating Unsloth configuration."""

    pass


# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(DistillAlignError):
    """Base exception for pipeline orchestration errors."""

    pass


class PipelineStageError(PipelineError):
    """Error in a specific pipeline stage."""

    pass


class PipelineTimeoutError(PipelineError):
    """Pipeline execution timeout."""

    pass
