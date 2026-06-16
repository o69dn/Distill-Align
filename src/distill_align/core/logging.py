"""
Logging configuration using Loguru.

Provides structured logging with async safety and configurable outputs.
Supports both text (default) and JSON log formats.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger


def _json_format(record: Any) -> str:
    """Serialize a log record to JSON format.

    Args:
        record: Loguru log record (dict-like).

    Returns:
        JSON string representation of the log record.
    """
    return json.dumps(
        {
            "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),  # type: ignore[union-attr]
            "level": record["level"].name,  # type: ignore[union-attr]
            "module": record["name"],  # type: ignore[union-attr]
            "function": record["function"],  # type: ignore[union-attr]
            "line": record["line"],  # type: ignore[union-attr]
            "message": record["message"],  # type: ignore[union-attr]
            "extra": record["extra"],  # type: ignore[union-attr]
        }
    )


def _json_sink_factory(stream: Any) -> Callable[..., None]:
    """Create a Loguru-compatible sink function that outputs JSON.

    Args:
        stream: Output stream (typically sys.stderr or a file path).

    Returns:
        A callable sink compatible with logger.add().
    """

    def sink(message: Any) -> None:
        """Process a Loguru message as JSON.

        Args:
            message: Loguru message object.
        """
        text = _json_format(message.record) + "\n"  # type: ignore[union-attr]
        stream.write(text)

    return sink


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    log_format: str = "text",
) -> None:
    """
    Configure Loguru logging.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output.
        rotation: Log file rotation size/time.
        retention: How long to keep old log files.
        log_format: Output format: "text" (default) or "json".
    """
    # Remove default handler
    logger.remove()

    if log_format == "json":
        logger.add(
            _json_sink_factory(sys.stderr),
            level=log_level.upper(),
            colorize=False,
        )
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                str(log_path),
                level=log_level.upper(),
                format=_json_format,  # type: ignore[arg-type]
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
            )
    else:
        logger.add(
            sys.stderr,
            level=log_level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>",
            colorize=True,
        )
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                str(log_path),
                level=log_level.upper(),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
            )

    logger.info(f"Logging initialized at {log_level.upper()} level ({log_format} format)")


def get_logger(name: str):
    """
    Get a named logger instance.

    Args:
        name: Logger name (typically module name).

    Returns:
        Logger instance with bound name.
    """
    return logger.bind(name=name)
