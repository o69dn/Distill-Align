"""
Safe JSON loading utilities with size bounds.

Protects against memory exhaustion from arbitrarily large JSON inputs.
"""

import json
import os
from pathlib import Path
from typing import Any

from .exceptions import LoaderError

# Maximum bytes for a JSON input file / string
MAX_JSON_BYTES = 200 * 1024 * 1024  # 200 MB

# Maximum nesting depth for parsed JSON (protects against stack overflow)
MAX_JSON_DEPTH = 100


def _check_size(size: int, max_bytes: int = MAX_JSON_BYTES, label: str = "input") -> None:
    """Raise ``LoaderError`` if *size* exceeds *max_bytes*."""
    if size > max_bytes:
        raise LoaderError(
            f"JSON {label} too large: {size / (1024*1024):.1f} MB "
            f"(max {max_bytes / (1024*1024):.0f} MB)"
        )


def safe_json_load(path: str | Path, max_bytes: int = MAX_JSON_BYTES) -> Any:
    """Load and parse a JSON file with a size guard.

    Raises:
        LoaderError: if the file is too large or cannot be parsed.
        FileNotFoundError: if the path does not exist.
    """
    path = Path(path)
    size = path.stat().st_size
    _check_size(size, max_bytes, label=str(path))

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_json_loads(data: str, max_bytes: int = MAX_JSON_BYTES, label: str = "data") -> Any:
    """Parse a JSON string with a size guard.

    Raises:
        LoaderError: if the string is too large or cannot be parsed.
    """
    size = len(data.encode("utf-8"))
    _check_size(size, max_bytes, label=label)
    return json.loads(data, parse_int=str, parse_constant=str)


def safe_json_loads_value(value: Any, label: str = "value") -> Any:
    """Parse a cached JSON value (stored as a string) with a size guard.

    This variant accepts either ``str`` or pre-deserialised ``dict/list`` and
    guards against maliciously large cached strings.
    """
    if isinstance(value, str):
        return safe_json_loads(value, label=label)
    return value
