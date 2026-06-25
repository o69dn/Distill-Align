"""
PyPI update checker for Distill-Align.

Checks whether a newer version is available on PyPI and notifies the user.
Respects DISTILL_NO_UPDATE_CHECK and CI environment variables to opt out.
Silently handles all network errors — never blocks or crashes the application.
"""

import os

import httpx
from loguru import logger

from .. import __version__

PYPI_URL = "https://pypi.org/pypi/distill-align/json"
REQUEST_TIMEOUT = 3.0  # seconds


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a semver string into a comparison-safe tuple."""
    return tuple(int(part) for part in v.split("."))


def check_pypi_version() -> str | None:
    """Check PyPI for a newer version of Distill-Align.

    Returns:
        The latest version string if newer than the installed version,
        otherwise ``None``.

    The check is skipped entirely when the ``DISTILL_NO_UPDATE_CHECK`` or
    ``CI`` environment variable is set (to any value).  All network and
    parsing errors are silently swallowed — a failure is treated as
    "no update available".
    """
    if os.environ.get("DISTILL_NO_UPDATE_CHECK") or os.environ.get("CI"):
        return None

    try:
        response = httpx.get(PYPI_URL, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.debug("Update check failed (network or parse error)")
        return None

    try:
        latest = data["info"]["version"]
    except (KeyError, TypeError):
        logger.debug("Update check failed: unexpected PyPI response format")
        return None

    installed = _parse_version(__version__)
    available = _parse_version(latest)

    if available > installed:
        logger.info("Update available: v{} → v{}", __version__, latest)
        return latest

    return None
