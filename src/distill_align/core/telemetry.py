"""
Anonymous telemetry for usage statistics.

Telemetry is opt-in and disabled by default. Enable with DISTILL_TELEMETRY=true.
No personal data is collected — only anonymized usage counts.

NOTE: The telemetry endpoint is not yet operational. Events are silently
discarded until the backend service is deployed.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

TELEMETRY_ENV_VAR = "DISTILL_TELEMETRY"
TELEMETRY_URL = ""  # Not yet operational — see note above


def is_enabled() -> bool:
    """Check if telemetry is enabled."""
    return False  # Telemetry backend not yet deployed


def track_event(event: str, properties: dict[str, Any] | None = None) -> None:
    """Track an anonymous event (currently a no-op)."""
    if not is_enabled():
        return
    # Intentionally disabled until backend is deployed
    logger.debug("Telemetry event (disabled): %s", event)


class TelemetryContext:
    """Context manager for telemetry tracking."""

    def __init__(self, event: str, properties: dict[str, Any] | None = None):
        self.event = event
        self.properties = properties or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        track_event(self.event, self.properties)
