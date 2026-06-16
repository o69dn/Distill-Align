"""
Anonymous telemetry for usage statistics.

Telemetry is opt-in and disabled by default. Enable with DISTILL_TELEMETRY=true.
No personal data is collected — only anonymized usage counts.
"""

import os
import platform
import uuid
from pathlib import Path
from typing import Any

import httpx

from .. import __version__

TELEMETRY_ENV_VAR = "DISTILL_TELEMETRY"
TELEMETRY_URL = "https://telemetry.distill-align.dev/collect"  # Placeholder


def _get_machine_id() -> str:
    """Get a persistent anonymous machine ID."""
    id_file = Path.home() / ".distill-align" / "telemetry_id"
    if id_file.exists():
        return id_file.read_text().strip()
    id_file.parent.mkdir(parents=True, exist_ok=True)
    machine_id = uuid.uuid4().hex[:16]
    id_file.write_text(machine_id)
    return machine_id


def is_enabled() -> bool:
    """Check if telemetry is enabled."""
    return os.getenv(TELEMETRY_ENV_VAR, "").lower() in ("true", "1", "yes")


def track_event(event: str, properties: dict[str, Any] | None = None) -> None:
    """Track an anonymous event (fire-and-forget)."""
    if not is_enabled():
        return
    data = {
        "event": event,
        "properties": properties or {},
        "machine_id": _get_machine_id(),
        "version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.system(),
    }
    try:
        with httpx.Client() as client:
            client.post(TELEMETRY_URL, json=data, timeout=2)
    except Exception:
        pass  # Silently fail — telemetry should never block


class TelemetryContext:
    """Context manager for telemetry tracking."""

    def __init__(self, event: str, properties: dict[str, Any] | None = None):
        self.event = event
        self.properties = properties or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        track_event(self.event, self.properties)
