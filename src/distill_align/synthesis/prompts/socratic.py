"""
Socratic Transformer prompt templates.

Converts raw content into guided, conversational multi-turn Q&A schemas.
Supports loading from external .j2 template files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template


# Default templates directory (alongside this file)
_TEMPLATES_DIR = Path(__file__).parent / "socratic"


def _load_template(name: str, custom_dir: Optional[Path] = None) -> Template:
    """
    Load a Jinja2 template from file.

    Args:
        name: Template filename (e.g., "system.j2", "code.j2").
        custom_dir: Optional custom directory to search first.

    Returns:
        Jinja2 Template object.
    """
    # Try custom directory first
    if custom_dir:
        custom_path = custom_dir / name
        if custom_path.exists():
            return Template(custom_path.read_text(encoding="utf-8"))

    # Fall back to default directory
    default_path = _TEMPLATES_DIR / name
    if default_path.exists():
        return Template(default_path.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"Template not found: {name}")


def get_system_prompt(custom_dir: Optional[Path] = None) -> str:
    """
    Get the Socratic system prompt.

    Args:
        custom_dir: Optional custom templates directory.

    Returns:
        System prompt string.
    """
    template = _load_template("system.j2", custom_dir)
    return template.render()


def render_socratic_prompt(
    content: str,
    metadata: Dict[str, Any],
    custom_dir: Optional[str | Path] = None,
) -> str:
    """
    Render a Socratic Transformer prompt.

    Args:
        content: Raw content to transform.
        metadata: Content metadata (source_type, title, language, etc.).
        custom_dir: Optional custom templates directory.

    Returns:
        Rendered prompt string.
    """
    source_type = metadata.get("source_type", "text")
    custom_path = Path(custom_dir) if custom_dir else None

    # Select template based on source type
    if source_type == "code":
        template = _load_template("code.j2", custom_path)
    else:
        template = _load_template("markdown.j2", custom_path)

    return template.render(
        content=content,
        **metadata,
    )


# Backward compatibility: keep the old constants for direct import
SOCRATIC_SYSTEM_PROMPT = get_system_prompt()
