"""
Scaffold Action prompt templates.

Extracts pure tool-calling or structural markdown output from conversational content.
Supports loading from external .j2 template files.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Template


# Default templates directory
_TEMPLATES_DIR = Path(__file__).parent / "scaffold"


def _load_template(name: str, custom_dir: Optional[Path] = None) -> Template:
    """
    Load a Jinja2 template from file.

    Args:
        name: Template filename.
        custom_dir: Optional custom directory to search first.

    Returns:
        Jinja2 Template object.
    """
    if custom_dir:
        custom_path = custom_dir / name
        if custom_path.exists():
            return Template(custom_path.read_text(encoding="utf-8"))

    default_path = _TEMPLATES_DIR / name
    if default_path.exists():
        return Template(default_path.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"Template not found: {name}")


def get_system_prompt(custom_dir: Optional[Path] = None) -> str:
    """
    Get the Scaffold system prompt.

    Args:
        custom_dir: Optional custom templates directory.

    Returns:
        System prompt string.
    """
    template = _load_template("system.j2", custom_dir)
    return template.render()


def render_scaffold_prompt(
    content: str,
    metadata: Dict[str, Any],
    custom_dir: Optional[str | Path] = None,
) -> str:
    """
    Render a Scaffold Action prompt.

    Args:
        content: Content to extract from.
        metadata: Content metadata.
        custom_dir: Optional custom templates directory.

    Returns:
        Rendered prompt string.
    """
    extraction_type = metadata.get("extraction_type", "auto")
    custom_path = Path(custom_dir) if custom_dir else None

    if extraction_type == "tool_call":
        template = _load_template("tool_call.j2", custom_path)
    elif extraction_type == "code":
        template = _load_template("code_extract.j2", custom_path)
    else:
        template = _load_template("system.j2", custom_path)

    return template.render(
        content=content,
        **metadata,
    )


# Backward compatibility
SCAFFOLD_SYSTEM_PROMPT = get_system_prompt()
