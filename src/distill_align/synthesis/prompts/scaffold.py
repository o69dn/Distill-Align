"""
Scaffold Action prompt templates.

Extracts pure tool-calling or structural markdown output from conversational content.
"""

from typing import Dict, Any, List

from jinja2 import Template


# System prompt for the Scaffold Action
SCAFFOLD_SYSTEM_PROMPT = """You are an expert at extracting structured, actionable content from conversational text. Your task is to transform verbose, conversational responses into clean, structured formats suitable for fine-tuning.

Guidelines:
1. Remove conversational filler (e.g., "Sure!", "Of course!", "Here's...")
2. Extract the core technical content
3. Preserve code blocks, commands, and structured data
4. Maintain logical flow and organization
5. Output in clean Markdown or JSON format

Your output should be a JSON object with:
{
    "extracted_content": "The clean, structured content",
    "extraction_type": "markdown|code|tool_call|mixed",
    "confidence": 0.0-1.0
}"""


# Template for extracting structured content
SCAFFOLD_USER_TEMPLATE = Template("""Extract the structured, actionable content from the following text.

**Original Text:**
{{ content }}

**Context:**
{% if source_type %}**Source Type:** {{ source_type }}{% endif %}
{% if expected_format %}**Expected Format:** {{ expected_format }}{% endif %}

**Instructions:**
1. Remove all conversational filler and pleasantries
2. Extract the core technical content
3. Preserve code blocks, commands, and structured data exactly
4. Output in clean, organized format
5. Assign a confidence score (0.0-1.0) to your extraction

Output a JSON object with the extracted content.""")


# Template for tool-calling extraction
SCAFFOLD_TOOL_TEMPLATE = Template("""Extract tool calls and structured actions from the following text.

**Original Text:**
{{ content }}

**Instructions:**
1. Identify any tool calls, commands, or API calls mentioned
2. Extract the tool name, parameters, and expected output
3. Format as a structured tool call schema
4. Remove any surrounding explanation or commentary

Output a JSON object with the extracted tool calls.""")


# Template for code extraction
SCAFFOLD_CODE_TEMPLATE = Template("""Extract and clean code from the following text.

**Original Text:**
{{ content }}

{% if language %}**Language:** {{ language }}{% endif %}

**Instructions:**
1. Extract all code blocks
2. Remove explanatory text between code blocks
3. Preserve comments within code
4. Ensure code is syntactically valid
5. If multiple code blocks are related, combine them logically

Output a JSON object with the extracted code.""")


def render_scaffold_prompt(content: str, metadata: Dict[str, Any]) -> str:
    """
    Render a Scaffold Action prompt.

    Args:
        content: Content to extract from.
        metadata: Content metadata.

    Returns:
        Rendered prompt string.
    """
    extraction_type = metadata.get("extraction_type", "auto")

    if extraction_type == "tool_call":
        template = SCAFFOLD_TOOL_TEMPLATE
    elif extraction_type == "code":
        template = SCAFFOLD_CODE_TEMPLATE
    else:
        template = SCAFFOLD_USER_TEMPLATE

    return template.render(
        content=content,
        **metadata,
    )
