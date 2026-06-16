"""
Socratic Transformer prompt templates.

Converts raw content into guided, conversational multi-turn Q&A schemas.
"""

from typing import Dict, Any, List

from jinja2 import Template


# System prompt for the Socratic Transformer
SOCRATIC_SYSTEM_PROMPT = """You are an expert educator and Socratic questioner. Your task is to transform raw technical content into a guided, multi-turn conversation that teaches the material through questions and answers.

Guidelines:
1. Break down complex concepts into digestible pieces
2. Ask probing questions that lead to deeper understanding
3. Provide clear, concise answers that build on previous context
4. Include reasoning traces showing your thought process
5. Use examples and analogies when helpful
6. Maintain a natural, conversational flow

Your output should be a JSON object with the following structure:
{
    "conversation": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "reasoning_trace": "Your detailed reasoning about how to structure this conversation..."
}"""


# Template for generating Socratic conversations
SOCRATIC_USER_TEMPLATE = Template("""Transform the following content into a Socratic multi-turn conversation.

**Source Content:**
{{ content }}

**Source Type:** {{ source_type }}
{% if title %}**Title:** {{ title }}{% endif %}
{% if language %}**Language:** {{ language }}{% endif %}
{% if section_headers %}**Sections:** {{ section_headers | join(' > ') }}{% endif %}

**Instructions:**
1. Create a system prompt that sets up the teaching context
2. Generate 3-5 user questions that progressively explore the topic
3. Provide detailed assistant answers that teach the concepts
4. Include your reasoning trace explaining your pedagogical choices

Output a JSON object with the conversation and reasoning trace.""")


# Template for code-specific Socratic conversations
SOCRATIC_CODE_TEMPLATE = Template("""Transform the following code into a Socratic teaching conversation.

**Source Code ({{ language }}):**
```{{ language }}
{{ content }}
```

**Module:** {{ module_path }}
{% if functions %}**Functions:** {{ functions | join(', ') }}{% endif %}
{% if classes %}**Classes:** {{ classes | join(', ') }}{% endif %}

**Instructions:**
1. Create a system prompt for teaching this code
2. Ask questions about the code's purpose, structure, and implementation
3. Explain the code line-by-line when answering
4. Discuss design patterns and best practices
5. Include your reasoning about how to teach this code effectively

Output a JSON object with the conversation and reasoning trace.""")


def render_socratic_prompt(content: str, metadata: Dict[str, Any]) -> str:
    """
    Render a Socratic Transformer prompt.

    Args:
        content: Raw content to transform.
        metadata: Content metadata (source_type, title, language, etc.).

    Returns:
        Rendered prompt string.
    """
    source_type = metadata.get("source_type", "text")

    if source_type == "code":
        template = SOCRATIC_CODE_TEMPLATE
    else:
        template = SOCRATIC_USER_TEMPLATE

    return template.render(
        content=content,
        **metadata,
    )
