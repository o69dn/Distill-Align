"""
Structural pruner for synthesized content.

Applies heuristics to clean, filter, and validate synthesized conversations.
"""

import json
import re
from typing import Any

from loguru import logger

from ..core.schemas import ConversationSchema, SynthesizedTurn


class ContentPruner:
    """
    Prunes and cleans synthesized content.

    Features:
    - Remove conversational filler
    - Validate conversation structure
    - Filter low-quality outputs
    - Extract structured data from responses
    """

    # Common filler phrases to remove
    FILLER_PHRASES = [
        r"^(Sure|Of course|Certainly|Absolutely|Here you go|Here's)[\s!,.]+",
        r"^(I'd be happy to|I can help|Let me)[\s]+",
        r"^(Hope this helps|Let me know if)[\s]+.*$",
        r"^(Feel free to|Don't hesitate to)[\s]+.*$",
    ]

    def __init__(
        self,
        min_content_length: int = 50,
        max_filler_ratio: float = 0.3,
        require_json_validity: bool = True,
    ):
        """
        Initialize the pruner.

        Args:
            min_content_length: Minimum content length to keep.
            max_filler_ratio: Maximum ratio of filler to content.
            require_json_validity: Whether to validate JSON in responses.
        """
        self.min_content_length = min_content_length
        self.max_filler_ratio = max_filler_ratio
        self.require_json_validity = require_json_validity

    def prune_conversation(self, conversation: ConversationSchema) -> ConversationSchema | None:
        """
        Prune a single conversation.

        Args:
            conversation: Input conversation.

        Returns:
            Pruned conversation or None if quality is too low.
        """
        pruned_turns = []

        for turn in conversation.turns:
            pruned_content = self._clean_content(turn.content)

            # Skip empty turns
            if not pruned_content.strip():
                continue

            pruned_turns.append(
                SynthesizedTurn(
                    role=turn.role,
                    content=pruned_content,
                )
            )

        # Validate minimum turns
        if len(pruned_turns) < 2:
            logger.debug(f"Conversation {conversation.id} has too few turns after pruning")
            return None

        # Validate conversation structure
        if not self._validate_structure(pruned_turns):
            logger.debug(f"Conversation {conversation.id} has invalid structure")
            return None

        return ConversationSchema(
            id=conversation.id,
            source_chunk_id=conversation.source_chunk_id,
            turns=pruned_turns,
            reasoning_trace=conversation.reasoning_trace,
            confidence_score=conversation.confidence_score,
        )

    def prune_batch(self, conversations: list[ConversationSchema]) -> list[ConversationSchema]:
        """
        Prune a batch of conversations.

        Args:
            conversations: List of conversations.

        Returns:
            List of pruned conversations (may be shorter).
        """
        pruned = []
        for conv in conversations:
            result = self.prune_conversation(conv)
            if result:
                pruned.append(result)

        logger.info(f"Pruned {len(conversations)} -> {len(pruned)} conversations")
        return pruned

    def _clean_content(self, content: str) -> str:
        """
        Clean content by removing filler and fixing formatting.

        Args:
            content: Raw content.

        Returns:
            Cleaned content.
        """
        # Remove filler phrases
        for pattern in self.FILLER_PHRASES:
            content = re.sub(pattern, "", content, flags=re.MULTILINE | re.IGNORECASE)

        # Clean up whitespace
        content = content.strip()
        content = re.sub(r"\n{3,}", "\n\n", content)

        return content

    def _validate_structure(self, turns: list[SynthesizedTurn]) -> bool:
        """
        Validate conversation structure.

        Args:
            turns: List of conversation turns.

        Returns:
            True if structure is valid.
        """
        if not turns:
            return False

        # Check that conversation starts with system or user
        if turns[0].role not in ("system", "user"):
            return False

        # Check that we have at least one user and one assistant turn
        roles = {t.role for t in turns}
        if "user" not in roles or "assistant" not in roles:
            return False

        # Check for alternating user/assistant (allowing system at start)
        non_system_turns = [t for t in turns if t.role != "system"]
        return all(non_system_turns[i].role != non_system_turns[i + 1].role for i in range(len(non_system_turns) - 1))

    def extract_json_from_response(self, content: str) -> dict[str, Any] | None:
        """
        Extract JSON from LLM response content.

        Args:
            content: Response content.

        Returns:
            Parsed JSON or None.
        """
        # Try to find JSON in code blocks
        json_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                continue

        # Try to find JSON object directly
        json_pattern = r"\{[^{}]*\}"
        matches = re.findall(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                continue

        # Try parsing the entire content
        try:
            return json.loads(content)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return None

    def validate_conversation_quality(self, conversation: ConversationSchema) -> tuple[bool, float, list[str]]:
        """
        Validate conversation quality.

        Args:
            conversation: Conversation to validate.

        Returns:
            Tuple of (is_valid, quality_score, issues).
        """
        issues = []
        score = 1.0

        # Check turn count
        if len(conversation.turns) < 2:
            issues.append("Too few turns")
            score -= 0.5

        # Check content length
        total_content = sum(len(t.content) for t in conversation.turns)
        if total_content < self.min_content_length:
            issues.append("Content too short")
            score -= 0.3

        # Check for filler ratio
        filler_count = sum(
            1 for t in conversation.turns if any(re.match(p, t.content, re.IGNORECASE) for p in self.FILLER_PHRASES)
        )
        filler_ratio = filler_count / len(conversation.turns) if conversation.turns else 0
        if filler_ratio > self.max_filler_ratio:
            issues.append(f"Too much filler: {filler_ratio:.1%}")
            score -= 0.2

        # Check structure validity
        if not self._validate_structure(conversation.turns):
            issues.append("Invalid conversation structure")
            score -= 0.4

        is_valid = score >= 0.5
        return is_valid, max(0.0, min(1.0, score)), issues
