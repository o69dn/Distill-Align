"""
Dataset validation, deduplication, and statistics.

Validates exported datasets for quality, deduplication, and generates statistics.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from ..core.schemas import ConversationSchema, SynthesizedTurn


class ValidationIssue(BaseModel):
    """A single validation issue."""

    severity: str  # "error", "warning", "info"
    category: str  # "structure", "quality", "duplication", "content"
    message: str
    conversation_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class DatasetStats(BaseModel):
    """Statistics about a dataset."""

    total_conversations: int = 0
    total_turns: int = 0
    avg_turns_per_conversation: float = 0.0
    min_turns: int = 0
    max_turns: int = 0

    total_characters: int = 0
    avg_characters_per_conversation: float = 0.0
    avg_characters_per_turn: float = 0.0

    # Token estimates (rough: 1 token ≈ 4 chars)
    estimated_total_tokens: int = 0
    avg_tokens_per_conversation: float = 0.0

    # Turn role distribution
    system_turns: int = 0
    user_turns: int = 0
    assistant_turns: int = 0
    tool_turns: int = 0

    # Quality metrics
    conversations_with_system: int = 0
    conversations_with_multi_turn: int = 0
    avg_filler_ratio: float = 0.0

    # Dedup
    duplicate_count: int = 0
    unique_count: int = 0

    # Conversation length histogram
    length_histogram: Dict[str, int] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Full validation report."""

    is_valid: bool = True
    quality_score: float = 1.0  # 0.0 to 1.0
    stats: DatasetStats = Field(default_factory=DatasetStats)
    issues: List[ValidationIssue] = Field(default_factory=list)
    duplicate_ids: List[str] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Validation Report",
            f"{'=' * 50}",
            f"Valid: {self.is_valid}",
            f"Quality Score: {self.quality_score:.2f}",
            f"Errors: {self.error_count}, Warnings: {self.warning_count}",
            f"",
            f"Dataset Statistics:",
            f"  Conversations: {self.stats.total_conversations}",
            f"  Total Turns: {self.stats.total_turns}",
            f"  Avg Turns/Conv: {self.stats.avg_turns_per_conversation:.1f}",
            f"  Est. Total Tokens: {self.stats.estimated_total_tokens:,}",
            f"  Duplicates Found: {self.stats.duplicate_count}",
            f"  Conversations w/ System: {self.stats.conversations_with_system}",
            f"  Multi-turn Conversations: {self.stats.conversations_with_multi_turn}",
        ]

        if self.issues:
            lines.append(f"")
            lines.append(f"Issues:")
            for issue in self.issues[:20]:  # Show first 20
                lines.append(f"  [{issue.severity.upper()}] {issue.category}: {issue.message}")
            if len(self.issues) > 20:
                lines.append(f"  ... and {len(self.issues) - 20} more")

        return "\n".join(lines)


# Filler phrases for quality scoring
FILLER_PATTERNS = [
    re.compile(r"^(Sure|Of course|Certainly|Absolutely|Here you go|Here's)[\s!,.]+", re.I),
    re.compile(r"^(I'd be happy to|I can help|Let me)[\s]+", re.I),
    re.compile(r"^(Hope this helps|Let me know if)[\s]+.*$", re.I),
    re.compile(r"^(Feel free to|Don't hesitate to)[\s]+.*$", re.I),
    re.compile(r"^(Great question|Interesting question)[\s!,.]+", re.I),
]


class DatasetValidator:
    """
    Validates and analyzes exported datasets.

    Features:
    - Structure validation (correct format, required fields)
    - Quality scoring (filler ratio, turn balance, content length)
    - Deduplication (exact hash + fuzzy matching)
    - Statistics generation
    """

    def __init__(
        self,
        min_conversations: int = 1,
        min_turns_per_conversation: int = 2,
        min_content_length: int = 20,
        max_filler_ratio: float = 0.3,
    ):
        """
        Initialize the validator.

        Args:
            min_conversations: Minimum conversations required.
            min_turns_per_conversation: Minimum turns per conversation.
            min_content_length: Minimum content length per turn.
            max_filler_ratio: Maximum acceptable filler ratio.
        """
        self.min_conversations = min_conversations
        self.min_turns_per_conversation = min_turns_per_conversation
        self.min_content_length = min_content_length
        self.max_filler_ratio = max_filler_ratio

    def validate(self, conversations: List[ConversationSchema]) -> ValidationReport:
        """
        Run full validation on a dataset.

        Args:
            conversations: List of conversations to validate.

        Returns:
            ValidationReport with stats, issues, and quality score.
        """
        issues: List[ValidationIssue] = []
        quality_deductions = 0.0

        # Structure validation
        structure_issues = self._validate_structure(conversations)
        issues.extend(structure_issues)
        quality_deductions += len(structure_issues) * 0.05

        # Quality scoring
        quality_issues, filler_ratio = self._validate_quality(conversations)
        issues.extend(quality_issues)
        quality_deductions += len(quality_issues) * 0.03

        # Deduplication
        duplicate_ids, dup_count = self._find_duplicates(conversations)
        if dup_count > 0:
            issues.append(ValidationIssue(
                severity="warning",
                category="duplication",
                message=f"Found {dup_count} duplicate conversations",
                details={"duplicate_ids": duplicate_ids[:10]},
            ))
            quality_deductions += min(dup_count * 0.01, 0.2)

        # Statistics
        stats = self._compute_stats(conversations, dup_count, filler_ratio)

        # Final quality score
        quality_score = max(0.0, min(1.0, 1.0 - quality_deductions))

        # Determine validity
        is_valid = all(i.severity != "error" for i in issues)

        report = ValidationReport(
            is_valid=is_valid,
            quality_score=quality_score,
            stats=stats,
            issues=issues,
            duplicate_ids=duplicate_ids,
        )

        logger.info(f"Validation complete: {len(conversations)} conversations, "
                     f"score={quality_score:.2f}, errors={report.error_count}, warnings={report.warning_count}")

        return report

    def _validate_structure(self, conversations: List[ConversationSchema]) -> List[ValidationIssue]:
        """Validate conversation structure."""
        issues = []

        if len(conversations) < self.min_conversations:
            issues.append(ValidationIssue(
                severity="error",
                category="structure",
                message=f"Too few conversations: {len(conversations)} < {self.min_conversations}",
            ))

        for conv in conversations:
            # Check minimum turns
            if len(conv.turns) < self.min_turns_per_conversation:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="structure",
                    message=f"Conversation {conv.id} has only {len(conv.turns)} turns",
                    conversation_id=conv.id,
                ))

            # Check for empty turns
            for i, turn in enumerate(conv.turns):
                if not turn.content.strip():
                    issues.append(ValidationIssue(
                        severity="warning",
                        category="structure",
                        message=f"Conversation {conv.id} turn {i} is empty",
                        conversation_id=conv.id,
                    ))

            # Check role sequence
            roles = [t.role for t in conv.turns]
            if "user" not in roles:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="structure",
                    message=f"Conversation {conv.id} has no user turn",
                    conversation_id=conv.id,
                ))
            if "assistant" not in roles:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="structure",
                    message=f"Conversation {conv.id} has no assistant turn",
                    conversation_id=conv.id,
                ))

        return issues

    def _validate_quality(self, conversations: List[ConversationSchema]) -> Tuple[List[ValidationIssue], float]:
        """Validate content quality."""
        issues = []
        total_filler = 0
        total_turns = 0

        for conv in conversations:
            for turn in conv.turns:
                total_turns += 1

                # Check content length
                if len(turn.content) < self.min_content_length:
                    issues.append(ValidationIssue(
                        severity="info",
                        category="quality",
                        message=f"Conversation {conv.id} turn content is very short ({len(turn.content)} chars)",
                        conversation_id=conv.id,
                    ))

                # Check for filler
                for pattern in FILLER_PATTERNS:
                    if pattern.match(turn.content.strip()):
                        total_filler += 1
                        break

        filler_ratio = total_filler / total_turns if total_turns > 0 else 0.0

        if filler_ratio > self.max_filler_ratio:
            issues.append(ValidationIssue(
                severity="warning",
                category="quality",
                message=f"High filler ratio: {filler_ratio:.1%} (max: {self.max_filler_ratio:.1%})",
                details={"filler_ratio": filler_ratio},
            ))

        return issues, filler_ratio

    def _find_duplicates(self, conversations: List[ConversationSchema]) -> Tuple[List[str], int]:
        """Find duplicate conversations by content hash."""
        seen_hashes: Dict[str, str] = {}  # hash -> first conversation_id
        duplicate_ids: List[str] = []

        for conv in conversations:
            # Hash based on all turn content
            content = "|".join(t.content for t in conv.turns)
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            if content_hash in seen_hashes:
                duplicate_ids.append(conv.id)
            else:
                seen_hashes[content_hash] = conv.id

        return duplicate_ids, len(duplicate_ids)

    def _compute_stats(
        self,
        conversations: List[ConversationSchema],
        duplicate_count: int,
        filler_ratio: float,
    ) -> DatasetStats:
        """Compute dataset statistics."""
        if not conversations:
            return DatasetStats()

        turn_counts = [len(c.turns) for c in conversations]
        char_counts = [sum(len(t.content) for t in c.turns) for c in conversations]
        total_chars = sum(char_counts)
        total_turns = sum(turn_counts)

        # Role counts
        role_counter: Counter = Counter()
        for conv in conversations:
            for turn in conv.turns:
                role_counter[turn.role] += 1

        # Conversations with system prompt
        with_system = sum(1 for c in conversations if any(t.role == "system" for t in c.turns))
        multi_turn = sum(1 for c in conversations if len(c.turns) > 2)

        # Length histogram
        histogram: Dict[str, int] = {}
        for count in turn_counts:
            bucket = f"{(count // 5) * 5}-{(count // 5) * 5 + 4}"
            histogram[bucket] = histogram.get(bucket, 0) + 1

        return DatasetStats(
            total_conversations=len(conversations),
            total_turns=total_turns,
            avg_turns_per_conversation=total_turns / len(conversations),
            min_turns=min(turn_counts),
            max_turns=max(turn_counts),
            total_characters=total_chars,
            avg_characters_per_conversation=total_chars / len(conversations),
            avg_characters_per_turn=total_chars / total_turns if total_turns else 0,
            estimated_total_tokens=total_chars // 4,
            avg_tokens_per_conversation=(total_chars // 4) / len(conversations),
            system_turns=role_counter.get("system", 0),
            user_turns=role_counter.get("user", 0),
            assistant_turns=role_counter.get("assistant", 0),
            tool_turns=role_counter.get("tool", 0),
            conversations_with_system=with_system,
            conversations_with_multi_turn=multi_turn,
            avg_filler_ratio=filler_ratio,
            duplicate_count=duplicate_count,
            unique_count=len(conversations) - duplicate_count,
            length_histogram=histogram,
        )

    def deduplicate(self, conversations: List[ConversationSchema]) -> List[ConversationSchema]:
        """
        Remove duplicate conversations, keeping the first occurrence.

        Args:
            conversations: Input conversations.

        Returns:
            Deduplicated conversations.
        """
        seen_hashes: Set[str] = set()
        unique: List[ConversationSchema] = []

        for conv in conversations:
            content = "|".join(t.content for t in conv.turns)
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique.append(conv)

        removed = len(conversations) - len(unique)
        if removed > 0:
            logger.info(f"Deduplicated: removed {removed} duplicates, {len(unique)} unique remaining")

        return unique
