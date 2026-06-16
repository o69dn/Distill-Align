"""
LLM-as-judge for evaluating conversation quality.

Uses a separate LLM call to score generated conversations on multiple criteria.
"""

import json
import re
from typing import Any, cast

from loguru import logger

from ..core.schemas import ConversationSchema
from .models.base import BaseLLMClient

JUDGE_PROMPT = """You are an expert evaluator of LLM training data. Evaluate the following conversation on these criteria:

1. **Relevance** (0-10): How well does the conversation relate to the source content?
2. **Coherence** (0-10): Is the conversation logical and well-structured?
3. **Correctness** (0-10): Is the information factually accurate?
4. **Completeness** (0-10): Does the conversation cover the key points?
5. **Safety** (0-10): Is the content safe and appropriate?

Source content:
{source_content}

Conversation:
{conversation_text}

Respond with a JSON object only:
{{"relevance": <score>, "coherence": <score>, "correctness": <score>, "completeness": <score>, "safety": <score>, "overall": <average>, "explanation": "<brief explanation>"}}
"""


class ConversationJudge:
    """LLM-as-judge for evaluating conversation quality."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize the judge with an LLM client.

        Args:
            llm_client: LLM client used for evaluation.
        """
        self.llm_client = llm_client

    async def evaluate(
        self,
        conversation: ConversationSchema,
        source_content: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single conversation.

        Args:
            conversation: The conversation to evaluate.
            source_content: Optional source content for context.

        Returns:
            Dictionary with evaluation scores.
        """
        conversation_text = json.dumps(
            [{"role": t.role, "content": t.content} for t in conversation.turns],
            indent=2,
        )

        prompt = JUDGE_PROMPT.format(
            source_content=source_content or "No source content provided",
            conversation_text=conversation_text,
        )

        try:
            result_text = await self.llm_client.generate(
                system_prompt="You are a quality evaluator.",
                user_prompt=prompt,
            )

            # Parse JSON from response
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                raw = json.loads(json_match.group())
                scores: dict[str, Any] = cast("dict[str, Any]", raw)
            else:
                scores = {"error": "Could not parse judge response", "raw": result_text}

            return scores
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {"error": str(e)}

    async def evaluate_batch(
        self,
        conversations: list[ConversationSchema],
        source_contents: list[str | None] | None = None,
        max_concurrency: int = 5,
    ) -> list[dict[str, Any]]:
        """Evaluate a batch of conversations.

        Args:
            conversations: List of conversations to evaluate.
            source_contents: Optional source contents for each conversation.
            max_concurrency: Maximum concurrent evaluations.

        Returns:
            List of evaluation result dictionaries.
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrency)

        async def evaluate_one(
            conv: ConversationSchema,
            source: str | None,
        ) -> dict[str, Any]:
            async with semaphore:
                return await self.evaluate(conv, source)

        if source_contents is None:
            source_contents = [None] * len(conversations)

        tasks = [evaluate_one(conv, src) for conv, src in zip(conversations, source_contents, strict=False)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error dicts
        processed: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, BaseException):
                processed.append({"error": str(result)})
            else:
                processed.append(result)
        return processed
