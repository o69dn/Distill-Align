"""
Token counting and cost estimation for LLM API calls.

Provides accurate token counting using tiktoken and cost estimation
based on model pricing.
"""

from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel


# Pricing per 1M tokens (USD, as of 2024)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Claude models (Anthropic)
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # Open-source (Ollama/vLLM) - typically free, but list for accounting
    "llama3.1": {"input": 0.0, "output": 0.0},
    "llama3.2": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "qwen2.5": {"input": 0.0, "output": 0.0},
}


# Model to tiktoken encoding mapping
MODEL_ENCODING_MAP: Dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o3-mini": "o200k_base",
}


class TokenCount(BaseModel):
    """Token count result for a single text."""

    text: str
    token_count: int
    model: str = ""
    estimated_cost: float = 0.0


class CostEstimate(BaseModel):
    """Cost estimate for a batch of API calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""
    num_requests: int = 0
    avg_tokens_per_request: float = 0.0


class Tokenizer:
    """
    Token counter and cost estimator.

    Uses tiktoken for accurate OpenAI token counting, with fallback
    to character-based estimation for other models.
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the tokenizer.

        Args:
            model: Model name for token counting and cost estimation.
        """
        self.model = model
        self._encoder = None

        # Cumulative stats
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._num_requests = 0

    def _get_encoder(self):
        """Get or create the tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken

                encoding_name = MODEL_ENCODING_MAP.get(self.model)
                if encoding_name:
                    self._encoder = tiktoken.get_encoding(encoding_name)
                else:
                    try:
                        self._encoder = tiktoken.encoding_for_model(self.model)
                    except KeyError:
                        self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken not installed, using character-based estimation")
                return None
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        # Fallback: ~4 characters per token (English)
        return max(1, len(text) // 4)

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of OpenAI-format messages.

        Accounts for per-message overhead (role, formatting).

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Total token count including message overhead.
        """
        encoder = self._get_encoder()
        if not encoder:
            # Fallback estimation
            total = 0
            for msg in messages:
                total += self.count_tokens(msg.get("content", ""))
                total += 4  # Per-message overhead
            return total

        # Tokens per message
        tokens_per_message = 3
        if "gpt-3.5" in self.model:
            tokens_per_message = 4

        total_tokens = 0
        for message in messages:
            total_tokens += tokens_per_message
            for key, value in message.items():
                total_tokens += len(encoder.encode(str(value)))
            total_tokens += 1  # Role marker

        total_tokens += 3  # Reply primer
        return total_tokens

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost in USD for given token counts.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        pricing = MODEL_PRICING.get(self.model, {"input": 0.0, "output": 0.0})

        # Pricing is per 1M tokens
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def record_usage(self, input_tokens: int, output_tokens: int = 0) -> None:
        """
        Record token usage for batch statistics.

        Args:
            input_tokens: Input token count.
            output_tokens: Output token count.
        """
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._num_requests += 1

    def get_stats(self) -> CostEstimate:
        """
        Get cumulative usage statistics.

        Returns:
            CostEstimate with totals and cost.
        """
        total_tokens = self._total_input_tokens + self._total_output_tokens
        cost = self.estimate_cost(self._total_input_tokens, self._total_output_tokens)
        avg = total_tokens / self._num_requests if self._num_requests > 0 else 0

        return CostEstimate(
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            model=self.model,
            num_requests=self._num_requests,
            avg_tokens_per_request=avg,
        )

    def estimate_batch_cost(
        self,
        texts: List[str],
        estimated_output_tokens: int = 500,
    ) -> CostEstimate:
        """
        Estimate the cost of processing a batch of texts.

        Args:
            texts: List of input texts.
            estimated_output_tokens: Estimated output tokens per request.

        Returns:
            CostEstimate for the batch.
        """
        total_input = sum(self.count_tokens(t) for t in texts)
        total_output = estimated_output_tokens * len(texts)
        cost = self.estimate_cost(total_input, total_output)

        return CostEstimate(
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            estimated_cost_usd=cost,
            model=self.model,
            num_requests=len(texts),
            avg_tokens_per_request=(total_input + total_output) / len(texts) if texts else 0,
        )

    def reset_stats(self) -> None:
        """Reset cumulative statistics."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._num_requests = 0
