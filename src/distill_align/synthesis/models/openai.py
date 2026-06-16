"""
OpenAI-compatible LLM client.

Supports OpenAI API and compatible endpoints (vLLM, Ollama with OpenAI compatibility mode).
"""

from typing import List, Dict, Any, Optional

import httpx
from loguru import logger

from .base import BaseLLMClient, LLMMessage, LLMResponse
from ...core.exceptions import LLMClientError, RateLimitError, ModelNotFoundError


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize the OpenAI client.

        Args:
            base_url: Base URL for the OpenAI API.
            api_key: OpenAI API key.
            model: Model name (e.g., "gpt-4o", "gpt-4-turbo").
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        super().__init__(base_url, api_key, model, timeout, max_retries)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat completion request to OpenAI.

        Args:
            messages: List of conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters (e.g., top_p, frequency_penalty).

        Returns:
            LLMResponse object.

        Raises:
            LLMClientError: If the request fails.
        """
        client = await self._get_client()

        # Build request payload
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            # Parse response
            choice = data["choices"][0]
            usage = data.get("usage", {})

            return LLMResponse(
                content=choice["message"]["content"],
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {self.model}")
            else:
                raise LLMClientError(f"API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise LLMClientError(f"Request failed: {e}")

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a text completion request to OpenAI.

        Args:
            prompt: Text prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse object.

        Raises:
            LLMClientError: If the request fails.
        """
        client = await self._get_client()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        try:
            response = await client.post("/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            usage = data.get("usage", {})

            return LLMResponse(
                content=choice["text"],
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {self.model}")
            else:
                raise LLMClientError(f"API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise LLMClientError(f"Request failed: {e}")
