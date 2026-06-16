"""
vLLM client.

Supports vLLM server for high-performance local inference.
"""

from typing import Any

from ...core.exceptions import LLMClientError
from .openai import OpenAIClient


class VLLMClient(OpenAIClient):
    """
    Client for vLLM server.

    vLLM exposes an OpenAI-compatible API, so we extend OpenAIClient
    with vLLM-specific configurations.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize the vLLM client.

        Args:
            base_url: Base URL for the vLLM server.
            model: Model name.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        super().__init__(
            base_url=base_url,
            api_key=None,  # vLLM typically doesn't require API key
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Model information dictionary.
        """
        client = await self._get_client()
        try:
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
            if data.get("data"):
                return data["data"][0]  # type: ignore[no-any-return]
            return {}
        except Exception as e:
            raise LLMClientError(f"Failed to get model info: {e}") from e
