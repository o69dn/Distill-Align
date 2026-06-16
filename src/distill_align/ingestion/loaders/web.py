"""
Web page loader for fetching content from URLs.

Uses httpx to download web pages and beautifulsoup4 to extract content.
"""

from typing import Any

from ...core.exceptions import LoaderError
from ...core.schemas import DataChunk, SourceMetadata


class WebLoader:
    """Loader for web pages/URLs.

    Unlike file-based loaders, this loader fetches content from URLs over HTTP.
    It does not extend BaseLoader due to its fundamentally different interface (URL-based, async).
    """

    SUPPORTED_EXTENSIONS: list[str] = []  # Not file-based

    async def load_url(self, url: str, **kwargs: Any) -> list[DataChunk]:
        """Load content from a URL.

        Args:
            url: URL to fetch.
            **kwargs:
                selector: CSS selector to extract specific content.
                max_length: Maximum content length.

        Returns:
            List of DataChunks.
        """
        import httpx
        from bs4 import BeautifulSoup

        selector = kwargs.get("selector")
        max_length = kwargs.get("max_length", 100_000)

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Distill-Align/1.0 WebLoader",
                    },
                )
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            if selector:
                elements = soup.select(selector)
                content = "\n\n".join(el.get_text(strip=True) for el in elements)
            else:
                # Get main content or body
                main = soup.find("main") or soup.find("article") or soup.find("body")
                content = (
                    main.get_text(strip=True, separator="\n") if main else soup.get_text(strip=True, separator="\n")
                )

            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + "\n\n[Content truncated...]"

            title = soup.title.string.strip() if soup.title and soup.title.string else url

            metadata = SourceMetadata(
                source_type="text",
                file_path=url,
                file_name=url.split("/")[-1] or "webpage",
                title=title,
                custom_tags={"url": url, "selector": selector},
            )

            return [DataChunk(content=content, metadata=metadata)]

        except Exception as e:
            raise LoaderError(f"Failed to load web page {url}: {e}") from e

    def load_sync(self, url: str, **kwargs: Any) -> list[DataChunk]:
        """Synchronous wrapper for load_url.

        Args:
            url: URL to fetch.
            **kwargs: Passed to load_url().

        Returns:
            List of DataChunks.
        """
        import asyncio

        return asyncio.run(self.load_url(url, **kwargs))

    def get_supported_extensions(self) -> list[str]:
        """Return empty list since web loader doesn't use file extensions."""
        return []
