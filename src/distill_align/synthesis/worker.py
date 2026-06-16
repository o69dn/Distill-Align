"""
Async batch worker pool for LLM synthesis.

Features:
- Semaphore-based concurrency control
- Rate limiting (requests per minute)
- Exponential backoff retries with tenacity
- Failure caching with diskcache
- Progress tracking
"""

import asyncio
from typing import List, Callable, Any, Optional, Dict
from datetime import datetime

from loguru import logger
import diskcache

from .models.base import BaseLLMClient, LLMMessage, LLMResponse
from ..core.exceptions import LLMClientError, RateLimitError


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_rpm: int = 60):
        """
        Initialize the rate limiter.

        Args:
            max_rpm: Maximum requests per minute.
        """
        self.max_rpm = max_rpm
        self.interval = 60.0 / max_rpm  # Seconds between requests
        self._last_request_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request can be made."""
        async with self._lock:
            now = datetime.now()
            if self._last_request_time:
                elapsed = (now - self._last_request_time).total_seconds()
                if elapsed < self.interval:
                    wait_time = self.interval - elapsed
                    await asyncio.sleep(wait_time)

            self._last_request_time = datetime.now()


class BatchWorker:
    """
    Async worker pool for batch LLM processing.

    Manages concurrency, rate limiting, retries, and caching.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_concurrency: int = 5,
        max_rpm: int = 60,
        retry_attempts: int = 5,
        cache_dir: str = ".cache/synthesis",
        cache_ttl: int = 86400,  # 24 hours
    ):
        """
        Initialize the batch worker.

        Args:
            llm_client: LLM client instance.
            max_concurrency: Maximum concurrent requests.
            max_rpm: Maximum requests per minute.
            retry_attempts: Maximum retry attempts per request.
            cache_dir: Directory for failure/result caching.
            cache_ttl: Cache time-to-live in seconds.
        """
        self.llm_client = llm_client
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rate_limiter = RateLimiter(max_rpm)
        self.retry_attempts = retry_attempts
        self.cache = diskcache.Cache(cache_dir)
        self.cache_ttl = cache_ttl

        # Statistics
        self.stats = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "cached": 0,
            "total_tokens": 0,
        }

    async def process_item(
        self,
        item: Dict[str, Any],
        processor_fn: Callable[..., Any],
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single item with retries and caching.

        Args:
            item: Item to process.
            processor_fn: Async function to process the item.
            use_cache: Whether to use caching.

        Returns:
            Processing result.
        """
        item_id = item.get("id", "unknown")
        cache_key = f"result:{item_id}"

        # Check cache
        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit for {item_id}")
            self.stats["cached"] += 1
            return self.cache[cache_key]

        async with self.semaphore:
            try:
                # Rate limiting
                await self.rate_limiter.acquire()

                # Process with retries
                result = await self._process_with_retry(item, processor_fn)

                # Cache successful result
                if use_cache:
                    self.cache.set(cache_key, result, expire=self.cache_ttl)

                self.stats["completed"] += 1
                return result

            except Exception as e:
                logger.error(f"Failed to process {item_id}: {e}")
                self.stats["failed"] += 1

                # Cache failure to avoid reprocessing
                if use_cache:
                    failure_result = {"status": "failed", "error": str(e), "item": item}
                    self.cache.set(cache_key, failure_result, expire=self.cache_ttl)

                raise

    async def _process_with_retry(
        self,
        item: Dict[str, Any],
        processor_fn: Callable[..., Any],
    ) -> Any:
        """
        Process item with exponential backoff retries.

        Args:
            item: Item to process.
            processor_fn: Processing function.

        Returns:
            Processing result.
        """
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                return await processor_fn(item, self.llm_client)
            except RateLimitError as e:
                last_exception = e
                wait_time = (2 ** attempt) * 1.0  # Exponential backoff
                logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
            except LLMClientError as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    wait_time = (2 ** attempt) * 0.5
                    logger.warning(f"LLM error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                last_exception = e
                raise

        raise last_exception or LLMClientError("Max retries exceeded")

    async def run_batch(
        self,
        items: List[Dict[str, Any]],
        processor_fn: Callable[..., Any],
        use_cache: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the worker pool over a batch of items.

        Args:
            items: List of items to process.
            processor_fn: Async function to process each item.
            use_cache: Whether to use caching.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of processing results.
        """
        self.stats["total"] = len(items)
        logger.info(f"Starting batch processing of {len(items)} items (concurrency: {self.semaphore._value})")

        tasks = []
        for i, item in enumerate(items):
            task = asyncio.create_task(
                self._process_with_progress(item, processor_fn, use_cache, i, len(items), progress_callback)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successes and failures
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "failed",
                    "error": str(result),
                    "item": items[i],
                })
            else:
                processed_results.append(result)

        logger.info(
            f"Batch complete: {self.stats['completed']} succeeded, "
            f"{self.stats['failed']} failed, {self.stats['cached']} cached"
        )

        return processed_results

    async def _process_with_progress(
        self,
        item: Dict[str, Any],
        processor_fn: Callable[..., Any],
        use_cache: bool,
        index: int,
        total: int,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> Any:
        """Process item with progress tracking."""
        try:
            result = await self.process_item(item, processor_fn, use_cache)
            if progress_callback:
                progress_callback(index + 1, total)
            return result
        except Exception as e:
            if progress_callback:
                progress_callback(index + 1, total)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["completed"] / self.stats["total"] * 100
                if self.stats["total"] > 0
                else 0
            ),
        }

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self.cache.clear()
        logger.info("Cache cleared")

    async def close(self) -> None:
        """Close the worker and cleanup resources."""
        if hasattr(self.llm_client, "close"):
            await self.llm_client.close()
        self.cache.close()
