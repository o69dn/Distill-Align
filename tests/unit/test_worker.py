"""
Tests for the BatchWorker and RateLimiter classes.

Covers:
- RateLimiter timing
- Cache integration (hit/miss)
- Retry logic with exponential backoff
- Batch processing with progress callbacks
- Checkpoint integration
- Cache disable via use_cache flag
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

from distill_align.core.cache import CacheManager
from distill_align.core.exceptions import RateLimitError
from distill_align.synthesis.models.base import BaseLLMClient
from distill_align.synthesis.worker import BatchWorker, RateLimiter

# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_allows_immediate_first_request() -> None:
    """First request should not be delayed."""
    limiter = RateLimiter(max_rpm=60)
    t0 = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"First request was delayed {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_rate_limiter_enforces_interval() -> None:
    """Requests should be spaced by at least the interval."""
    limiter = RateLimiter(max_rpm=120)  # 0.5s interval
    await limiter.acquire()
    t0 = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.4, f"Interval too short: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_rate_limiter_low_rpm() -> None:
    """Low RPM should have longer intervals."""
    limiter = RateLimiter(max_rpm=10)  # 6s interval
    await limiter.acquire()
    t0 = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 5.5, f"Interval too short: {elapsed:.3f}s"


# ------------------------------------------------------------------
# Mock helpers
# ------------------------------------------------------------------


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing worker behavior."""

    def __init__(self, fail_count: int = 0, fail_on_call: set[int] | None = None):
        super().__init__(
            base_url="http://test.local",
            api_key="test",
            model="test-model",
        )
        self.call_count = 0
        self.fail_count = fail_count  # Number of sequential failures before success
        self.fail_on_call = fail_on_call or set()

    async def chat(self, messages: list, **kwargs) -> Any:
        self.call_count += 1
        if self.call_count <= self.fail_count or self.call_count in self.fail_on_call:
            raise RateLimitError("Simulated failure")
        return SimpleNamespace(
            content=f"Response {self.call_count}",
            model="test-model",
            usage={"total_tokens": 10},
        )

    async def complete(self, prompt: str, **kwargs) -> Any:
        return await self.chat(messages=[])

    async def close(self) -> None:
        pass


async def _mock_processor(item: dict[str, Any], client: BaseLLMClient) -> dict[str, Any]:
    """Simple processor that calls the LLM client."""
    result = await client.complete(prompt=str(item.get("data", "")))
    return {"status": "success", "data": result.content}


# ------------------------------------------------------------------
# Cache integration tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_cache_hit(tmp_path) -> None:
    """Verify cached results are returned without calling the LLM."""
    cache_dir = tmp_path / ".cache"
    cache = CacheManager(cache_dir=str(cache_dir))
    client = MockLLMClient()
    worker = BatchWorker(
        llm_client=client,
        max_concurrency=2,
        cache_manager=cache,
        use_cache=True,
    )

    items = [{"id": "1", "data": "hello"}, {"id": "2", "data": "world"}]
    results = await worker.run_batch(items, _mock_processor)

    assert len(results) == 2
    first_call_count = client.call_count

    # Second run should hit cache — no new LLM calls
    results2 = await worker.run_batch(items, _mock_processor)
    assert len(results2) == 2
    assert client.call_count == first_call_count, "Cache was not used"

    # Stats should show cached hits
    stats = worker.get_stats()
    assert stats["cached"] > 0

    await worker.close()


@pytest.mark.asyncio
async def test_worker_cache_disabled(tmp_path) -> None:
    """With cache disabled, every call should go to the LLM."""
    cache_dir = tmp_path / ".cache"
    cache = CacheManager(cache_dir=str(cache_dir))
    client = MockLLMClient()
    worker = BatchWorker(
        llm_client=client,
        max_concurrency=2,
        cache_manager=cache,
        use_cache=False,  # Disabled at construction
    )

    items = [{"id": "1", "data": "hello"}]
    await worker.run_batch(items, _mock_processor)
    first_count = client.call_count

    # Second run — cache should NOT be used
    await worker.run_batch(items, _mock_processor)
    assert client.call_count > first_count, "Cache should not have been used"

    await worker.close()


@pytest.mark.asyncio
async def test_worker_per_call_cache_flag(tmp_path) -> None:
    """The per-call use_cache flag overrides the construction default."""
    cache_dir = tmp_path / ".cache"
    cache = CacheManager(cache_dir=str(cache_dir))
    client = MockLLMClient()
    worker = BatchWorker(
        llm_client=client,
        max_concurrency=2,
        cache_manager=cache,
        use_cache=True,  # Enabled at construction
    )

    items = [{"id": "1", "data": "hello"}]

    # First run with cache
    await worker.run_batch(items, _mock_processor, use_cache=True)
    first_count = client.call_count

    # Second run without cache for this batch
    await worker.run_batch(items, _mock_processor, use_cache=False)
    assert client.call_count > first_count, "Per-call cache disable was ignored"

    await worker.close()


# ------------------------------------------------------------------
# Retry and error handling tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_retry_on_rate_limit_exhausted() -> None:
    """Worker should raise after exhausting retries on persistent failures."""
    client = MockLLMClient(fail_count=10)  # Always fails
    worker = BatchWorker(
        llm_client=client,
        max_concurrency=1,
        retry_attempts=2,  # Only 2 attempts
        use_cache=False,
    )

    items = [{"id": "1", "data": "test"}]
    results = await worker.run_batch(items, _mock_processor)

    # With return_exceptions=True, failures are captured as result dicts
    assert len(results) == 1
    assert results[0]["status"] == "failed", f"Expected failure, got: {results[0]}"


@pytest.mark.asyncio
async def test_worker_retry_then_succeed() -> None:
    """Worker should succeed after transient failures recover."""
    client = MockLLMClient(fail_count=2)  # Fails first 2, succeeds on 3rd
    worker = BatchWorker(
        llm_client=client,
        max_concurrency=1,
        retry_attempts=5,  # Enough retries
        use_cache=False,
    )

    items = [{"id": "1", "data": "test"}]
    results = await worker.run_batch(items, _mock_processor)
    assert len(results) == 1
    assert results[0]["status"] == "success", f"Expected success, got: {results[0]}"


# ------------------------------------------------------------------
# Batch processing tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_processes_all_items() -> None:
    """All items in a batch should be processed."""
    client = MockLLMClient()
    worker = BatchWorker(llm_client=client, max_concurrency=5, use_cache=False)

    items = [{"id": str(i), "data": f"item-{i}"} for i in range(10)]
    results = await worker.run_batch(items, _mock_processor)

    assert len(results) == 10
    successes = sum(1 for r in results if r.get("status") == "success")
    assert successes == 10, f"Expected 10 successes, got {successes}"

    stats = worker.get_stats()
    assert stats["total"] == 10
    assert stats["completed"] == 10

    await worker.close()


@pytest.mark.asyncio
async def test_batch_concurrency_limit(tmp_path) -> None:
    """Concurrency should be limited by the semaphore."""
    cache_dir = tmp_path / ".cache"
    cache = CacheManager(cache_dir=str(cache_dir))
    client = MockLLMClient()
    worker = BatchWorker(
        llm_client=client,
        max_concurrency=2,
        max_rpm=6000,  # High RPM to avoid rate-limiter serialization
        cache_manager=cache,
        use_cache=False,
    )

    # Track concurrent calls
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def _concurrent_processor(item: dict, llm_client: BaseLLMClient) -> dict:
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.2)  # Simulate work — long enough for concurrency to kick in
        async with lock:
            active -= 1
        return {"status": "success", "data": "ok"}

    items = [{"id": str(i), "data": f"item-{i}"} for i in range(6)]
    await worker.run_batch(items, _concurrent_processor, use_cache=False)

    assert max_active <= 2, f"Expected max 2 concurrent, got {max_active}"
    assert max_active >= 2, f"Expected at least 2 concurrent, got {max_active}"

    await worker.close()


@pytest.mark.asyncio
async def test_progress_callback() -> None:
    """Progress callback should be called with correct counts."""
    client = MockLLMClient()
    worker = BatchWorker(llm_client=client, max_concurrency=5, use_cache=False)

    progress_updates: list[tuple[int, int]] = []

    def progress(current: int, total: int) -> None:
        progress_updates.append((current, total))

    items = [{"id": str(i), "data": f"item-{i}"} for i in range(5)]
    await worker.run_batch(items, _mock_processor, progress_callback=progress)

    assert len(progress_updates) == 5
    assert progress_updates[-1] == (5, 5), f"Final progress: {progress_updates[-1]}"

    await worker.close()


# ------------------------------------------------------------------
# Stats tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_stats() -> None:
    """Worker stats should accurately reflect processing."""
    client = MockLLMClient(fail_count=3)
    worker = BatchWorker(
        llm_client=client,
        max_concurrency=2,
        retry_attempts=2,
        use_cache=False,
    )

    items = [{"id": "1", "data": "fail"}, {"id": "2", "data": "ok"}]
    await worker.run_batch(items, _mock_processor)

    stats = worker.get_stats()
    assert stats["total"] == 2
    assert stats["completed"] + stats["failed"] == 2
    assert 0 <= stats["success_rate"] <= 100

    await worker.close()
