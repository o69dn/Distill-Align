"""
SQLite-backed cache manager for synthesis results.

Provides persistent caching with statistics, pruning, and resume support.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel


class CacheStats(BaseModel):
    """Cache statistics."""

    total_entries: int = 0
    hit_count: int = 0
    miss_count: int = 0
    hit_rate: float = 0.0
    db_size_bytes: int = 0
    db_size_mb: float = 0.0
    oldest_entry: Optional[float] = None
    newest_entry: Optional[float] = None


class CacheEntry(BaseModel):
    """A single cache entry."""

    key: str
    value: Any
    model: str = ""
    provider: str = ""
    tokens_used: int = 0
    created_at: float = 0.0
    accessed_at: float = 0.0
    access_count: int = 0


class CacheManager:
    """
    SQLite-backed cache for synthesis results.

    Features:
    - Persistent storage across runs
    - Hit/miss statistics
    - TTL-based expiration
    - Pruning by age or size
    - Resume support (skip cached chunks)
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        db_name: str = "synthesis_cache.db",
        ttl_days: int = 30,
    ):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory for cache database.
            db_name: SQLite database filename.
            ttl_days: Default time-to-live in days.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / db_name
        self.ttl_seconds = ttl_days * 86400

        # Stats tracking (in-memory, resets each session)
        self._hits = 0
        self._misses = 0

        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    model TEXT DEFAULT '',
                    provider TEXT DEFAULT '',
                    tokens_used INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_created_at
                ON cache(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_model
                ON cache(model)
            """)
            conn.commit()

    @staticmethod
    def make_key(content: str, model: str = "", prompt_id: str = "") -> str:
        """
        Generate a deterministic cache key from content and parameters.

        Args:
            content: The input content (chunk text).
            model: Model name used for synthesis.
            prompt_id: Identifier for the prompt template used.

        Returns:
            Hex digest cache key.
        """
        hash_input = f"{content}|{model}|{prompt_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached result.

        Args:
            key: Cache key.

        Returns:
            Cached value as dict, or None if not found/expired.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM cache WHERE key = ?", (key,)
            ).fetchone()

            if row is None:
                self._misses += 1
                return None

            # Check TTL
            if time.time() - row["created_at"] > self.ttl_seconds:
                self._misses += 1
                self.delete(key)
                return None

            # Update access stats
            conn.execute(
                "UPDATE cache SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
                (time.time(), key),
            )
            conn.commit()

            self._hits += 1
            return {
                "value": json.loads(row["value"]),
                "model": row["model"],
                "provider": row["provider"],
                "tokens_used": row["tokens_used"],
                "created_at": row["created_at"],
            }

    def set(
        self,
        key: str,
        value: Any,
        model: str = "",
        provider: str = "",
        tokens_used: int = 0,
    ) -> None:
        """
        Store a result in the cache.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON-serializable).
            model: Model name used.
            provider: Provider name.
            tokens_used: Token count for this request.
        """
        now = time.time()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache
                (key, value, model, provider, tokens_used, created_at, accessed_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (key, json.dumps(value), model, provider, tokens_used, now, now),
            )
            conn.commit()

    def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if entry was deleted.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0

    def has(self, key: str) -> bool:
        """
        Check if a key exists and is not expired.

        Args:
            key: Cache key.

        Returns:
            True if key exists and is valid.
        """
        return self.get(key) is not None

    def stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats with hit rate, entry counts, and size info.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total, MIN(created_at) as oldest, MAX(created_at) as newest FROM cache"
            ).fetchone()

            total = row[0]
            oldest = row[1]
            newest = row[2]

        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return CacheStats(
            total_entries=total,
            hit_count=self._hits,
            miss_count=self._misses,
            hit_rate=hit_rate,
            db_size_bytes=db_size,
            db_size_mb=round(db_size / (1024 * 1024), 2),
            oldest_entry=oldest,
            newest_entry=newest,
        )

    def prune(self, older_than_days: Optional[int] = None) -> int:
        """
        Remove old or expired entries.

        Args:
            older_than_days: Remove entries older than N days. Uses TTL if not specified.

        Returns:
            Number of entries removed.
        """
        cutoff = time.time() - (older_than_days * 86400 if older_than_days else self.ttl_seconds)
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE created_at < ?", (cutoff,))
            conn.commit()
            removed = cursor.rowcount

        if removed > 0:
            logger.info(f"Pruned {removed} cache entries older than {older_than_days or self.ttl_seconds // 86400} days")
        return removed

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("DELETE FROM cache")
            conn.commit()
            removed = cursor.rowcount

        logger.info(f"Cleared {removed} cache entries")
        return removed

    def get_cached_keys(self, model: Optional[str] = None) -> List[str]:
        """
        Get all cached keys, optionally filtered by model.

        Args:
            model: Optional model name filter.

        Returns:
            List of cache keys.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            if model:
                rows = conn.execute(
                    "SELECT key FROM cache WHERE model = ?", (model,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT key FROM cache").fetchall()
        return [row[0] for row in rows]

    def get_total_tokens(self, model: Optional[str] = None) -> int:
        """
        Get total tokens used across all cached entries.

        Args:
            model: Optional model name filter.

        Returns:
            Total token count.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            if model:
                row = conn.execute(
                    "SELECT COALESCE(SUM(tokens_used), 0) FROM cache WHERE model = ?",
                    (model,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(SUM(tokens_used), 0) FROM cache"
                ).fetchone()
        return row[0]

    def close(self) -> None:
        """Close the cache (no-op for SQLite, but maintains interface)."""
        pass
