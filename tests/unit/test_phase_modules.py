"""
Unit tests for the new Phase 1-5 components.
"""

import pytest

from distill_align.core.cache import CacheManager
from distill_align.core.checkpoint import CheckpointManager, JobStatus
from distill_align.core.config_file import (
    DistillAlignConfig,
    generate_default_config,
    load_config,
    save_config,
)
from distill_align.exporter.splitter import DatasetSplitter
from distill_align.exporter.validator import DatasetValidator
from distill_align.synthesis.tokenizer import Tokenizer

# =============================================================================
# CacheManager Tests
# =============================================================================


class TestCacheManager:
    """Tests for the SQLite-backed CacheManager."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a temporary cache."""
        return CacheManager(cache_dir=str(tmp_path), ttl_days=1)

    def test_set_and_get(self, cache):
        cache.set(key="test_key", value={"data": "hello"})
        result = cache.get("test_key")
        assert result is not None
        assert result["value"] == {"data": "hello"}

    def test_cache_miss(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_make_key_deterministic(self):
        key1 = CacheManager.make_key("content", "model", "prompt_id")
        key2 = CacheManager.make_key("content", "model", "prompt_id")
        assert key1 == key2

    def test_make_key_unique(self):
        key1 = CacheManager.make_key("content1", "model", "prompt_id")
        key2 = CacheManager.make_key("content2", "model", "prompt_id")
        assert key1 != key2

    def test_stats(self, cache):
        cache.set("k1", "v1")
        cache.get("k1")  # Hit
        cache.get("k2")  # Miss

        stats = cache.stats()
        assert stats.total_entries == 1
        assert stats.hit_count == 1
        assert stats.miss_count == 1
        assert stats.hit_rate == 0.5

    def test_clear(self, cache):
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        removed = cache.clear()
        assert removed == 2
        assert cache.get("k1") is None


# =============================================================================
# CheckpointManager Tests
# =============================================================================


class TestCheckpointManager:
    """Tests for the CheckpointManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a temporary checkpoint manager."""
        return CheckpointManager(checkpoint_dir=str(tmp_path))

    def test_create_job(self, manager):
        job = manager.create_job("synthesize", total_items=100)
        assert job.job_type == "synthesize"
        assert job.total_items == 100
        assert job.status == JobStatus.PENDING

    def test_start_and_complete_job(self, manager):
        job = manager.create_job("synthesize", total_items=10)
        manager.start_job(job.job_id)
        manager.record_processed(job.job_id, "item-1")
        manager.record_processed(job.job_id, "item-2")
        completed = manager.complete_job(job.job_id, stats={"total_tokens": 1000})

        assert completed.status == JobStatus.COMPLETED
        assert completed.processed_items == 2

    def test_record_failed(self, manager):
        job = manager.create_job("synthesize", total_items=10)
        manager.record_failed(job.job_id, "item-1", "Test error")
        checkpoint = manager.load_job(job.job_id)
        assert checkpoint.failed_items == 1
        assert "item-1" in checkpoint.failed_errors

    def test_get_unprocessed_ids(self, manager):
        job = manager.create_job("synthesize", total_items=5)
        manager.record_processed(job.job_id, "a")
        manager.record_processed(job.job_id, "b")

        all_ids = ["a", "b", "c", "d", "e"]
        unprocessed = manager.get_unprocessed_ids(job.job_id, all_ids)
        assert set(unprocessed) == {"c", "d", "e"}

    def test_list_jobs(self, manager):
        manager.create_job("synthesize", total_items=10)
        manager.create_job("ingest", total_items=5)

        jobs = manager.list_jobs()
        assert len(jobs) == 2

    def test_delete_job(self, manager):
        job = manager.create_job("test", total_items=1)
        assert manager.delete_job(job.job_id)
        assert manager.load_job(job.job_id) is None


# =============================================================================
# Config File Tests
# =============================================================================


class TestConfigFile:
    """Tests for config file loading/saving."""

    def test_generate_default_config(self, tmp_path):
        config_path = generate_default_config("test-project", str(tmp_path / "config.yaml"))
        assert config_path.exists()
        content = config_path.read_text()
        assert "test-project" in content

    def test_save_and_load_config(self, tmp_path):
        config = DistillAlignConfig()
        config.project.name = "roundtrip-test"

        path = tmp_path / "test.yaml"
        save_config(config, str(path))
        loaded = load_config(str(path))

        assert loaded.project.name == "roundtrip-test"

    def test_load_nonexistent_uses_defaults(self, tmp_path, monkeypatch):
        # Change to a temp dir so find_config_file doesn't pick up an existing config
        monkeypatch.chdir(tmp_path)
        # Explicitly pass a None path so we go through the find_config_file branch
        config = load_config(None)
        assert config.project.name == "my-dataset"


# =============================================================================
# Dataset Splitter Tests
# =============================================================================


class TestDatasetSplitter:
    """Tests for DatasetSplitter."""

    @pytest.fixture
    def conversations(self):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        return [
            ConversationSchema(
                id=f"conv-{i}",
                source_chunk_id=f"chunk-{i // 2}",
                turns=[
                    SynthesizedTurn(role="user", content=f"Q{i}"),
                    SynthesizedTurn(role="assistant", content=f"A{i}"),
                ],
            )
            for i in range(10)
        ]

    def test_split(self, conversations):
        splitter = DatasetSplitter(seed=42)
        result = splitter.split(conversations, 0.8, 0.1, 0.1)

        assert len(result.train) == 8
        assert len(result.val) == 1
        assert len(result.test) == 1
        assert result.total == 10

    def test_split_invalid_ratios(self, conversations):
        splitter = DatasetSplitter()
        with pytest.raises(ValueError):
            splitter.split(conversations, 0.5, 0.3, 0.3)  # Sums to 1.1

    def test_stratified_split(self, conversations):
        splitter = DatasetSplitter(seed=42)
        result = splitter.split(conversations, 0.8, 0.1, 0.1, stratify_by="source_chunk_id")
        # Each source_chunk_id should have its items split proportionally
        assert result.total == 10


# =============================================================================
# Dataset Validator Tests
# =============================================================================


class TestDatasetValidator:
    """Tests for DatasetValidator."""

    @pytest.fixture
    def conversations(self):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        return [
            ConversationSchema(
                id=f"conv-{i}",
                source_chunk_id="chunk-1",
                turns=[
                    SynthesizedTurn(role="system", content="You are helpful."),
                    SynthesizedTurn(role="user", content=f"Q{i}"),
                    SynthesizedTurn(role="assistant", content=f"A{i}"),
                ],
            )
            for i in range(5)
        ]

    def test_validate_good_dataset(self, conversations):
        validator = DatasetValidator()
        report = validator.validate(conversations)

        assert report.is_valid
        assert report.quality_score > 0.5
        assert report.stats.total_conversations == 5

    def test_validate_empty_dataset(self):
        validator = DatasetValidator()
        report = validator.validate([])

        assert report.stats.total_conversations == 0
        assert not report.is_valid  # Error: too few conversations

    def test_deduplicate(self):
        from distill_align.core.schemas import ConversationSchema, SynthesizedTurn

        conv = ConversationSchema(
            id="1",
            source_chunk_id="c1",
            turns=[
                SynthesizedTurn(role="user", content="Hi"),
                SynthesizedTurn(role="assistant", content="Hello!"),
            ],
        )

        validator = DatasetValidator()
        unique = validator.deduplicate([conv, conv, conv])
        assert len(unique) == 1


# =============================================================================
# Tokenizer Tests
# =============================================================================


class TestTokenizer:
    """Tests for Tokenizer."""

    def test_count_tokens_fallback(self):
        tokenizer = Tokenizer(model="unknown-model")
        count = tokenizer.count_tokens("Hello, world!")
        assert count > 0

    def test_count_message_tokens(self):
        tokenizer = Tokenizer(model="unknown-model")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        count = tokenizer.count_message_tokens(messages)
        assert count > 0

    def test_estimate_cost_zero_for_oss(self):
        tokenizer = Tokenizer(model="llama3.1")
        cost = tokenizer.estimate_cost(1000, 500)
        assert cost == 0.0  # Open-source models

    def test_estimate_cost_for_openai(self):
        tokenizer = Tokenizer(model="gpt-4o")
        cost = tokenizer.estimate_cost(1_000_000, 0)
        assert cost > 0  # Should have cost

    def test_record_and_get_stats(self):
        tokenizer = Tokenizer(model="gpt-4o")
        tokenizer.record_usage(100, 50)
        tokenizer.record_usage(200, 100)

        stats = tokenizer.get_stats()
        assert stats.total_input_tokens == 300
        assert stats.total_output_tokens == 150
        assert stats.num_requests == 2

    def test_estimate_batch_cost(self):
        tokenizer = Tokenizer(model="gpt-4o")
        texts = ["Hello world", "Another text", "More content"]
        estimate = tokenizer.estimate_batch_cost(texts, estimated_output_tokens=100)
        assert estimate.num_requests == 3
        assert estimate.total_input_tokens > 0
        assert estimate.total_output_tokens == 300
