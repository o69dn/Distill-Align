"""
Pipeline checkpoint system for save/resume support.

Tracks job state across runs so synthesis can survive crashes.
"""

import json
import time
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    """Status of a pipeline job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESUMED = "resumed"


class JobCheckpoint(BaseModel):
    """Checkpoint state for a pipeline job."""

    job_id: str
    job_type: str  # "ingest", "synthesize", "export", "full_pipeline"
    status: JobStatus = JobStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    completed_at: float | None = None

    # Progress tracking
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0

    # Item tracking
    processed_ids: list[str] = Field(default_factory=list)
    failed_ids: list[str] = Field(default_factory=list)
    failed_errors: dict[str, str] = Field(default_factory=dict)

    # Stats
    stats: dict[str, Any] = Field(default_factory=dict)

    # Configuration snapshot
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def progress_pct(self) -> float:
        """Progress as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    @property
    def remaining_items(self) -> int:
        """Number of items still to process."""
        return self.total_items - self.processed_items - self.failed_items

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        end = self.completed_at or time.time()
        return end - self.created_at

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time remaining in seconds."""
        if self.processed_items == 0:
            return None
        elapsed = self.elapsed_seconds
        rate = self.processed_items / elapsed
        remaining = self.remaining_items
        return remaining / rate if rate > 0 else None

    @property
    def is_terminal(self) -> bool:
        """Whether the job is in a terminal state."""
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class CheckpointManager:
    """
    Manages pipeline job checkpoints.

    Stores job state to disk so pipelines can survive crashes and resume.
    """

    def __init__(self, checkpoint_dir: str | Path = ".distill-align/jobs"):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        """Get the checkpoint file path for a job."""
        return self.checkpoint_dir / f"{job_id}.json"

    def create_job(
        self,
        job_type: str,
        total_items: int = 0,
        config: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> JobCheckpoint:
        """
        Create a new job checkpoint.

        Args:
            job_type: Type of job (ingest, synthesize, export, full_pipeline).
            total_items: Total number of items to process.
            config: Configuration snapshot.
            job_id: Optional custom job ID.

        Returns:
            New JobCheckpoint.
        """
        if job_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = uuid.uuid4().hex[:6]
            job_id = f"{job_type}_{timestamp}_{short_id}"

        checkpoint = JobCheckpoint(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            total_items=total_items,
            config=config or {},
        )

        self._save(checkpoint)
        logger.info(f"Created job {job_id} ({job_type}, {total_items} items)")
        return checkpoint

    def load_job(self, job_id: str) -> JobCheckpoint | None:
        """
        Load a job checkpoint from disk.

        Args:
            job_id: Job identifier.

        Returns:
            JobCheckpoint or None if not found.
        """
        path = self._job_path(job_id)
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return JobCheckpoint(**data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {job_id}: {e}")
            return None

    def save_job(self, checkpoint: JobCheckpoint) -> None:
        """
        Save a job checkpoint to disk.

        Args:
            checkpoint: JobCheckpoint to save.
        """
        checkpoint.updated_at = time.time()
        self._save(checkpoint)

    def _save(self, checkpoint: JobCheckpoint) -> None:
        """Internal save method."""
        path = self._job_path(checkpoint.job_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.model_dump(), f, indent=2, default=str)

    def start_job(self, job_id: str) -> JobCheckpoint | None:
        """
        Mark a job as running.

        Args:
            job_id: Job identifier.

        Returns:
            Updated checkpoint or None.
        """
        checkpoint = self.load_job(job_id)
        if checkpoint is None:
            return None

        checkpoint.status = JobStatus.RUNNING
        checkpoint.updated_at = time.time()
        self.save_job(checkpoint)
        return checkpoint

    def complete_job(self, job_id: str, stats: dict[str, Any] | None = None) -> JobCheckpoint | None:
        """
        Mark a job as completed.

        Args:
            job_id: Job identifier.
            stats: Final statistics.

        Returns:
            Updated checkpoint or None.
        """
        checkpoint = self.load_job(job_id)
        if checkpoint is None:
            return None

        checkpoint.status = JobStatus.COMPLETED
        checkpoint.completed_at = time.time()
        checkpoint.updated_at = time.time()
        if stats:
            checkpoint.stats = stats
        self.save_job(checkpoint)
        logger.info(f"Job {job_id} completed: {checkpoint.processed_items}/{checkpoint.total_items} items")
        return checkpoint

    def fail_job(self, job_id: str, error: str = "") -> JobCheckpoint | None:
        """
        Mark a job as failed.

        Args:
            job_id: Job identifier.
            error: Error message.

        Returns:
            Updated checkpoint or None.
        """
        checkpoint = self.load_job(job_id)
        if checkpoint is None:
            return None

        checkpoint.status = JobStatus.FAILED
        checkpoint.completed_at = time.time()
        checkpoint.updated_at = time.time()
        if error:
            checkpoint.stats["error"] = error
        self.save_job(checkpoint)
        logger.error(f"Job {job_id} failed: {error}")
        return checkpoint

    def record_processed(self, job_id: str, item_id: str) -> JobCheckpoint | None:
        """
        Record a successfully processed item.

        Args:
            job_id: Job identifier.
            item_id: ID of the processed item.

        Returns:
            Updated checkpoint or None.
        """
        checkpoint = self.load_job(job_id)
        if checkpoint is None:
            return None

        if item_id not in checkpoint.processed_ids:
            checkpoint.processed_ids.append(item_id)
            checkpoint.processed_items = len(checkpoint.processed_ids)
            checkpoint.updated_at = time.time()
            self.save_job(checkpoint)

        return checkpoint

    def record_failed(self, job_id: str, item_id: str, error: str = "") -> JobCheckpoint | None:
        """
        Record a failed item.

        Args:
            job_id: Job identifier.
            item_id: ID of the failed item.
            error: Error message.

        Returns:
            Updated checkpoint or None.
        """
        checkpoint = self.load_job(job_id)
        if checkpoint is None:
            return None

        if item_id not in checkpoint.failed_ids:
            checkpoint.failed_ids.append(item_id)
            checkpoint.failed_items = len(checkpoint.failed_ids)
            if error:
                checkpoint.failed_errors[item_id] = error
            checkpoint.updated_at = time.time()
            self.save_job(checkpoint)

        return checkpoint

    def is_processed(self, job_id: str, item_id: str) -> bool:
        """
        Check if an item has already been processed.

        Args:
            job_id: Job identifier.
            item_id: ID of the item.

        Returns:
            True if item was already processed.
        """
        checkpoint = self.load_job(job_id)
        if checkpoint is None:
            return False
        return item_id in checkpoint.processed_ids

    def get_unprocessed_ids(self, job_id: str, all_ids: list[str]) -> list[str]:
        """
        Get IDs that haven't been processed yet.

        Args:
            job_id: Job identifier.
            all_ids: All item IDs.

        Returns:
            List of unprocessed IDs.
        """
        checkpoint = self.load_job(job_id)
        if checkpoint is None:
            return all_ids

        processed = set(checkpoint.processed_ids)
        failed = set(checkpoint.failed_ids)
        return [id_ for id_ in all_ids if id_ not in processed and id_ not in failed]

    def list_jobs(
        self,
        status: JobStatus | None = None,
        job_type: str | None = None,
        limit: int = 20,
    ) -> list[JobCheckpoint]:
        """
        List all jobs, optionally filtered.

        Args:
            status: Filter by status.
            job_type: Filter by job type.
            limit: Maximum number of jobs to return.

        Returns:
            List of JobCheckpoint objects.
        """
        jobs = []
        for path in self.checkpoint_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                job = JobCheckpoint(**data)

                if status and job.status != status:
                    continue
                if job_type and job.job_type != job_type:
                    continue

                jobs.append(job)
            except Exception:
                continue

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job checkpoint.

        Args:
            job_id: Job identifier.

        Returns:
            True if job was deleted.
        """
        path = self._job_path(job_id)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted job checkpoint {job_id}")
            return True
        return False

    def cleanup_old_jobs(self, older_than_days: int = 30) -> int:
        """
        Remove old job checkpoints.

        Args:
            older_than_days: Remove jobs older than N days.

        Returns:
            Number of jobs removed.
        """
        cutoff = time.time() - (older_than_days * 86400)
        removed = 0

        for path in self.checkpoint_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("created_at", 0) < cutoff:
                    path.unlink()
                    removed += 1
            except Exception:
                continue

        if removed > 0:
            logger.info(f"Cleaned up {removed} old job checkpoints")
        return removed
