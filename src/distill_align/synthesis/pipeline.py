"""
Synthesis pipeline orchestrator.

Handles the full synthesis workflow: processing chunks through LLMs, applying Socratic
Transformer and Scaffold Action pipelines, and outputting structured conversations.

Supports:
- Checkpoint-based resume for crash recovery
- Cache-aware processing (skip already-synthesized chunks)
- Job tracking with statistics
"""

import uuid
from collections.abc import Callable
from typing import Any

from loguru import logger

from ..core.cache import CacheManager
from ..core.checkpoint import CheckpointManager
from ..core.exceptions import LLMClientError, SynthesisError
from ..core.schemas import (
    ConversationSchema,
    DataChunk,
    SynthesisConfig,
    SynthesizedTurn,
)
from .models.base import BaseLLMClient
from .models.ollama import OllamaClient
from .models.openai import OpenAIClient
from .models.vllm import VLLMClient
from .prompts.scaffold import SCAFFOLD_SYSTEM_PROMPT, render_scaffold_prompt
from .prompts.socratic import SOCRATIC_SYSTEM_PROMPT, render_socratic_prompt
from .pruner import ContentPruner
from .worker import BatchWorker


class SynthesisPipeline:
    """Orchestrates the synthesis of DataChunks into structured conversations."""

    def __init__(
        self,
        config: SynthesisConfig | None = None,
        cache_manager: CacheManager | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        """
        Initialize the synthesis pipeline.

        Args:
            config: Optional synthesis configuration.
            cache_manager: Optional cache manager for persistent caching.
            checkpoint_manager: Optional checkpoint manager for resume support.
        """
        self.config = config or SynthesisConfig()
        self._client: BaseLLMClient | None = None
        self._worker: BatchWorker | None = None
        self._pruner = ContentPruner()
        self._cache = cache_manager
        self._checkpoint = checkpoint_manager

    def _get_client(self) -> BaseLLMClient:
        """Get or create the LLM client."""
        if self._client is None:
            if self.config.llm_provider == "openai":
                self._client = OpenAIClient(
                    base_url=self.config.base_url or "https://api.openai.com/v1",
                    api_key=self.config.api_key,
                    model=self.config.model_name,
                )
            elif self.config.llm_provider == "ollama":
                self._client = OllamaClient(
                    base_url=self.config.base_url or "http://localhost:11434",
                    model=self.config.model_name,
                )
            elif self.config.llm_provider == "vllm":
                self._client = VLLMClient(
                    base_url=self.config.base_url or "http://localhost:8000/v1",
                    model=self.config.model_name,
                )
            else:
                raise SynthesisError(f"Unknown provider: {self.config.llm_provider}")
        return self._client

    def _get_worker(self) -> BatchWorker:
        """Get or create the batch worker."""
        if self._worker is None:
            client = self._get_client()
            self._worker = BatchWorker(
                llm_client=client,
                max_concurrency=self.config.max_concurrency,
                max_rpm=self.config.max_rpm,
                retry_attempts=self.config.retry_attempts,
                cache_manager=self._cache,
                checkpoint_manager=self._checkpoint,
            )
        return self._worker

    async def synthesize_chunk(
        self,
        chunk: DataChunk,
        llm_client: BaseLLMClient,
    ) -> ConversationSchema:
        """
        Synthesize a single chunk into a conversation.

        Args:
            chunk: DataChunk to synthesize.
            llm_client: LLM client to use.

        Returns:
            ConversationSchema object.

        Raises:
            SynthesisError: If synthesis fails.
        """
        chunk_id = chunk.id
        logger.debug(f"Synthesizing chunk {chunk_id}")

        metadata = {
            "source_type": chunk.metadata.source_type,
            "title": chunk.metadata.title,
            "language": chunk.metadata.language,
            "section_headers": chunk.metadata.section_headers,
            "module_path": chunk.metadata.module_path,
            "functions": chunk.metadata.custom_tags.get("functions", []),
            "classes": chunk.metadata.custom_tags.get("classes", []),
        }

        conversation = None

        # Step 1: Socratic Transformer
        if self.config.socratic_enabled:
            conversation = await self._apply_socratic(chunk.content, metadata, llm_client)

        # Step 2: Scaffold Action
        if self.config.scaffold_enabled and conversation:
            conversation = await self._apply_scaffold(conversation, metadata, llm_client)

        # Fallback: simple conversation
        if conversation is None:
            conversation = ConversationSchema(
                id=str(uuid.uuid4()),
                source_chunk_id=chunk_id,
                turns=[
                    SynthesizedTurn(role="user", content=f"Explain the following:\n\n{chunk.content[:500]}"),
                    SynthesizedTurn(role="assistant", content=chunk.content),
                ],
            )

        # Step 3: Prune and validate
        pruned = self._pruner.prune_conversation(conversation)
        if pruned is None:
            raise SynthesisError(f"Conversation failed quality check for chunk {chunk_id}")

        return pruned

    async def _apply_socratic(
        self,
        content: str,
        metadata: dict[str, Any],
        llm_client: BaseLLMClient,
    ) -> ConversationSchema | None:
        """Apply the Socratic Transformer pipeline."""
        user_prompt = render_socratic_prompt(content, metadata)

        try:
            response = await llm_client.generate(
                system_prompt=SOCRATIC_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.config.temperature,
            )

            parsed = self._pruner.extract_json_from_response(response)
            if not parsed or "conversation" not in parsed:
                logger.warning("Failed to parse Socratic response")
                return None

            turns = [SynthesizedTurn(role=t["role"], content=t["content"]) for t in parsed["conversation"]]

            return ConversationSchema(
                id=str(uuid.uuid4()),
                source_chunk_id="",
                turns=turns,
                reasoning_trace=parsed.get("reasoning_trace"),
            )

        except LLMClientError as e:
            logger.error(f"Socratic pipeline failed: {e}")
            return None

    async def _apply_scaffold(
        self,
        conversation: ConversationSchema,
        metadata: dict[str, Any],
        llm_client: BaseLLMClient,
    ) -> ConversationSchema:
        """Apply the Scaffold Action pipeline to clean assistant responses."""
        cleaned_turns = []

        for turn in conversation.turns:
            if turn.role == "assistant":
                user_prompt = render_scaffold_prompt(
                    turn.content,
                    {**metadata, "extraction_type": "auto"},
                )
                try:
                    response = await llm_client.generate(
                        system_prompt=SCAFFOLD_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        temperature=0.3,
                    )
                    parsed = self._pruner.extract_json_from_response(response)
                    if parsed and "extracted_content" in parsed:
                        cleaned_turns.append(
                            SynthesizedTurn(
                                role="assistant",
                                content=parsed["extracted_content"],
                            )
                        )
                    else:
                        cleaned_turns.append(turn)
                except LLMClientError as e:
                    logger.warning(f"Scaffold failed for turn, keeping original: {e}")
                    cleaned_turns.append(turn)
            else:
                cleaned_turns.append(turn)

        return ConversationSchema(
            id=conversation.id,
            source_chunk_id=conversation.source_chunk_id,
            turns=cleaned_turns,
            reasoning_trace=conversation.reasoning_trace,
            confidence_score=conversation.confidence_score,
        )

    async def synthesize_batch(
        self,
        chunks: list[DataChunk],
        progress_callback: Callable[[int, int], None] | None = None,
        job_id: str | None = None,
        resume: bool = False,
    ) -> list[ConversationSchema]:
        """
        Synthesize a batch of chunks into conversations.

        Args:
            chunks: List of DataChunks to synthesize.
            progress_callback: Optional callback for progress updates.
            job_id: Optional job ID for checkpointing/resume.
            resume: Whether to resume an existing job.

        Returns:
            List of ConversationSchema objects.
        """
        worker = self._get_worker()

        # Create or resume checkpoint
        if self._checkpoint and job_id:
            if resume:
                checkpoint = self._checkpoint.load_job(job_id)
                if checkpoint:
                    logger.info(f"Resuming job {job_id}: {checkpoint.processed_items}/{checkpoint.total_items} done")
                    # Filter out already-processed chunks
                    processed_set = set(checkpoint.processed_ids)
                    chunks = [c for c in chunks if c.id not in processed_set]
                    logger.info(f"Processing {len(chunks)} remaining chunks")
                    self._checkpoint.start_job(job_id)
                else:
                    logger.warning(f"Job {job_id} not found, starting fresh")
                    self._checkpoint.create_job("synthesize", total_items=len(chunks), job_id=job_id)
                    self._checkpoint.start_job(job_id)
            else:
                self._checkpoint.create_job("synthesize", total_items=len(chunks), job_id=job_id)
                self._checkpoint.start_job(job_id)

        # Prepare items for worker
        items = [{"id": chunk.id, "chunk": chunk} for chunk in chunks]

        async def processor(item: dict[str, Any], llm_client: BaseLLMClient) -> dict[str, Any]:
            chunk = item["chunk"]
            conversation = await self.synthesize_chunk(chunk, llm_client)
            return {"status": "success", "conversation": conversation.model_dump()}

        # Run batch
        results = await worker.run_batch(
            items=items,
            processor_fn=processor,
            job_id=job_id,
            progress_callback=progress_callback,
        )

        # Extract conversations
        conversations = []
        for result in results:
            if result.get("status") == "success":
                conv_data = result["conversation"]
                conversations.append(ConversationSchema(**conv_data))
            else:
                logger.warning(f"Failed to synthesize: {result.get('error')}")

        # Complete checkpoint
        if self._checkpoint and job_id:
            stats = worker.get_stats()
            if worker.stats["failed"] == 0:
                self._checkpoint.complete_job(job_id, stats=stats)
            else:
                self._checkpoint.fail_job(job_id, error=f"{worker.stats['failed']} items failed")

        logger.info(f"Synthesized {len(conversations)} conversations from {len(chunks)} chunks")
        return conversations

    async def close(self) -> None:
        """Close the pipeline and cleanup resources."""
        if self._worker:
            await self._worker.close()
        if self._client and hasattr(self._client, "close"):
            await self._client.close()
