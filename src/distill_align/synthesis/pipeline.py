"""
Synthesis pipeline orchestrator.

Handles the full synthesis workflow: processing chunks through LLMs, applying Socratic
Transformer and Scaffold Action pipelines, and outputting structured conversations.
"""

import asyncio
import json
import uuid
from typing import List, Optional, Dict, Any, Callable

from loguru import logger

from ..core.schemas import (
    DataChunk,
    ConversationSchema,
    SynthesizedTurn,
    SynthesisConfig,
)
from ..core.exceptions import SynthesisError
from .models.base import BaseLLMClient, LLMMessage
from .models.openai import OpenAIClient
from .models.ollama import OllamaClient
from .models.vllm import VLLMClient
from .worker import BatchWorker
from .prompts.socratic import SOCRATIC_SYSTEM_PROMPT, render_socratic_prompt
from .prompts.scaffold import SCAFFOLD_SYSTEM_PROMPT, render_scaffold_prompt
from .pruner import ContentPruner


class SynthesisPipeline:
    """Orchestrates the synthesis of DataChunks into structured conversations."""

    def __init__(self, config: Optional[SynthesisConfig] = None):
        """
        Initialize the synthesis pipeline.

        Args:
            config: Optional synthesis configuration. Uses defaults if not provided.
        """
        self.config = config or SynthesisConfig()
        self._client: Optional[BaseLLMClient] = None
        self._worker: Optional[BatchWorker] = None
        self._pruner = ContentPruner()

    def _get_client(self) -> BaseLLMClient:
        """
        Get or create the LLM client.

        Returns:
            LLM client instance.
        """
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
        """
        Get or create the batch worker.

        Returns:
            BatchWorker instance.
        """
        if self._worker is None:
            client = self._get_client()
            self._worker = BatchWorker(
                llm_client=client,
                max_concurrency=self.config.max_concurrency,
                max_rpm=self.config.max_rpm,
                retry_attempts=self.config.retry_attempts,
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

        # Prepare metadata for prompts
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

        # Step 1: Socratic Transformer (if enabled)
        if self.config.socratic_enabled:
            conversation = await self._apply_socratic(chunk.content, metadata, llm_client)

        # Step 2: Scaffold Action (if enabled)
        if self.config.scaffold_enabled and conversation:
            conversation = await self._apply_scaffold(conversation, metadata, llm_client)

        # If neither pipeline ran, create a simple conversation
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
        metadata: Dict[str, Any],
        llm_client: BaseLLMClient,
    ) -> Optional[ConversationSchema]:
        """
        Apply the Socratic Transformer pipeline.

        Args:
            content: Raw content.
            metadata: Content metadata.
            llm_client: LLM client.

        Returns:
            ConversationSchema or None if parsing fails.
        """
        user_prompt = render_socratic_prompt(content, metadata)

        try:
            response = await llm_client.generate(
                system_prompt=SOCRATIC_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.config.temperature,
            )

            # Parse response
            parsed = self._pruner.extract_json_from_response(response)
            if not parsed or "conversation" not in parsed:
                logger.warning("Failed to parse Socratic response")
                return None

            # Convert to ConversationSchema
            turns = [
                SynthesizedTurn(role=t["role"], content=t["content"])
                for t in parsed["conversation"]
            ]

            return ConversationSchema(
                id=str(uuid.uuid4()),
                source_chunk_id="",  # Will be set by caller
                turns=turns,
                reasoning_trace=parsed.get("reasoning_trace"),
            )

        except Exception as e:
            logger.error(f"Socratic pipeline failed: {e}")
            return None

    async def _apply_scaffold(
        self,
        conversation: ConversationSchema,
        metadata: Dict[str, Any],
        llm_client: BaseLLMClient,
    ) -> ConversationSchema:
        """
        Apply the Scaffold Action pipeline to clean assistant responses.

        Args:
            conversation: Input conversation.
            metadata: Content metadata.
            llm_client: LLM client.

        Returns:
            Cleaned ConversationSchema.
        """
        cleaned_turns = []

        for turn in conversation.turns:
            if turn.role == "assistant":
                # Apply scaffold to assistant responses
                user_prompt = render_scaffold_prompt(
                    turn.content,
                    {**metadata, "extraction_type": "auto"},
                )

                try:
                    response = await llm_client.generate(
                        system_prompt=SCAFFOLD_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        temperature=0.3,  # Lower temperature for extraction
                    )

                    parsed = self._pruner.extract_json_from_response(response)
                    if parsed and "extracted_content" in parsed:
                        cleaned_turns.append(SynthesizedTurn(
                            role="assistant",
                            content=parsed["extracted_content"],
                        ))
                    else:
                        # Keep original if parsing fails
                        cleaned_turns.append(turn)

                except Exception as e:
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
        chunks: List[DataChunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ConversationSchema]:
        """
        Synthesize a batch of chunks into conversations.

        Args:
            chunks: List of DataChunks to synthesize.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of ConversationSchema objects.
        """
        worker = self._get_worker()

        # Prepare items for worker
        items = [{"id": chunk.id, "chunk": chunk} for chunk in chunks]

        # Define processor function
        async def processor(item: Dict[str, Any], llm_client: BaseLLMClient) -> Dict[str, Any]:
            chunk = item["chunk"]
            conversation = await self.synthesize_chunk(chunk, llm_client)
            return {"status": "success", "conversation": conversation.model_dump()}

        # Run batch
        results = await worker.run_batch(
            items=items,
            processor_fn=processor,
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

        logger.info(f"Synthesized {len(conversations)} conversations from {len(chunks)} chunks")
        return conversations

    async def close(self) -> None:
        """Close the pipeline and cleanup resources."""
        if self._worker:
            await self._worker.close()
        if self._client and hasattr(self._client, "close"):
            await self._client.close()
