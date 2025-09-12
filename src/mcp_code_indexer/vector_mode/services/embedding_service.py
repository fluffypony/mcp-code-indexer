"""
Embedding Service for converting code chunks to vector embeddings.

Provides a clean abstraction layer between chunked code and embedding providers,
handling text preparation, batching, and provider communication.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Tuple



from ..chunking.ast_chunker import CodeChunk
from ..providers.voyage_client import VoyageClient
from ..config import VectorConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for converting code chunks to embeddings.

    Handles text preparation, context enhancement, batching, and provider
    communication while maintaining separation from daemon orchestration logic.
    """

    def __init__(self, embedding_client: VoyageClient, config: VectorConfig):
        """Initialize embedding service with client and configuration."""
        self.embedding_client = embedding_client
        self.config = config

        # Validate API access immediately during initialization
        self.embedding_client.validate_api_access()

    async def generate_embeddings_for_chunks(
        self, chunks: List[CodeChunk], project_name: str, file_path: Path
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of code chunks.

        Args:
            chunks: List of code chunks to embed
            project_name: Name of the project (for logging)
            file_path: Path to source file (for logging)

        Returns:
            List of embedding vectors (one per chunk)

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not chunks:
            logger.debug(f"No chunks provided for {file_path}")
            return []

        logger.info(
            f"Generating embeddings for {len(chunks)} chunks from {file_path}",
            extra={
                "structured_data": {
                    "project_name": project_name,
                    "file_path": str(file_path),
                    "chunk_count": len(chunks),
                    "embedding_model": self.config.embedding_model,
                }
            },
        )

        try:
            # Extract text content from chunks with context enhancement
            texts = self._prepare_chunk_texts(chunks)

            # Process chunks in batches to respect API limits
            all_embeddings = []
            batch_size = self.config.batch_size

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_chunks = chunks[i : i + batch_size]

                logger.debug(
                    f"Processing embedding batch {i//batch_size + 1} "
                    f"({len(batch_texts)} chunks) for {file_path}"
                )

                # Generate embeddings using async/sync bridge
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.embedding_client.generate_embeddings(
                        batch_texts, input_type="document"  # Code chunks are documents
                    ),
                )

                all_embeddings.extend(embeddings)

                # Log batch statistics
                self._log_batch_stats(
                    batch_chunks, i // batch_size + 1, len(embeddings)
                )

            logger.info(
                f"Successfully generated {len(all_embeddings)} embeddings for {file_path}",
                extra={
                    "structured_data": {
                        "project_name": project_name,
                        "file_path": str(file_path),
                        "embedding_count": len(all_embeddings),
                    }
                },
            )

            return all_embeddings

        except Exception as e:
            logger.error(
                f"Failed to generate embeddings for {file_path}: {e}",
                extra={
                    "structured_data": {
                        "project_name": project_name,
                        "file_path": str(file_path),
                        "chunk_count": len(chunks),
                        "error": str(e),
                    }
                },
                exc_info=True,
            )
            raise

    def _prepare_chunk_texts(self, chunks: List[CodeChunk]) -> List[str]:
        """
        Prepare text content from chunks with context enhancement.

        Args:
            chunks: List of code chunks

        Returns:
            List of prepared text strings ready for embedding
        """
        texts = []
        for chunk in chunks:
            # Include chunk context for better embeddings
            text_content = chunk.content
            if chunk.name:
                # Prefix with chunk name for context
                text_content = f"# {chunk.name}\n{chunk.content}"
            texts.append(text_content)
        return texts

    def _log_batch_stats(
        self, batch_chunks: List[CodeChunk], batch_num: int, embedding_count: int
    ) -> None:
        """Log statistics for a processed batch."""
        chunk_types = {}
        redacted_count = 0

        for chunk in batch_chunks:
            chunk_type = chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            if chunk.redacted:
                redacted_count += 1

        logger.debug(
            f"Batch {batch_num} complete: "
            f"{embedding_count} embeddings generated, "
            f"chunk types: {chunk_types}, "
            f"redacted: {redacted_count}"
        )

    async def generate_embeddings_for_multiple_files(
        self, file_chunks: Dict[str, List[CodeChunk]], project_name: str
    ) -> Dict[str, List[List[float]]]:
        """
        Generate embeddings for chunks from multiple files in a single batch operation.

        Args:
            file_chunks: Dictionary mapping file paths to their code chunks
            project_name: Name of the project (for logging)

        Returns:
            Dictionary mapping file paths to their corresponding embeddings

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not file_chunks:
            logger.debug("No file chunks provided for batch processing")
            return {}

        total_chunks = sum(len(chunks) for chunks in file_chunks.values())
        logger.info(
            f"Generating batch embeddings for {len(file_chunks)} files "
            f"({total_chunks} total chunks)",
            extra={
                "structured_data": {
                    "project_name": project_name,
                    "file_count": len(file_chunks),
                    "chunk_count": total_chunks,
                    "embedding_model": self.config.embedding_model,
                }
            },
        )

        try:
            # Build batches from the start based on actual token counts and text limits
            batches = await self._build_token_aware_batches(file_chunks, project_name)

            if not batches:
                logger.debug("No valid chunks found after text preparation")
                return {}

            # Process each batch
            logger.info(
                f"Processing {len(batches)} token-aware batches for project {project_name}"
            )
            all_file_embeddings = {}

            for i, (batch_texts, batch_boundaries) in enumerate(batches):
                logger.debug(
                    f"Processing batch {i + 1}/{len(batches)}: {len(batch_texts)} texts"
                )

                # Generate embeddings for this batch
                batch_file_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda texts=batch_texts, boundaries=batch_boundaries: self.embedding_client.generate_embeddings_batch(
                        all_texts=texts,
                        file_boundaries=boundaries,
                        input_type="document",
                    ),
                )

                # Merge results
                all_file_embeddings.update(batch_file_embeddings)

            file_embeddings = all_file_embeddings

            # Log batch statistics
            self._log_batch_embedding_stats(file_chunks, file_embeddings)

            logger.info(
                f"Successfully generated batch embeddings for {len(file_embeddings)} files",
                extra={
                    "structured_data": {
                        "project_name": project_name,
                        "files_processed": len(file_embeddings),
                        "total_embeddings": sum(
                            len(embs) for embs in file_embeddings.values()
                        ),
                    }
                },
            )

            return file_embeddings

        except Exception as e:
            logger.error(
                f"Failed to generate batch embeddings: {e}",
                extra={
                    "structured_data": {
                        "project_name": project_name,
                        "file_count": len(file_chunks),
                        "chunk_count": total_chunks,
                        "error": str(e),
                    }
                },
                exc_info=True,
            )
            raise

    async def _build_token_aware_batches(
        self, file_chunks: Dict[str, List[CodeChunk]], project_name: str
    ) -> List[Tuple[List[str], List[Tuple[str, int, int]]]]:
        """
        Build batches from file chunks respecting both token and text count limits.

        Args:
            file_chunks: Dictionary mapping file paths to their code chunks
            project_name: Name of the project (for logging)

        Returns:
            List of tuples: (batch_texts, batch_file_boundaries)
        """
        batches = []  # List of (batch_texts, batch_file_boundaries)
        current_batch_texts = []
        current_batch_boundaries = []
        current_batch_tokens = 0
        batch_idx = 0

        for file_path, chunks in file_chunks.items():
            if not chunks:
                continue

            # Prepare texts for this file
            file_texts = self._prepare_chunk_texts(chunks)
            if not file_texts:
                continue

            # Count tokens for this file using accurate Voyage API
            file_tokens = await asyncio.get_event_loop().run_in_executor(
                None, lambda texts=file_texts: self.embedding_client.count_tokens(texts)
            )

            logger.debug(
                f"File {file_path}: {len(file_texts)} texts, {file_tokens} tokens"
            )

            # If adding this file would exceed token limit OR text count limit, finalize current batch
            if (
                current_batch_tokens + file_tokens
                > self.config.voyage_max_tokens_per_batch
                or len(current_batch_texts) + len(file_texts)
                > self.config.voyage_batch_size_limit
            ) and current_batch_texts:

                # Determine which limit was exceeded for logging
                token_exceeded = (
                    current_batch_tokens + file_tokens
                    > self.config.voyage_max_tokens_per_batch
                )
                count_exceeded = (
                    len(current_batch_texts) + len(file_texts)
                    > self.config.voyage_batch_size_limit
                )

                logger.info(
                    f"Finalizing batch {len(batches) + 1}: {len(current_batch_texts)} texts, "
                    f"{current_batch_tokens} tokens (limit exceeded: "
                    f"tokens={token_exceeded}, count={count_exceeded})"
                )

                batches.append((current_batch_texts, current_batch_boundaries))

                # Start new batch
                current_batch_texts = []
                current_batch_boundaries = []
                current_batch_tokens = 0
                batch_idx = 0

            # Add this file to current batch
            start_idx = batch_idx
            end_idx = batch_idx + len(file_texts)
            current_batch_texts.extend(file_texts)
            current_batch_boundaries.append((file_path, start_idx, end_idx))
            current_batch_tokens += file_tokens
            batch_idx = end_idx

        # Add final batch if it has content
        if current_batch_texts:
            logger.info(
                f"Finalizing final batch {len(batches) + 1}: {len(current_batch_texts)} texts, "
                f"{current_batch_tokens} tokens"
            )
            batches.append((current_batch_texts, current_batch_boundaries))

        return batches

    def _log_batch_embedding_stats(
        self,
        file_chunks: Dict[str, List[CodeChunk]],
        file_embeddings: Dict[str, List[List[float]]],
    ) -> None:
        """Log statistics for batch embedding processing."""
        total_chunks = 0
        chunk_types = {}
        redacted_count = 0
        languages = set()

        for file_path, chunks in file_chunks.items():
            total_chunks += len(chunks)
            for chunk in chunks:
                chunk_type = chunk.chunk_type.value
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                if chunk.redacted:
                    redacted_count += 1
                languages.add(chunk.language)

        total_embeddings = sum(len(embs) for embs in file_embeddings.values())

        logger.debug(
            f"Batch embedding complete: "
            f"{len(file_chunks)} files, "
            f"{total_chunks} chunks, "
            f"{total_embeddings} embeddings generated, "
            f"chunk types: {chunk_types}, "
            f"languages: {sorted(languages)}, "
            f"redacted: {redacted_count}"
        )
