"""
Embedding Service for converting code chunks to vector embeddings.

Provides a clean abstraction layer between chunked code and embedding providers,
handling text preparation, batching, and provider communication.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

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
