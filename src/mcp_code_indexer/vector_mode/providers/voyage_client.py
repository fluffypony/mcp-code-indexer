"""
Voyage AI client for embedding generation using official SDK.

Provides clean integration with Voyage AI's embedding API for generating
high-quality code embeddings using the voyage-code-2 model.
"""

import logging
from typing import List, Dict, Any, Tuple
import voyageai

from ..config import VectorConfig, DEFAULT_EMBEDDING_MODEL
from ..const import MODEL_DIMENSIONS

logger = logging.getLogger(__name__)


class VoyageClient:
    """Clean Voyage AI client using official SDK."""

    def __init__(self, api_key: str, model: str = DEFAULT_EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self._embedding_dimension: int | None = None

        # Initialize official Voyage AI client
        self.client = voyageai.Client(api_key=api_key)
        logger.info(f"Initialized Voyage AI client with model {model}")

    def health_check(self) -> bool:
        """Check if Voyage AI service is healthy."""
        try:
            result = self.client.embed(["test"], model=self.model, input_type="query")
            return len(result.embeddings) > 0
        except Exception as e:
            logger.warning(f"Voyage AI health check failed: {e}")
            return False

    def validate_api_access(self) -> None:
        """
        Validate API key and access to Voyage AI service.

        Raises:
            RuntimeError: If API access validation fails with specific error details
        """
        logger.info("Validating Voyage AI API access...")
        try:
            result = self.client.embed(["test"], model=self.model, input_type="query")
            if not result or not result.embeddings:
                raise RuntimeError("Voyage AI API returned empty response")
            logger.debug("Voyage AI API access validated successfully")
        except Exception as e:
            error_msg = str(e).lower()

            if (
                "401" in error_msg
                or "unauthorized" in error_msg
                or "api key" in error_msg
            ):
                raise RuntimeError(
                    f"Voyage AI API authentication failed: Invalid or expired API key. "
                    f"Please check your VOYAGE_API_KEY. Error: {e}"
                )
            elif "403" in error_msg or "forbidden" in error_msg:
                raise RuntimeError(
                    f"Voyage AI API access denied: API key lacks required permissions. Error: {e}"
                )
            elif "429" in error_msg or "rate limit" in error_msg:
                raise RuntimeError(
                    f"Voyage AI API rate limit exceeded: Too many requests. Error: {e}"
                )
            elif "quota" in error_msg or "usage" in error_msg:
                raise RuntimeError(
                    f"Voyage AI API quota exceeded: Usage limit reached. Error: {e}"
                )
            elif "5" in error_msg and ("error" in error_msg or "server" in error_msg):
                raise RuntimeError(
                    f"Voyage AI service unavailable: Server error. Error: {e}"
                )
            else:
                raise RuntimeError(f"Voyage AI API access validation failed: {e}")

        logger.info("Voyage AI API access validated successfully")

    def generate_embeddings(
        self, texts: List[str], input_type: str = "document", **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for texts using official SDK."""
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model}")

        try:
            result = self.client.embed(
                texts=texts, model=self.model, input_type=input_type, truncation=True
            )

            # Log usage if available
            if hasattr(result, "usage") and result.usage:
                logger.debug(f"Token usage: {result.usage.total_tokens}")

            logger.info(f"Successfully generated {len(result.embeddings)} embeddings")
            return result.embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self._embedding_dimension is not None:
            return self._embedding_dimension

        # Generate a test embedding to determine dimension
        try:
            test_embeddings = self.generate_embeddings(["test"], input_type="query")
            if test_embeddings:
                self._embedding_dimension = len(test_embeddings[0])
                logger.info(
                    f"Detected embedding dimension: {self._embedding_dimension}"
                )
                return self._embedding_dimension
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")

        self._embedding_dimension = MODEL_DIMENSIONS[self.model]
        logger.info(f"Using default embedding dimension: {self._embedding_dimension}")
        return self._embedding_dimension

    def estimate_cost(self, texts: List[str]) -> Dict[str, Any]:
        """Estimate the cost of embedding generation."""
        # Rough token estimation (4 chars per token)
        total_tokens = sum(len(text) // 4 for text in texts)

        # Voyage AI pricing (approximate, may change)
        cost_per_1k_tokens = 0.00013  # voyage-code-2 pricing
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

        return {
            "total_tokens": total_tokens,
            "total_texts": len(texts),
            "estimated_cost_usd": round(estimated_cost, 6),
            "model": self.model,
        }

    def generate_embeddings_batch(
        self,
        all_texts: List[str],
        file_boundaries: List[Tuple[str, int, int]],
        input_type: str = "document",
        max_tokens_per_batch: int = 120000,
        **kwargs,
    ) -> Dict[str, List[List[float]]]:
        """
        Generate embeddings for texts from multiple files with automatic token-based batching.

        Args:
            all_texts: Flattened list of all text chunks from all files
            file_boundaries: List of (file_path, start_idx, end_idx) tuples indicating
                           which embeddings belong to which file
            input_type: Type of input for embedding generation
            max_tokens_per_batch: Maximum tokens allowed per batch (default: 120000)
            **kwargs: Additional arguments for embedding generation

        Returns:
            Dictionary mapping file paths to their corresponding embeddings

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not all_texts:
            return {}

        if not file_boundaries:
            raise ValueError(
                "file_boundaries cannot be empty when all_texts is provided"
            )

        logger.info(
            f"Generating batch embeddings for {len(all_texts)} texts from {len(file_boundaries)} files using {self.model}"
        )

        try:
            # Check if we need to split into sub-batches based on token count
            cost_estimate = self.estimate_cost(all_texts)
            total_tokens = cost_estimate["total_tokens"]

            if total_tokens <= max_tokens_per_batch:
                # Single batch - generate embeddings for all texts at once
                logger.debug(f"Processing single batch with {total_tokens} tokens")
                all_embeddings = self.generate_embeddings(
                    all_texts, input_type=input_type, **kwargs
                )
            else:
                # Multiple batches needed - split based on token count
                logger.info(
                    f"Total tokens ({total_tokens}) exceeds limit ({max_tokens_per_batch}). "
                    f"Splitting into token-based sub-batches."
                )
                all_embeddings = self._generate_embeddings_with_token_batching(
                    all_texts, max_tokens_per_batch, input_type, **kwargs
                )

            # Group embeddings by file using boundaries
            file_embeddings = {}
            for file_path, start_idx, end_idx in file_boundaries:
                file_embeddings[file_path] = all_embeddings[start_idx:end_idx]

            logger.info(
                f"Successfully generated batch embeddings for {len(file_boundaries)} files "
                f"({len(all_embeddings)} total embeddings)"
            )

            return file_embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}")

    def _generate_embeddings_with_token_batching(
        self,
        all_texts: List[str],
        max_tokens_per_batch: int,
        input_type: str = "document",
        **kwargs,
    ) -> List[List[float]]:
        """
        Generate embeddings with token-based batching to respect API limits.

        Args:
            all_texts: List of all text chunks
            max_tokens_per_batch: Maximum tokens allowed per batch
            input_type: Type of input for embedding generation
            **kwargs: Additional arguments for embedding generation

        Returns:
            List of embeddings for all texts
        """
        all_embeddings = []
        current_batch = []
        current_tokens = 0
        batch_count = 0

        for i, text in enumerate(all_texts):
            # Estimate tokens for this text (rough estimate: 4 chars per token)
            text_tokens = len(text) // 4

            # If adding this text would exceed the limit, process current batch
            if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
                batch_count += 1
                logger.debug(
                    f"Processing token-based sub-batch {batch_count}: "
                    f"{len(current_batch)} texts, ~{current_tokens} tokens"
                )

                batch_embeddings = self.generate_embeddings(
                    current_batch, input_type=input_type, **kwargs
                )
                all_embeddings.extend(batch_embeddings)

                # Reset for next batch
                current_batch = []
                current_tokens = 0

            # Add current text to batch
            current_batch.append(text)
            current_tokens += text_tokens

        # Process final batch if any texts remaining
        if current_batch:
            batch_count += 1
            logger.debug(
                f"Processing final token-based sub-batch {batch_count}: "
                f"{len(current_batch)} texts, ~{current_tokens} tokens"
            )

            batch_embeddings = self.generate_embeddings(
                current_batch, input_type=input_type, **kwargs
            )
            all_embeddings.extend(batch_embeddings)

        logger.info(
            f"Token-based batching completed: {len(all_embeddings)} total embeddings "
            f"across {batch_count} sub-batches"
        )

        return all_embeddings


def create_voyage_client(config: VectorConfig) -> VoyageClient:
    """Create a Voyage client from configuration."""
    if not config.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY is required for embedding generation")

    return VoyageClient(
        api_key=config.voyage_api_key,
        model=config.embedding_model,
    )
