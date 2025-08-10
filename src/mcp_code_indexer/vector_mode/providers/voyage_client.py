"""
Voyage AI client for embedding generation.

Provides integration with Voyage AI's embedding API for generating
high-quality code embeddings using the voyage-code-2 model.

Now uses the official voyageai Python SDK for improved reliability and type safety.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import voyageai

from .base_provider import ProviderError
from ..config import VectorConfig

logger = logging.getLogger(__name__)

class VoyageClient:
    """Client for Voyage AI embedding generation using official SDK."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "voyage-code-2",
        **kwargs
    ):
        self.api_key = api_key
        self.model = model
        self._embedding_dimension: Optional[int] = None
        
        # Initialize official Voyage AI client
        self.client = voyageai.Client(api_key=api_key)
        logger.info(f"Initialized Voyage AI client with model {model}")
    
    def health_check(self) -> bool:
        """Check if Voyage AI service is healthy."""
        try:
            # Make a small test request using official SDK
            result = self.client.embed(["test"], model=self.model, input_type="query")
            return len(result.embeddings) > 0
        except Exception as e:
            logger.warning(f"Voyage AI health check failed: {e}")
            return False
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count - Voyage AI handles exact tokenization."""
        # Voyage AI uses proprietary tokenizer - this is just for batching estimates
        # Rough approximation: 4 characters per token (conservative estimate)
        return len(text) // 4
    
    def _batch_texts_by_tokens(
        self,
        texts: List[str],
        max_tokens_per_batch: int = 120000  # Leave buffer under 128k limit
    ) -> List[List[str]]:
        """Batch texts to stay under token limits."""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            text_tokens = self._count_tokens(text)
            
            # If single text exceeds limit, truncate it (let Voyage API handle exact truncation)
            if text_tokens > max_tokens_per_batch:
                # Rough character-based truncation - Voyage API will handle exact tokenization
                target_chars = (max_tokens_per_batch - 100) * 4  # Conservative estimate
                text = text[:target_chars]
                text_tokens = self._count_tokens(text)
                
                logger.warning(f"Pre-truncated text to ~{text_tokens} tokens (Voyage API will handle exact tokenization)")
            
            # Check if adding this text would exceed the batch limit
            if current_tokens + text_tokens > max_tokens_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def generate_embeddings(
        self,
        texts: List[str],
        input_type: str = "document",
        truncation: bool = True,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using official SDK.
        
        Args:
            texts: List of texts to embed
            input_type: Type of input ("document" or "query")
            truncation: Whether to enable truncation
            **kwargs: Additional arguments
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model}")
        
        # Batch texts to stay under token limits
        batches = self._batch_texts_by_tokens(texts)
        all_embeddings = []
        
        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)} with {len(batch)} texts")
            
            try:
                # Use official SDK
                result = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type=input_type,
                    truncation=truncation
                )
                
                # Extract embeddings from SDK response
                batch_embeddings = result.embeddings
                all_embeddings.extend(batch_embeddings)
                
                # Log usage information if available
                if hasattr(result, 'usage') and result.usage:
                    logger.debug(
                        f"Batch {i+1} usage: {result.usage.total_tokens} tokens"
                    )
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i+1}: {e}")
                raise ProviderError(f"Embedding generation failed: {e}")
        
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self._embedding_dimension is not None:
            return self._embedding_dimension
        
        # Generate a test embedding to determine dimension
        try:
            test_embeddings = self.generate_embeddings(["test"], input_type="query")
            if test_embeddings:
                self._embedding_dimension = len(test_embeddings[0])
                logger.info(f"Detected embedding dimension: {self._embedding_dimension}")
                return self._embedding_dimension
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
        
        # Default dimensions for known Voyage models (as of 2024)
        # Note: These may change - verify with Voyage AI documentation
        model_dimensions = {
            "voyage-code-2": 1536,    # Code-optimized model
            "voyage-2": 1024,         # General purpose
            "voyage-large-2": 1536,   # Large general purpose
            "voyage-3": 1024,         # Newer general purpose (if available)
        }
        
        self._embedding_dimension = model_dimensions.get(self.model, 1536)
        logger.info(f"Using default dimension for {self.model}: {self._embedding_dimension}")
        return self._embedding_dimension
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate a single embedding for a search query."""
        embeddings = await self.generate_embeddings([query], input_type="query")
        return embeddings[0] if embeddings else []
    
    async def estimate_cost(self, texts: List[str]) -> Dict[str, Any]:
        """Estimate the cost of embedding generation."""
        total_tokens = sum(self._count_tokens(text) for text in texts)
        
        # Voyage AI pricing (approximate, may change)
        cost_per_1k_tokens = 0.00013  # voyage-code-2 pricing
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        return {
            "total_tokens": total_tokens,
            "total_texts": len(texts),
            "estimated_cost_usd": round(estimated_cost, 6),
            "model": self.model,
        }

def create_voyage_client(config: VectorConfig) -> VoyageClient:
    """Create a Voyage client from configuration using official SDK."""
    if not config.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY is required for embedding generation")
    
    return VoyageClient(
        api_key=config.voyage_api_key,
        model=config.embedding_model,
        timeout=30.0,
        max_retries=3,
    )
