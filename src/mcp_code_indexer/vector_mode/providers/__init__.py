"""
External service providers for vector mode.

This package provides integrations with external services including:
- Voyage AI for embedding generation
- Turbopuffer for vector storage and search
"""

from typing import Protocol, List, Dict, Any, Optional
from abc import abstractmethod

class EmbeddingProvider(Protocol):
    """Protocol for embedding generation providers."""
    
    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        input_type: str = "document",
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        ...
    
    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        ...

class VectorStoreProvider(Protocol):
    """Protocol for vector storage providers."""
    
    @abstractmethod
    async def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Store or update vectors in the database."""
        ...
    
    @abstractmethod
    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        ...
    
    @abstractmethod
    async def delete_vectors(
        self,
        vector_ids: List[str],
        namespace: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Delete vectors by ID."""
        ...
    
    @abstractmethod
    async def get_namespace_stats(
        self,
        namespace: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get statistics about a namespace."""
        ...
