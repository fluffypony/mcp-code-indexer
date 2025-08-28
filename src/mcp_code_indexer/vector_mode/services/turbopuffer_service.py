"""
TurbopufferService for vector storage and retrieval.

Provides a clean abstraction layer between code chunks/embeddings and the
TurbopufferClient, handling namespace management, vector formatting,
and error handling.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..chunking.ast_chunker import CodeChunk
from ..providers.turbopuffer_client import TurbopufferClient
from ..config import VectorConfig

logger = logging.getLogger(__name__)


class TurbopufferService:
    """
    Service for storing code embeddings in Turbopuffer vector database.

    Handles namespace management, vector formatting, and provides a clean
    abstraction layer between the daemon orchestration and vector storage.
    """

    def __init__(self, turbopuffer_client: TurbopufferClient, config: VectorConfig):
        """Initialize TurbopufferService with client and configuration."""
        self.turbopuffer_client = turbopuffer_client
        self.config = config
        self._namespace_cache: Dict[str, bool] = {}  # Cache for namespace existence

    async def store_embeddings(
        self,
        embeddings: List[List[float]],
        chunks: List[CodeChunk],
        project_name: str,
        file_path: str,
    ) -> None:
        """
        Store embeddings for code chunks in Turbopuffer.

        Args:
            embeddings: List of embedding vectors
            chunks: List of code chunks corresponding to embeddings
            project_name: Name of the project
            file_path: Path to the source file

        Raises:
            ValueError: If embeddings and chunks count mismatch
            RuntimeError: If Turbopuffer operations fail
        """
        if not embeddings:
            logger.debug(f"No embeddings to store for {file_path}")
            return

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings and chunks count mismatch: "
                f"{len(embeddings)} embeddings vs {len(chunks)} chunks"
            )

        try:
            # Ensure namespace exists
            namespace = await self._ensure_namespace_exists(project_name)
            
            # Format vectors for storage
            vectors = self._format_vectors_for_storage(
                embeddings, chunks, project_name, file_path
            )
            
            # Store in Turbopuffer
            result = self.turbopuffer_client.upsert_vectors(vectors, namespace)
            
            logger.info(
                f"Stored {result['upserted']} vectors for {file_path} "
                f"in namespace {namespace}"
            )

        except Exception as e:
            logger.error(f"Failed to store embeddings for {file_path}: {e}")
            raise RuntimeError(f"Vector storage failed: {e}")

    async def _ensure_namespace_exists(self, project_name: str) -> str:
        """
        Ensure the namespace for a project exists, creating it if necessary.

        Args:
            project_name: Name of the project

        Returns:
            The namespace name

        Raises:
            RuntimeError: If namespace operations fail
        """
        namespace = self.turbopuffer_client.get_namespace_for_project(project_name)
        
        # Check cache first
        if namespace in self._namespace_cache:
            return namespace

        try:
            # List existing namespaces
            existing_namespaces = self.turbopuffer_client.list_namespaces()
            
            if namespace not in existing_namespaces:
                # Create namespace with embedding dimension
                embedding_dimension = self._get_embedding_dimension()
                self.turbopuffer_client.create_namespace(
                    namespace=namespace, 
                    dimension=embedding_dimension
                )
                logger.info(f"Created namespace '{namespace}' for project '{project_name}'")
            
            # Cache the result
            self._namespace_cache[namespace] = True
            return namespace

        except Exception as e:
            logger.error(f"Failed to ensure namespace {namespace}: {e}")
            raise RuntimeError(f"Namespace operation failed: {e}")

    def _format_vectors_for_storage(
        self,
        embeddings: List[List[float]],
        chunks: List[CodeChunk], 
        project_name: str,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Format embeddings and chunks into vectors suitable for Turbopuffer storage.

        Args:
            embeddings: List of embedding vectors
            chunks: List of corresponding code chunks
            project_name: Name of the project
            file_path: Path to the source file

        Returns:
            List of formatted vector dictionaries
        """
        vectors = []
        
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            # Generate unique vector ID
            vector_id = self.turbopuffer_client.generate_vector_id(project_name, i)
            
            # Prepare metadata
            metadata = {
                "project_id": project_name,
                "project_name": project_name,
                "file_path": file_path,
                "chunk_type": chunk.chunk_type.value,
                "chunk_name": chunk.name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content_hash": chunk.content_hash,
                "language": chunk.language,
                "redacted": chunk.redacted,
                "chunk_index": i,
                "imports": ",".join(chunk.imports) if chunk.imports else "",
            }
            
            # Add custom metadata if present
            if chunk.metadata:
                metadata.update(chunk.metadata)

            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": metadata,
            }
            vectors.append(vector)

        logger.debug(f"Formatted {len(vectors)} vectors for storage")
        return vectors

    def _get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension based on the configured model.

        Returns:
            The dimension size for the embedding model
        """
        # Map embedding models to their dimensions
        model_dimensions = {
            "voyage-code-2": 1536,
            "voyage-2": 1024,
            "voyage-large-2": 1536,
            "voyage-law-2": 1024,
        }
        
        dimension = model_dimensions.get(self.config.embedding_model, 1536)
        logger.debug(f"Using dimension {dimension} for model {self.config.embedding_model}")
        return dimension

    async def delete_vectors_for_file(
        self, project_name: str, file_path: str
    ) -> None:
        """
        Delete all vectors associated with a specific file.

        Args:
            project_name: Name of the project
            file_path: Path to the source file

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            namespace = self.turbopuffer_client.get_namespace_for_project(project_name)
            
            # Search for vectors with matching file_path metadata
            # This is a simplified approach - in practice, you might need
            # to track vector IDs separately for efficient deletion
            logger.info(f"Deleting vectors for file {file_path} in namespace {namespace}")
            
            # Note: Actual implementation would require searching by metadata
            # and then deleting found vector IDs. This is left as TODO since
            # it depends on specific Turbopuffer query capabilities.
            
        except Exception as e:
            logger.error(f"Failed to delete vectors for {file_path}: {e}")
            raise RuntimeError(f"Vector deletion failed: {e}")

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        project_name: str,
        top_k: int = 10,
        chunk_type: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar code chunks using embedding similarity.

        Args:
            query_embedding: The query embedding vector
            project_name: Name of the project to search in
            top_k: Number of results to return
            chunk_type: Optional filter by chunk type
            file_path: Optional filter by file path

        Returns:
            List of similar chunks with metadata and similarity scores

        Raises:
            RuntimeError: If search fails
        """
        try:
            results = self.turbopuffer_client.search_with_metadata_filter(
                query_vector=query_embedding,
                project_id=project_name,
                chunk_type=chunk_type,
                file_path=file_path,
                top_k=top_k,
            )
            
            logger.debug(f"Found {len(results)} similar chunks in {project_name}")
            return results

        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            raise RuntimeError(f"Vector search failed: {e}")
