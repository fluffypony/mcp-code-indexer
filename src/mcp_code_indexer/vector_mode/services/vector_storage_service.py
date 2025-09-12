"""
VectorStorageService for vector storage and retrieval.

Provides a clean abstraction layer between code chunks/embeddings and the
vector storage backend, handling namespace management, vector formatting,
and error handling.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from mcp_code_indexer.vector_mode.monitoring.utils import _write_debug_log

from turbopuffer.types import Row

from ..chunking.ast_chunker import CodeChunk
from ..providers.turbopuffer_client import TurbopufferClient
from ..config import VectorConfig

logger = logging.getLogger(__name__)


class VectorStorageService:
    """
    Service for storing code embeddings in vector database.

    Handles namespace management, vector formatting, and provides a clean
    abstraction layer between the daemon orchestration and vector storage.
    """

    def __init__(
        self,
        turbopuffer_client: TurbopufferClient,
        embedding_dimension: int,
        config: VectorConfig,
    ):
        """Initialize VectorStorageService with client, embedding dimension, and configuration."""
        self.turbopuffer_client = turbopuffer_client
        self.embedding_dimension = embedding_dimension
        self.config = config

        # Validate API access immediately during initialization
        self.turbopuffer_client.validate_api_access()
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
            # Get namespace name (will be created implicitly by upsert_vectors)
            namespace = self.turbopuffer_client.get_namespace_for_project(project_name)

            # Clear existing vectors for this file to prevent redundant entries
            logger.info(
                f"Clearing existing vectors for file {file_path} before upserting new ones"
            )
            try:
                delete_result = self.turbopuffer_client.delete_vectors_for_file(
                    namespace, file_path
                )
                logger.info(
                    f"Cleared {delete_result['deleted']} existing vectors for {file_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to clear existing vectors for {file_path}: {e}")
                # Continue with upsert even if deletion fails - better to have duplicates than no data

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

    async def store_embeddings_batch(
        self,
        file_embeddings: Dict[str, List[List[float]]],
        file_chunks: Dict[str, List[CodeChunk]],
        project_name: str,
    ) -> None:
        """
        Store embeddings for multiple files in a single batch operation.

        Args:
            file_embeddings: Dictionary mapping file paths to their embeddings
            file_chunks: Dictionary mapping file paths to their code chunks
            project_name: Name of the project

        Raises:
            ValueError: If embeddings and chunks count mismatch for any file
            RuntimeError: If Turbopuffer operations fail
        """
        if not file_embeddings:
            logger.debug("No embeddings to store in batch")
            return

        if set(file_embeddings.keys()) != set(file_chunks.keys()):
            raise ValueError("file_embeddings and file_chunks must have matching keys")

        # Validate embeddings and chunks counts for each file
        for file_path in file_embeddings.keys():
            embeddings = file_embeddings[file_path]
            chunks = file_chunks[file_path]
            if len(embeddings) != len(chunks):
                raise ValueError(
                    f"Embeddings and chunks count mismatch for {file_path}: "
                    f"{len(embeddings)} embeddings vs {len(chunks)} chunks"
                )

        total_vectors = sum(len(embs) for embs in file_embeddings.values())
        logger.info(
            f"Batch storing {total_vectors} vectors from {len(file_embeddings)} files "
            f"for project {project_name}"
        )

        try:
            # Get namespace name (will be created implicitly by batch upsert)
            namespace = self.turbopuffer_client.get_namespace_for_project(project_name)

            # Delete existing vectors for all files in a single batch operation
            files_to_clear = list(file_embeddings.keys())
            try:
                total_cleared = await self.batch_delete_vectors_for_files(
                    project_name, files_to_clear
                )
                if total_cleared > 0:
                    logger.info(
                        f"Cleared {total_cleared} existing vectors from {len(files_to_clear)} files"
                    )
            except Exception as e:
                logger.warning(f"Failed to batch clear existing vectors: {e}")
                # Continue with batch upsert even if deletion fails

            # Format all vectors for batch storage
            all_vectors = []
            for file_path in file_embeddings.keys():
                embeddings = file_embeddings[file_path]
                chunks = file_chunks[file_path]

                file_vectors = self._format_vectors_for_storage(
                    embeddings, chunks, project_name, file_path
                )
                all_vectors.extend(file_vectors)

            # Perform batch upsert
            result = self.turbopuffer_client.upsert_vectors_batch(
                all_vectors, namespace
            )

            logger.info(
                f"Batch stored {result['upserted']} vectors from {len(file_embeddings)} files "
                f"in namespace {namespace}"
            )

        except Exception as e:
            logger.error(f"Failed to batch store embeddings: {e}")
            raise RuntimeError(f"Batch vector storage failed: {e}")

    async def _ensure_namespace_exists(self, project_name: str) -> str | None:
        """
        Ensure the namespace for a project exists.

        Args:
            project_name: Name of the project

        Returns:
            The namespace name if it exists, None otherwise

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
                # Namespace doesn't exist and will be created implicitly on first write
                logger.info(
                    f"Namespace '{namespace}' for project '{project_name}' does not exist"
                )
                return None

            # Cache the result for existing namespace
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
        from datetime import datetime

        vectors = []

        # Get file modification time once for all chunks
        try:
            file_path_obj = Path(file_path)
            mtime_unix = file_path_obj.stat().st_mtime
            mtime_iso = datetime.fromtimestamp(mtime_unix).isoformat()
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Failed to get mtime for {file_path}: {e}")
            mtime_unix = 0.0
            mtime_iso = datetime.fromtimestamp(0.0).isoformat()

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
                "file_mtime": str(mtime_unix),
                "file_mtime_iso": mtime_iso,
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

    async def delete_vectors_for_file(self, project_name: str, file_path: str) -> None:
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

            logger.info(
                f"Deleting vectors for file {file_path} in namespace {namespace}"
            )

            # Use the TurbopufferClient method to delete by file_path filter
            result = self.turbopuffer_client.delete_vectors_for_file(
                namespace, file_path
            )

            logger.info(
                f"Successfully deleted {result['deleted']} vectors for file {file_path}"
            )

        except Exception as e:
            logger.error(f"Failed to delete vectors for {file_path}: {e}")
            raise RuntimeError(f"Vector deletion failed: {e}")

    async def batch_delete_vectors_for_files(
        self, project_name: str, file_paths: list[str]
    ) -> int:
        """
        Delete all vectors associated with multiple files in a single batch operation.

        Args:
            project_name: Name of the project
            file_paths: List of file paths to delete vectors for

        Returns:
            Number of vectors deleted

        Raises:
            RuntimeError: If deletion fails
        """
        if not file_paths:
            return 0

        try:
            namespace = self.turbopuffer_client.get_namespace_for_project(project_name)

            logger.info(
                f"Batch deleting vectors for {len(file_paths)} files in namespace {namespace}"
            )

            # Create dummy vector with correct dimensions for search
            dummy_vector = [0.0] * self.embedding_dimension

            # Find all vectors for the specified files
            rows = self.turbopuffer_client.search_vectors(
                query_vector=dummy_vector,
                top_k=1200,  # Set high enough to catch all chunks for multiple files
                namespace=namespace,
                filters=(
                    "And",
                    [
                        ("project_id", "Eq", project_name),
                        ("file_path", "In", file_paths),
                    ],
                ),
            )

            if not rows:
                logger.info(
                    f"No vectors found for {len(file_paths)} files in namespace {namespace}"
                )
                return 0

            # Extract vector IDs to delete
            ids_to_delete = [row.id for row in rows]
            logger.info(
                f"Found {len(ids_to_delete)} vectors to delete for {len(file_paths)} files"
            )

            # Delete vectors by ID in batch
            delete_result = self.turbopuffer_client.delete_vectors(
                ids_to_delete, namespace
            )

            deleted_count = delete_result["deleted"]
            logger.info(
                f"Batch deletion completed: removed {deleted_count} vectors "
                f"for {len(file_paths)} files from namespace {namespace}"
            )

            return deleted_count

        except Exception as e:
            logger.error(
                f"Failed to batch delete vectors for {len(file_paths)} files: {e}"
            )
            raise RuntimeError(f"Batch vector deletion failed: {e}")

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        project_name: str,
        top_k: int = 10,
        chunk_type: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> List[Row] | None:
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

    async def get_file_metadata(
        self, project_name: str, file_paths: list[str] | None = None
    ) -> dict[str, float]:
        """
        Retrieve file modification times from stored vector metadata.

        Args:
            project_name: Name of the project
            file_paths: Optional list of specific file paths to query. If None, queries all files.

        Returns:
            Dictionary mapping file_path to mtime (Unix timestamp)

        Raises:
            RuntimeError: If metadata query fails
        """
        try:
            namespace = await self._ensure_namespace_exists(project_name)

            # If namespace doesn't exist, return empty metadata
            if namespace is None:
                logger.debug(
                    f"No namespace found for project {project_name}, returning empty metadata"
                )
                return {}

            # Create dummy vector with correct dimensions
            dummy_vector = [0.0] * self.embedding_dimension

            if file_paths is None or len(file_paths) == 0:
                # Query all files for the project
                rows = self.turbopuffer_client.search_vectors(
                    query_vector=dummy_vector,  # Dummy vector since we only want metadata
                    top_k=1200,  # High limit to get all files (warning: it still can limit results)
                    namespace=namespace,
                    filters=("project_id", "Eq", project_name),
                )
            else:
                # Query specific files
                rows = self.turbopuffer_client.search_vectors(
                    query_vector=dummy_vector,
                    top_k=1200,
                    namespace=namespace,
                    filters=(
                        "And",
                        [
                            ("project_id", "Eq", project_name),
                            ("file_path", "In", file_paths),
                        ],
                    ),
                )

            if not rows:
                logger.debug(
                    f"No rows found for project {project_name}, returning empty metadata"
                )
                return {}

            # Extract file metadata, keeping only the most recent mtime per file
            file_metadata = {}
            for row in rows:
                # Check if row has file_path and file_mtime attributes
                if hasattr(row, "file_path") and hasattr(row, "file_mtime"):
                    file_path = row.file_path
                    mtime = float(row.file_mtime)

                    # Keep the most recent mtime for each file
                    if (
                        file_path not in file_metadata
                        or mtime > file_metadata[file_path]
                    ):
                        file_metadata[file_path] = mtime

            logger.debug(
                f"Retrieved metadata for {len(file_metadata)} files from {project_name}"
            )
            return file_metadata

        except Exception as e:
            logger.error(f"Failed to get file metadata for {project_name}: {e}")
            return {}
