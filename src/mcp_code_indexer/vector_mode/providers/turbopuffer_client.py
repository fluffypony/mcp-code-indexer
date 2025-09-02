"""
Turbopuffer client for vector storage and search using official SDK.

Provides clean integration with Turbopuffer's vector database for storing
embeddings and performing similarity searches. Supports configurable
regions for optimal latency and data residency compliance.

Default region: gcp-europe-west3 (Frankfurt)
Configure via TURBOPUFFER_REGION environment variable.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
import turbopuffer

from ..config import VectorConfig

logger = logging.getLogger(__name__)


class TurbopufferClient:
    """Clean Turbopuffer client using official SDK."""

    def __init__(self, api_key: str, region: str = "gcp-europe-west3"):
        self.api_key = api_key
        self.region = region

        # Initialize official TurboPuffer client
        self.client = turbopuffer.Turbopuffer(api_key=api_key, region=region)
        logger.info(f"Initialized TurboPuffer client with region {region}")

    def health_check(self) -> bool:
        """Check if Turbopuffer service is healthy."""
        try:
            self.client.namespaces()
            return True
        except Exception as e:
            logger.warning(f"Turbopuffer health check failed: {e}")
            return False

    def validate_api_access(self) -> None:
        """
        Validate API key and access to Turbopuffer service.

        Raises:
            RuntimeError: If API access validation fails with specific error details
        """
        logger.info("Validating Turbopuffer API access...")
        try:
            self.client.namespaces()
            logger.debug("Turbopuffer API access validated successfully")
        except Exception as e:
            error_msg = str(e).lower()

            if "401" in error_msg or "unauthorized" in error_msg:
                raise RuntimeError(
                    f"Turbopuffer API authentication failed: Invalid or expired API key. "
                    f"Please check your TURBOPUFFER_API_KEY. Error: {e}"
                )
            elif "403" in error_msg or "forbidden" in error_msg:
                raise RuntimeError(
                    f"Turbopuffer API access denied: API key lacks required permissions. Error: {e}"
                )
            elif "429" in error_msg or "rate limit" in error_msg:
                raise RuntimeError(
                    f"Turbopuffer API rate limit exceeded: Too many requests. Error: {e}"
                )
            elif "5" in error_msg and ("error" in error_msg or "server" in error_msg):
                raise RuntimeError(
                    f"Turbopuffer service unavailable: Server error. Error: {e}"
                )
            else:
                raise RuntimeError(f"Turbopuffer API access validation failed: {e}")

    def generate_vector_id(self, project_id: str, chunk_id: int) -> str:
        """Generate a unique vector ID."""
        return f"{project_id}_{chunk_id}_{uuid.uuid4().hex[:8]}"

    def upsert_vectors(
        self, vectors: List[Dict[str, Any]], namespace: str, **kwargs
    ) -> Dict[str, Any]:
        """Store or update vectors in the database."""
        if not vectors:
            return {"upserted": 0}

        logger.info(f"Upserting {len(vectors)} vectors to namespace '{namespace}'")

        # Convert row-based data to columnar format for v0.5+ API
        if not all("id" in vector and "values" in vector for vector in vectors):
            raise ValueError("Each vector must have 'id' and 'values' fields")

        # Build columnar data structure
        data = {
            "id": [str(vector["id"]) for vector in vectors],
            "vector": [vector["values"] for vector in vectors],
        }

        # Add metadata attributes as separate columns
        all_metadata_keys = set()
        for vector in vectors:
            metadata = vector.get("metadata", {})
            all_metadata_keys.update(metadata.keys())

        # Add each metadata attribute as a column
        for key in all_metadata_keys:
            data[key] = [vector.get("metadata", {}).get(key) for vector in vectors]

        try:
            # Get namespace object and use write() with upsert_columns
            ns = self.client.namespace(namespace)
            response = ns.write(
                upsert_columns=data,
                distance_metric="cosine_distance",  # Default metric TODO: which one to use?
            )

            # Log actual results from the response
            rows_affected = getattr(response, "rows_affected", len(vectors))
            logger.info(
                f"Upsert operation completed: requested {len(vectors)} vectors, "
                f"actually affected {rows_affected} rows"
            )

            return {"upserted": rows_affected}

        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise RuntimeError(f"Vector upsert failed: {e}")

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: str = "default",
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        logger.debug(f"Searching {top_k} vectors in namespace '{namespace}'")

        try:
            ns = self.client.namespace(namespace)

            # Convert filters to proper tuple format for v0.5+
            query_filters = None
            if filters:
                # Convert dict filters to turbopuffer filter format
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append((key, "Eq", value))

                if len(filter_conditions) == 1:
                    query_filters = filter_conditions[0]
                else:
                    query_filters = ("And", tuple(filter_conditions))

            results = ns.query(
                rank_by=("vector", "ANN", query_vector),  # Use tuple format for v0.5+
                top_k=top_k,
                filters=query_filters,
                include_attributes=True,
            )

            # Convert results to expected format
            if hasattr(results, "rows") and results.rows:
                formatted_results = []
                for row in results.rows:
                    formatted_results.append(
                        {
                            "id": row.id,
                            "score": getattr(row, "$dist", 0.0),  # Distance as score
                            "metadata": {
                                k: v
                                for k, v in row.__dict__.items()
                                if k not in ["id", "vector", "$dist"]
                            },
                        }
                    )
                logger.debug(f"Found {len(formatted_results)} similar vectors")
                return formatted_results
            else:
                logger.debug("Found 0 similar vectors")
                return []

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RuntimeError(f"Vector search failed: {e}")

    def delete_vectors(
        self, vector_ids: List[str], namespace: str, **kwargs
    ) -> Dict[str, Any]:
        """Delete vectors by ID."""
        if not vector_ids:
            return {"deleted": 0}

        logger.info(f"Deleting {len(vector_ids)} vectors from namespace '{namespace}'")

        try:
            ns = self.client.namespace(namespace)

            # Use the write method with deletes parameter (v0.5+ API)
            response = ns.write(deletes=vector_ids)

            # Log actual results from the response
            rows_affected = getattr(response, "rows_affected", 0)
            logger.info(
                f"Delete operation completed: requested {len(vector_ids)} vectors, "
                f"actually affected {rows_affected} rows"
            )

            return {"deleted": rows_affected}

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise RuntimeError(f"Vector deletion failed: {e}")

    def list_namespaces(self) -> List[str]:
        """List all available namespaces."""
        try:
            namespaces = self.client.namespaces()
            return [ns.id for ns in namespaces.namespaces]

        except Exception as e:
            logger.error(f"Failed to list namespaces: {e}")
            raise RuntimeError(f"Namespace listing failed: {e}")

    def create_namespace(
        self, namespace: str, dimension: int, **kwargs
    ) -> Dict[str, Any]:
        """Create a new namespace - handled implicitly by Turbopuffer."""
        logger.info(
            f"Namespace '{namespace}' will be created automatically on first write "
            f"(dimension: {dimension})"
        )

        # Turbopuffer creates namespaces implicitly when data is first written
        # No explicit creation is needed or supported
        logger.debug("Turbopuffer creates namespaces implicitly on first data write")
        return {"name": namespace, "dimension": dimension, "created_implicitly": True}

    def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """Delete a namespace and all its vectors."""
        logger.warning(f"Deleting namespace '{namespace}' and all its vectors")

        try:
            ns = self.client.namespace(namespace)

            # Use delete_all method to delete the namespace (v0.5+ API)
            response = ns.delete_all()

            # Log actual results from the response
            rows_affected = getattr(response, "rows_affected", 0)
            logger.info(
                f"Namespace deletion completed: '{namespace}' deleted, "
                f"affected {rows_affected} rows"
            )

            return {"deleted": namespace, "rows_affected": rows_affected}

        except Exception as e:
            logger.error(f"Failed to delete namespace: {e}")
            raise RuntimeError(f"Namespace deletion failed: {e}")

    def get_namespace_for_project(self, project_id: str) -> str:
        """Get the namespace name for a project."""
        # Use project ID as namespace, with prefix for safety
        safe_project_id = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in project_id
        )
        return f"mcp_code_{safe_project_id}".lower()

    def delete_vectors_for_file(self, namespace: str, file_path: str) -> Dict[str, Any]:
        """
        Delete all vectors associated with a specific file.

        Args:
            namespace: The namespace to delete from
            file_path: Path to the source file

        Returns:
            Dictionary with deletion results

        Raises:
            RuntimeError: If deletion fails
        """
        logger.info(
            f"Deleting vectors for file '{file_path}' in namespace '{namespace}'"
        )

        try:
            ns = self.client.namespace(namespace)

            # First, query for vectors with matching file_path
            filter_condition = ("file_path", "Eq", file_path)
            results = ns.query(
                filters=filter_condition,
                top_k=1200,  # Set high enough to catch all chunks for a single file. 1200 is max
                include_attributes=False,  # We only need IDs
            )

            if not hasattr(results, "rows") or not results.rows:
                logger.info(
                    f"No vectors found for file '{file_path}' in namespace '{namespace}'"
                )
                return {"deleted": 0, "file_path": file_path}

            # Extract vector IDs to delete
            ids_to_delete = [row.id for row in results.rows]
            logger.info(
                f"Found {len(ids_to_delete)} vectors to delete for file '{file_path}'"
            )

            # Delete vectors by ID using existing method
            delete_result = self.delete_vectors(ids_to_delete, namespace)

            logger.info(
                f"File deletion completed: removed {delete_result['deleted']} vectors "
                f"for file '{file_path}' from namespace '{namespace}'"
            )

            return {"deleted": delete_result["deleted"], "file_path": file_path}

        except Exception as e:
            logger.error(f"Failed to delete vectors for file '{file_path}': {e}")
            raise RuntimeError(f"File vector deletion failed: {e}")

    def search_with_metadata_filter(
        self,
        query_vector: List[float],
        project_id: str,
        chunk_type: Optional[str] = None,
        file_path: Optional[str] = None,
        top_k: int = 10,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Search vectors with metadata filtering."""
        namespace = self.get_namespace_for_project(project_id)

        # Build metadata filters
        filters = {"project_id": project_id}
        if chunk_type:
            filters["chunk_type"] = chunk_type
        if file_path:
            filters["file_path"] = file_path

        return self.search_vectors(
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filters=filters,
            **kwargs,
        )


def create_turbopuffer_client(config: VectorConfig) -> TurbopufferClient:
    """Create a Turbopuffer client from configuration."""
    if not config.turbopuffer_api_key:
        raise ValueError("TURBOPUFFER_API_KEY is required for vector storage")

    return TurbopufferClient(
        api_key=config.turbopuffer_api_key,
        region=config.turbopuffer_region,
    )
