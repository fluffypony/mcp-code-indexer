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
                raise RuntimeError(
                    f"Turbopuffer API access validation failed: {e}"
                )

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

        # Format vectors for Turbopuffer SDK
        formatted_vectors = []
        for vector in vectors:
            if "id" not in vector or "values" not in vector:
                raise ValueError("Each vector must have 'id' and 'values' fields")

            formatted_vector = {
                "id": str(vector["id"]),
                "vector": vector["values"],
                "attributes": vector.get("metadata", {}),
            }
            formatted_vectors.append(formatted_vector)

        try:
            ns = self.client.namespace(namespace)
            ns.upsert(vectors=formatted_vectors)

            logger.info(f"Successfully upserted {len(vectors)} vectors")
            return {"upserted": len(vectors)}

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

            results = ns.query(
                rank_by=[("vector", "ANN", query_vector)],
                top_k=top_k,
                filters=filters,
                include_attributes=True,
            )

            logger.debug(f"Found {len(results)} similar vectors")
            return results

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
            ns.delete(ids=vector_ids)

            logger.info(f"Successfully deleted vectors")
            return {"deleted": len(vector_ids)}

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise RuntimeError(f"Vector deletion failed: {e}")

    def list_namespaces(self) -> List[str]:
        """List all available namespaces."""
        try:
            namespaces = self.client.namespaces()
            return [ns.name for ns in namespaces]

        except Exception as e:
            logger.error(f"Failed to list namespaces: {e}")
            raise RuntimeError(f"Namespace listing failed: {e}")

    def create_namespace(
        self, namespace: str, dimension: int, **kwargs
    ) -> Dict[str, Any]:
        """Create a new namespace."""
        logger.info(f"Creating namespace '{namespace}' with dimension {dimension}")

        try:
            self.client.create_namespace(name=namespace, dimension=dimension)

            logger.info(f"Successfully created namespace '{namespace}'")
            return {"name": namespace, "dimension": dimension}

        except Exception as e:
            logger.error(f"Failed to create namespace: {e}")
            raise RuntimeError(f"Namespace creation failed: {e}")

    def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """Delete a namespace and all its vectors."""
        logger.warning(f"Deleting namespace '{namespace}' and all its vectors")

        try:
            self.client.delete_namespace(namespace)

            logger.info(f"Successfully deleted namespace '{namespace}'")
            return {"deleted": namespace}

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
