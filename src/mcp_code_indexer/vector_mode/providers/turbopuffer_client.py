"""
Turbopuffer client for vector storage and search.

Provides integration with Turbopuffer's vector database for storing
embeddings and performing similarity searches.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Union
import json

from .base_provider import BaseProvider, ProviderError
from ..config import VectorConfig

logger = logging.getLogger(__name__)

class TurbopufferClient(BaseProvider):
    """Client for Turbopuffer vector database."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.turbopuffer.com/v1",
        **kwargs
    ):
        super().__init__(api_key, base_url, **kwargs)
    
    async def health_check(self) -> bool:
        """Check if Turbopuffer service is healthy."""
        try:
            # List namespaces to test connectivity
            await self.list_namespaces()
            return True
        except Exception as e:
            logger.warning(f"Turbopuffer health check failed: {e}")
            return False
    
    def _generate_vector_id(self, project_id: str, chunk_id: int) -> str:
        """Generate a unique vector ID."""
        return f"{project_id}_{chunk_id}_{uuid.uuid4().hex[:8]}"
    
    async def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Store or update vectors in the database.
        
        Args:
            vectors: List of vector objects with id, values, and metadata
            namespace: Turbopuffer namespace to store vectors in
            **kwargs: Additional arguments
            
        Returns:
            Response from Turbopuffer API
        """
        if not vectors:
            return {"upserted": 0}
        
        logger.info(f"Upserting {len(vectors)} vectors to namespace '{namespace}'")
        
        # Format vectors for Turbopuffer API
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
        
        request_data = {
            "vectors": formatted_vectors,
        }
        
        try:
            response = await self._make_request(
                method="POST",
                endpoint=f"/namespaces/{namespace}/vectors",
                data=request_data,
            )
            
            logger.info(f"Successfully upserted {len(vectors)} vectors")
            return response
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise ProviderError(f"Vector upsert failed: {e}")
    
    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        namespace: str = "default",
        filters: Optional[Dict[str, Any]] = None,
        include_attributes: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector to search with
            top_k: Number of results to return
            namespace: Turbopuffer namespace to search in
            filters: Metadata filters to apply
            include_attributes: Whether to include vector attributes in results
            **kwargs: Additional arguments
            
        Returns:
            List of search results with id, score, and metadata
        """
        logger.debug(f"Searching {top_k} vectors in namespace '{namespace}'")
        
        request_data = {
            "vector": query_vector,
            "top_k": top_k,
            "include_attributes": include_attributes,
        }
        
        if filters:
            request_data["filters"] = filters
        
        try:
            response = await self._make_request(
                method="POST",
                endpoint=f"/namespaces/{namespace}/search",
                data=request_data,
            )
            
            results = response.get("results", [])
            logger.debug(f"Found {len(results)} similar vectors")
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise ProviderError(f"Vector search failed: {e}")
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        namespace: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delete vectors by ID.
        
        Args:
            vector_ids: List of vector IDs to delete
            namespace: Turbopuffer namespace
            **kwargs: Additional arguments
            
        Returns:
            Response from Turbopuffer API
        """
        if not vector_ids:
            return {"deleted": 0}
        
        logger.info(f"Deleting {len(vector_ids)} vectors from namespace '{namespace}'")
        
        request_data = {
            "ids": vector_ids,
        }
        
        try:
            response = await self._make_request(
                method="DELETE",
                endpoint=f"/namespaces/{namespace}/vectors",
                data=request_data,
            )
            
            logger.info(f"Successfully deleted vectors")
            return response
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise ProviderError(f"Vector deletion failed: {e}")
    
    async def get_namespace_stats(
        self,
        namespace: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get statistics about a namespace.
        
        Args:
            namespace: Turbopuffer namespace
            **kwargs: Additional arguments
            
        Returns:
            Namespace statistics
        """
        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/namespaces/{namespace}",
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get namespace stats: {e}")
            raise ProviderError(f"Namespace stats failed: {e}")
    
    async def list_namespaces(self) -> List[str]:
        """List all available namespaces."""
        try:
            response = await self._make_request(
                method="GET",
                endpoint="/namespaces",
            )
            
            namespaces = response.get("namespaces", [])
            return [ns["name"] for ns in namespaces]
            
        except Exception as e:
            logger.error(f"Failed to list namespaces: {e}")
            raise ProviderError(f"Namespace listing failed: {e}")
    
    async def create_namespace(
        self,
        namespace: str,
        dimension: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new namespace.
        
        Args:
            namespace: Name of the namespace to create
            dimension: Vector dimension for the namespace
            **kwargs: Additional arguments
            
        Returns:
            Response from Turbopuffer API
        """
        logger.info(f"Creating namespace '{namespace}' with dimension {dimension}")
        
        request_data = {
            "name": namespace,
            "dimension": dimension,
        }
        
        try:
            response = await self._make_request(
                method="POST",
                endpoint="/namespaces",
                data=request_data,
            )
            
            logger.info(f"Successfully created namespace '{namespace}'")
            return response
            
        except Exception as e:
            logger.error(f"Failed to create namespace: {e}")
            raise ProviderError(f"Namespace creation failed: {e}")
    
    async def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """Delete a namespace and all its vectors."""
        logger.warning(f"Deleting namespace '{namespace}' and all its vectors")
        
        try:
            response = await self._make_request(
                method="DELETE",
                endpoint=f"/namespaces/{namespace}",
            )
            
            logger.info(f"Successfully deleted namespace '{namespace}'")
            return response
            
        except Exception as e:
            logger.error(f"Failed to delete namespace: {e}")
            raise ProviderError(f"Namespace deletion failed: {e}")
    
    def get_namespace_for_project(self, project_id: str) -> str:
        """Get the namespace name for a project."""
        # Use project ID as namespace, with prefix for safety
        safe_project_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_id)
        return f"mcp_code_{safe_project_id}".lower()
    
    async def search_with_metadata_filter(
        self,
        query_vector: List[float],
        project_id: str,
        chunk_type: Optional[str] = None,
        file_path: Optional[str] = None,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search vectors with metadata filtering.
        
        Args:
            query_vector: Query vector
            project_id: Project to search within
            chunk_type: Filter by chunk type (optional)
            file_path: Filter by file path (optional)
            top_k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            Filtered search results
        """
        namespace = self.get_namespace_for_project(project_id)
        
        # Build metadata filters
        filters = {"project_id": project_id}
        if chunk_type:
            filters["chunk_type"] = chunk_type
        if file_path:
            filters["file_path"] = file_path
        
        return await self.search_vectors(
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filters=filters,
            **kwargs
        )

def create_turbopuffer_client(config: VectorConfig) -> TurbopufferClient:
    """Create a Turbopuffer client from configuration."""
    if not config.turbopuffer_api_key:
        raise ValueError("TURBOPUFFER_API_KEY is required for vector storage")
    
    return TurbopufferClient(
        api_key=config.turbopuffer_api_key,
        timeout=30.0,
        max_retries=3,
    )
