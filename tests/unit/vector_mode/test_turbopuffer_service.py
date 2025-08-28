"""
Unit tests for TurbopufferService.

Tests the service layer that handles vector storage operations,
namespace management, and vector formatting for Turbopuffer integration.
"""

from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
import pytest_asyncio

from mcp_code_indexer.vector_mode.services.turbopuffer_service import TurbopufferService
from mcp_code_indexer.vector_mode.providers.turbopuffer_client import TurbopufferClient
from mcp_code_indexer.vector_mode.config import VectorConfig
from mcp_code_indexer.vector_mode.chunking.ast_chunker import CodeChunk
from mcp_code_indexer.database.models import ChunkType


class TestTurbopufferService:
    """Test TurbopufferService vector storage operations."""

    @pytest.fixture
    def mock_turbopuffer_client(self) -> TurbopufferClient:
        """Create a mock TurbopufferClient for testing."""
        mock_client = MagicMock(spec=TurbopufferClient)
        mock_client.get_namespace_for_project.return_value = "mcp_code_test_project"
        mock_client.list_namespaces.return_value = ["mcp_code_test_project"]
        mock_client.generate_vector_id.side_effect = lambda project, idx: f"{project}_vec_{idx}_abc123"
        mock_client.upsert_vectors.return_value = {"upserted": 2}
        return mock_client

    @pytest.fixture
    def vector_config(self) -> VectorConfig:
        """Create a VectorConfig for testing."""
        return VectorConfig(
            voyage_api_key="test-voyage-key",
            turbopuffer_api_key="test-turbopuffer-key",
            embedding_model="voyage-code-2",
            batch_size=32,
            turbopuffer_region="gcp-europe-west3"
        )

    @pytest.fixture
    def turbopuffer_service(
        self, mock_turbopuffer_client: TurbopufferClient, vector_config: VectorConfig
    ) -> TurbopufferService:
        """Create a TurbopufferService for testing."""
        return TurbopufferService(mock_turbopuffer_client, vector_config)

    @pytest.fixture
    def sample_embeddings(self) -> List[List[float]]:
        """Create sample embeddings for testing."""
        return [
            [0.1, 0.2, 0.3] * 512,  # 1536 dimensions
            [0.4, 0.5, 0.6] * 512
        ]

    @pytest.fixture
    def sample_chunks(self) -> List[CodeChunk]:
        """Create sample code chunks for testing."""
        return [
            CodeChunk(
                content="def hello_world():\n    print('Hello, World!')",
                chunk_type=ChunkType.FUNCTION,
                name="hello_world",
                file_path="/test/file.py",
                start_line=1,
                end_line=2,
                content_hash="abc123",
                language="python",
                redacted=False,
                imports=["print"]
            ),
            CodeChunk(
                content="class TestClass:\n    def __init__(self):\n        pass",
                chunk_type=ChunkType.CLASS,
                name="TestClass",
                file_path="/test/file.py", 
                start_line=4,
                end_line=6,
                content_hash="def456",
                language="python",
                redacted=False,
                imports=[]
            )
        ]

    async def test_store_embeddings_success(
        self,
        turbopuffer_service: TurbopufferService,
        sample_embeddings: List[List[float]],
        sample_chunks: List[CodeChunk],
    ):
        """Test successful storage of embeddings."""
        project_name = "test_project"
        file_path = "/test/file.py"

        await turbopuffer_service.store_embeddings(
            sample_embeddings, sample_chunks, project_name, file_path
        )

        # Verify client methods were called
        turbopuffer_service.turbopuffer_client.get_namespace_for_project.assert_called_once_with(project_name)
        turbopuffer_service.turbopuffer_client.list_namespaces.assert_called_once()
        turbopuffer_service.turbopuffer_client.upsert_vectors.assert_called_once()

        # Verify vector formatting
        call_args = turbopuffer_service.turbopuffer_client.upsert_vectors.call_args
        vectors = call_args[0][0]  # First argument
        namespace = call_args[0][1]  # Second argument

        assert namespace == "mcp_code_test_project"
        assert len(vectors) == 2
        assert vectors[0]["id"] == "test_project_vec_0_abc123"
        assert vectors[1]["id"] == "test_project_vec_1_abc123"
        assert vectors[0]["values"] == sample_embeddings[0]
        assert vectors[1]["values"] == sample_embeddings[1]

    async def test_store_embeddings_empty_list(
        self, turbopuffer_service: TurbopufferService
    ):
        """Test storing empty embeddings list."""
        project_name = "test_project"
        file_path = "/test/file.py"

        await turbopuffer_service.store_embeddings(
            [], [], project_name, file_path
        )

        # Should not call any client methods for empty list
        turbopuffer_service.turbopuffer_client.upsert_vectors.assert_not_called()

    async def test_store_embeddings_count_mismatch(
        self,
        turbopuffer_service: TurbopufferService,
        sample_embeddings: List[List[float]],
        sample_chunks: List[CodeChunk],
    ):
        """Test handling of mismatched embeddings and chunks count."""
        project_name = "test_project"
        file_path = "/test/file.py"

        # Remove one embedding to create mismatch
        mismatched_embeddings = sample_embeddings[:-1]

        with pytest.raises(ValueError, match="Embeddings and chunks count mismatch"):
            await turbopuffer_service.store_embeddings(
                mismatched_embeddings, sample_chunks, project_name, file_path
            )

    async def test_ensure_namespace_exists_existing(
        self, turbopuffer_service: TurbopufferService
    ):
        """Test namespace check when namespace already exists."""
        project_name = "test_project"

        # Mock existing namespace
        turbopuffer_service.turbopuffer_client.list_namespaces.return_value = ["mcp_code_test_project"]

        namespace = await turbopuffer_service._ensure_namespace_exists(project_name)

        assert namespace == "mcp_code_test_project"
        turbopuffer_service.turbopuffer_client.create_namespace.assert_not_called()

    async def test_ensure_namespace_exists_create_new(
        self, turbopuffer_service: TurbopufferService
    ):
        """Test namespace creation when namespace doesn't exist."""
        project_name = "test_project"

        # Mock no existing namespaces
        turbopuffer_service.turbopuffer_client.list_namespaces.return_value = []

        namespace = await turbopuffer_service._ensure_namespace_exists(project_name)

        assert namespace == "mcp_code_test_project"
        turbopuffer_service.turbopuffer_client.create_namespace.assert_called_once_with(
            namespace="mcp_code_test_project", dimension=1536
        )

    async def test_ensure_namespace_exists_cached(
        self, turbopuffer_service: TurbopufferService
    ):
        """Test namespace caching to avoid repeated API calls."""
        project_name = "test_project"

        # First call
        await turbopuffer_service._ensure_namespace_exists(project_name)
        
        # Second call should use cache
        await turbopuffer_service._ensure_namespace_exists(project_name)

        # Should only call list_namespaces once
        assert turbopuffer_service.turbopuffer_client.list_namespaces.call_count == 1

    async def test_format_vectors_for_storage(
        self,
        turbopuffer_service: TurbopufferService,
        sample_embeddings: List[List[float]],
        sample_chunks: List[CodeChunk],
    ):
        """Test vector formatting for Turbopuffer storage."""
        project_name = "test_project"
        file_path = "/test/file.py"

        vectors = turbopuffer_service._format_vectors_for_storage(
            sample_embeddings, sample_chunks, project_name, file_path
        )

        assert len(vectors) == 2
        
        # Check first vector
        vector1 = vectors[0]
        assert vector1["id"] == "test_project_vec_0_abc123"
        assert vector1["values"] == sample_embeddings[0]
        assert vector1["metadata"]["project_name"] == project_name
        assert vector1["metadata"]["file_path"] == file_path
        assert vector1["metadata"]["chunk_type"] == "function"
        assert vector1["metadata"]["chunk_name"] == "hello_world"
        assert vector1["metadata"]["start_line"] == 1
        assert vector1["metadata"]["end_line"] == 2
        assert vector1["metadata"]["content_hash"] == "abc123"
        assert vector1["metadata"]["language"] == "python"
        assert vector1["metadata"]["redacted"] == False
        assert vector1["metadata"]["chunk_index"] == 0
        assert vector1["metadata"]["imports"] == "print"

        # Check second vector
        vector2 = vectors[1]
        assert vector2["id"] == "test_project_vec_1_abc123"
        assert vector2["values"] == sample_embeddings[1]
        assert vector2["metadata"]["chunk_type"] == "class"
        assert vector2["metadata"]["chunk_name"] == "TestClass"
        assert vector2["metadata"]["imports"] == ""  # No imports

    async def test_get_embedding_dimension(
        self, turbopuffer_service: TurbopufferService
    ):
        """Test embedding dimension detection for different models."""
        # voyage-code-2 (default in config)
        dimension = turbopuffer_service._get_embedding_dimension()
        assert dimension == 1536

        # Test other models
        turbopuffer_service.config.embedding_model = "voyage-2"
        dimension = turbopuffer_service._get_embedding_dimension()
        assert dimension == 1024

        turbopuffer_service.config.embedding_model = "unknown-model"
        dimension = turbopuffer_service._get_embedding_dimension()
        assert dimension == 1536  # Default fallback

    async def test_store_embeddings_with_client_error(
        self,
        turbopuffer_service: TurbopufferService,
        sample_embeddings: List[List[float]],
        sample_chunks: List[CodeChunk],
    ):
        """Test handling of TurbopufferClient errors."""
        project_name = "test_project"
        file_path = "/test/file.py"

        # Mock client error
        turbopuffer_service.turbopuffer_client.upsert_vectors.side_effect = RuntimeError("API Error")

        with pytest.raises(RuntimeError, match="Vector storage failed: API Error"):
            await turbopuffer_service.store_embeddings(
                sample_embeddings, sample_chunks, project_name, file_path
            )

    async def test_search_similar_chunks(
        self, turbopuffer_service: TurbopufferService
    ):
        """Test searching for similar chunks."""
        query_embedding = [0.1] * 1536
        project_name = "test_project"
        
        # Mock search results
        mock_results = [
            {"id": "test_vec_1", "score": 0.9, "metadata": {"chunk_name": "similar_function"}},
            {"id": "test_vec_2", "score": 0.8, "metadata": {"chunk_name": "another_function"}}
        ]
        turbopuffer_service.turbopuffer_client.search_with_metadata_filter.return_value = mock_results

        results = await turbopuffer_service.search_similar_chunks(
            query_embedding, project_name, top_k=5, chunk_type="function"
        )

        assert results == mock_results
        turbopuffer_service.turbopuffer_client.search_with_metadata_filter.assert_called_once_with(
            query_vector=query_embedding,
            project_id=project_name,
            chunk_type="function",
            file_path=None,
            top_k=5,
        )

    async def test_search_similar_chunks_with_error(
        self, turbopuffer_service: TurbopufferService
    ):
        """Test search error handling."""
        query_embedding = [0.1] * 1536
        project_name = "test_project"

        # Mock search error
        turbopuffer_service.turbopuffer_client.search_with_metadata_filter.side_effect = RuntimeError("Search Error")

        with pytest.raises(RuntimeError, match="Vector search failed: Search Error"):
            await turbopuffer_service.search_similar_chunks(
                query_embedding, project_name
            )

    async def test_vector_metadata_with_custom_metadata(
        self,
        turbopuffer_service: TurbopufferService,
        sample_embeddings: List[List[float]],
    ):
        """Test vector formatting with custom chunk metadata."""
        # Create chunk with custom metadata
        chunk = CodeChunk(
            content="def custom_function():\n    pass",
            chunk_type=ChunkType.FUNCTION,
            name="custom_function",
            file_path="/test/file.py",
            start_line=1,
            end_line=2,
            content_hash="custom123",
            language="python",
            redacted=False,
            metadata={"custom_field": "custom_value", "priority": "high"}
        )

        vectors = turbopuffer_service._format_vectors_for_storage(
            sample_embeddings[:1], [chunk], "test_project", "/test/file.py"
        )

        assert len(vectors) == 1
        metadata = vectors[0]["metadata"]
        assert metadata["custom_field"] == "custom_value"
        assert metadata["priority"] == "high"
        assert metadata["chunk_name"] == "custom_function"  # Original metadata preserved

    async def test_namespace_creation_failure(
        self, turbopuffer_service: TurbopufferService
    ):
        """Test handling of namespace creation failures."""
        project_name = "test_project"

        # Mock namespace doesn't exist
        turbopuffer_service.turbopuffer_client.list_namespaces.return_value = []
        # Mock creation failure
        turbopuffer_service.turbopuffer_client.create_namespace.side_effect = RuntimeError("Creation failed")

        with pytest.raises(RuntimeError, match="Namespace operation failed: Creation failed"):
            await turbopuffer_service._ensure_namespace_exists(project_name)
