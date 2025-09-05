"""
Unit tests for VectorStorageService.

Tests the service layer that handles vector storage operations,
namespace management, and vector formatting for vector database integration.
"""

from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
import pytest_asyncio

from mcp_code_indexer.vector_mode.services.vector_storage_service import VectorStorageService
from mcp_code_indexer.vector_mode.providers.turbopuffer_client import TurbopufferClient
from mcp_code_indexer.vector_mode.config import VectorConfig
from mcp_code_indexer.vector_mode.chunking.ast_chunker import CodeChunk
from mcp_code_indexer.database.models import ChunkType


class TestVectorStorageService:
    """Test VectorStorageService vector storage operations."""

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
    def vector_storage_service(
        self, mock_turbopuffer_client: TurbopufferClient, vector_config: VectorConfig
    ) -> VectorStorageService:
        """Create a VectorStorageService for testing."""
        # Mock the validate_api_access method to avoid actual API calls during testing
        mock_turbopuffer_client.validate_api_access = MagicMock()
        return VectorStorageService(mock_turbopuffer_client, 1536, vector_config)

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
        vector_storage_service: VectorStorageService,
        sample_embeddings: List[List[float]],
        sample_chunks: List[CodeChunk],
    ):
        """Test successful storage of embeddings."""
        project_name = "test_project"
        file_path = "/test/file.py"

        # Mock successful deletion before upsert
        vector_storage_service.turbopuffer_client.delete_vectors_for_file.return_value = {
            "deleted": 0, "file_path": file_path
        }

        await vector_storage_service.store_embeddings(
            sample_embeddings, sample_chunks, project_name, file_path
        )

        # Verify client methods were called including deletion
        vector_storage_service.turbopuffer_client.get_namespace_for_project.assert_called_once_with(project_name)
        vector_storage_service.turbopuffer_client.delete_vectors_for_file.assert_called_once_with(
            "mcp_code_test_project", file_path
        )
        vector_storage_service.turbopuffer_client.upsert_vectors.assert_called_once()

        # Verify vector formatting
        call_args = vector_storage_service.turbopuffer_client.upsert_vectors.call_args
        vectors = call_args[0][0]  # First argument
        namespace = call_args[0][1]  # Second argument

        assert namespace == "mcp_code_test_project"
        assert len(vectors) == 2
        assert vectors[0]["id"] == "test_project_vec_0_abc123"
        assert vectors[1]["id"] == "test_project_vec_1_abc123"
        assert vectors[0]["values"] == sample_embeddings[0]
        assert vectors[1]["values"] == sample_embeddings[1]

    async def test_store_embeddings_empty_list(
        self, vector_storage_service: VectorStorageService
    ):
        """Test storing empty embeddings list."""
        project_name = "test_project"
        file_path = "/test/file.py"

        await vector_storage_service.store_embeddings(
            [], [], project_name, file_path
        )

        # Should not call any client methods for empty list
        vector_storage_service.turbopuffer_client.upsert_vectors.assert_not_called()

    async def test_store_embeddings_count_mismatch(
        self,
        vector_storage_service: VectorStorageService,
        sample_embeddings: List[List[float]],
        sample_chunks: List[CodeChunk],
    ):
        """Test handling of mismatched embeddings and chunks count."""
        project_name = "test_project"
        file_path = "/test/file.py"

        # Remove one embedding to create mismatch
        mismatched_embeddings = sample_embeddings[:-1]

        with pytest.raises(ValueError, match="Embeddings and chunks count mismatch"):
            await vector_storage_service.store_embeddings(
                mismatched_embeddings, sample_chunks, project_name, file_path
            )

    async def test_ensure_namespace_exists_existing(
        self, vector_storage_service: VectorStorageService
    ):
        """Test namespace check when namespace already exists."""
        project_name = "test_project"

        # Mock existing namespace
        vector_storage_service.turbopuffer_client.list_namespaces.return_value = ["mcp_code_test_project"]

        namespace = await vector_storage_service._ensure_namespace_exists(project_name)

        assert namespace == "mcp_code_test_project"

    async def test_ensure_namespace_exists_create_new(
        self, vector_storage_service: VectorStorageService
    ):
        """Test namespace handling when namespace doesn't exist (returns None)."""
        project_name = "test_project"

        # Mock no existing namespaces
        vector_storage_service.turbopuffer_client.list_namespaces.return_value = []

        namespace = await vector_storage_service._ensure_namespace_exists(project_name)

        assert namespace is None

    async def test_ensure_namespace_exists_cached(
        self, vector_storage_service: VectorStorageService
    ):
        """Test namespace caching to avoid repeated API calls."""
        project_name = "test_project"

        # First call
        await vector_storage_service._ensure_namespace_exists(project_name)
        
        # Second call should use cache
        await vector_storage_service._ensure_namespace_exists(project_name)

        # Should only call list_namespaces once
        assert vector_storage_service.turbopuffer_client.list_namespaces.call_count == 1

    async def test_format_vectors_for_storage(
        self,
        vector_storage_service: VectorStorageService,
        sample_embeddings: List[List[float]],
        sample_chunks: List[CodeChunk],
    ):
        """Test vector formatting for Turbopuffer storage."""
        project_name = "test_project"
        file_path = "/test/file.py"

        vectors = vector_storage_service._format_vectors_for_storage(
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

    async def test_store_embeddings_dimension_from_embeddings(
        self,
        vector_storage_service: VectorStorageService,
        sample_chunks: List[CodeChunk],
    ):
        """Test that embedding dimension is derived from actual embeddings."""
        project_name = "test_project"
        file_path = "/test/file.py"
        
        # Mock namespace doesn't exist so it gets created
        vector_storage_service.turbopuffer_client.list_namespaces.return_value = []
        
        # Create embeddings with non-standard dimension
        custom_embeddings = [
            [0.1, 0.2, 0.3, 0.4],  # 4 dimensions
            [0.5, 0.6, 0.7, 0.8]
        ]

        await vector_storage_service.store_embeddings(
            custom_embeddings, sample_chunks, project_name, file_path
        )

        # Verify that embeddings were stored by checking upsert_vectors was called
        vector_storage_service.turbopuffer_client.upsert_vectors.assert_called_once()

    async def test_store_embeddings_with_client_error(
        self,
        vector_storage_service: VectorStorageService,
        sample_embeddings: List[List[float]],
        sample_chunks: List[CodeChunk],
    ):
        """Test handling of TurbopufferClient errors."""
        project_name = "test_project"
        file_path = "/test/file.py"

        # Mock client error
        vector_storage_service.turbopuffer_client.upsert_vectors.side_effect = RuntimeError("API Error")

        with pytest.raises(RuntimeError, match="Vector storage failed: API Error"):
            await vector_storage_service.store_embeddings(
                sample_embeddings, sample_chunks, project_name, file_path
            )

    async def test_search_similar_chunks(
        self, vector_storage_service: VectorStorageService
    ):
        """Test searching for similar chunks."""
        query_embedding = [0.1] * 1536
        project_name = "test_project"
        
        # Mock search results - raw results from turbopuffer_client
        mock_results = [
            {"id": "test_vec_1", "score": 0.9, "metadata": {"chunk_name": "similar_function"}},
            {"id": "test_vec_2", "score": 0.8, "metadata": {"chunk_name": "another_function"}}
        ]
        vector_storage_service.turbopuffer_client.search_with_metadata_filter.return_value = mock_results

        results = await vector_storage_service.search_similar_chunks(
            query_embedding, project_name, top_k=5, chunk_type="function"
        )

        # Should return results directly without formatting
        assert results == mock_results
        vector_storage_service.turbopuffer_client.search_with_metadata_filter.assert_called_once_with(
            query_vector=query_embedding,
            project_id=project_name,
            chunk_type="function",
            file_path=None,
            top_k=5,
        )

    async def test_search_similar_chunks_with_error(
        self, vector_storage_service: VectorStorageService
    ):
        """Test search error handling."""
        query_embedding = [0.1] * 1536
        project_name = "test_project"

        # Mock search error
        vector_storage_service.turbopuffer_client.search_with_metadata_filter.side_effect = RuntimeError("Search Error")

        with pytest.raises(RuntimeError, match="Vector search failed: Search Error"):
            await vector_storage_service.search_similar_chunks(
                query_embedding, project_name
            )

    async def test_vector_metadata_with_custom_metadata(
        self,
        vector_storage_service: VectorStorageService,
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

        vectors = vector_storage_service._format_vectors_for_storage(
            sample_embeddings[:1], [chunk], "test_project", "/test/file.py"
        )

        assert len(vectors) == 1
        metadata = vectors[0]["metadata"]
        assert metadata["custom_field"] == "custom_value"
        assert metadata["priority"] == "high"
        assert metadata["chunk_name"] == "custom_function"  # Original metadata preserved

    async def test_service_initialization_validates_api_access(
        self, mock_turbopuffer_client: TurbopufferClient, vector_config: VectorConfig
    ):
        """Test that API access is validated during service initialization."""
        # Mock successful validation
        mock_turbopuffer_client.validate_api_access = MagicMock()
        
        # Create service (should call validate_api_access)
        service = VectorStorageService(mock_turbopuffer_client, 1536, vector_config)
        
        # Verify validation was called
        mock_turbopuffer_client.validate_api_access.assert_called_once()

    async def test_service_initialization_validation_failure(
        self, mock_turbopuffer_client: TurbopufferClient, vector_config: VectorConfig
    ):
        """Test that service initialization fails when API validation fails."""
        # Mock validation failure
        mock_turbopuffer_client.validate_api_access = MagicMock(
            side_effect=RuntimeError("API validation failed")
        )
        
        # Service creation should fail
        with pytest.raises(RuntimeError, match="API validation failed"):
            VectorStorageService(mock_turbopuffer_client, 1536, vector_config)

    async def test_delete_vectors_for_file_success(
        self, vector_storage_service: VectorStorageService
    ):
        """Test successful deletion of vectors for a file."""
        project_name = "test_project"
        file_path = "/test/file.py"
        
        # Mock successful deletion
        vector_storage_service.turbopuffer_client.delete_vectors_for_file.return_value = {
            "deleted": 3, "file_path": file_path
        }
        
        # Test deletion
        await vector_storage_service.delete_vectors_for_file(project_name, file_path)
        
        # Verify client method was called correctly
        vector_storage_service.turbopuffer_client.get_namespace_for_project.assert_called_once_with(project_name)
        vector_storage_service.turbopuffer_client.delete_vectors_for_file.assert_called_once_with(
            "mcp_code_test_project", file_path
        )

    async def test_delete_vectors_for_file_failure(
        self, vector_storage_service: VectorStorageService
    ):
        """Test handling of deletion failures."""
        project_name = "test_project"
        file_path = "/test/file.py"
        
        # Mock deletion failure
        vector_storage_service.turbopuffer_client.delete_vectors_for_file.side_effect = RuntimeError(
            "Deletion failed"
        )
        
        # Test deletion should raise RuntimeError
        with pytest.raises(RuntimeError, match="Vector deletion failed: Deletion failed"):
            await vector_storage_service.delete_vectors_for_file(project_name, file_path)

    async def test_store_embeddings_with_deletion_step(
        self,
        vector_storage_service: VectorStorageService,
        sample_embeddings: List[List[float]],
        sample_chunks: List[CodeChunk],
    ):
        """Test that store_embeddings now includes deletion step before upserting."""
        project_name = "test_project"
        file_path = "/test/file.py"
        
        # Mock successful deletion before upsert
        vector_storage_service.turbopuffer_client.delete_vectors_for_file.return_value = {
            "deleted": 1, "file_path": file_path
        }
        
        # Store embeddings
        await vector_storage_service.store_embeddings(
            sample_embeddings, sample_chunks, project_name, file_path
        )
        
        # Verify deletion was called before upsert
        vector_storage_service.turbopuffer_client.delete_vectors_for_file.assert_called_once_with(
            "mcp_code_test_project", file_path
        )
        
        # Verify upsert was still called
        vector_storage_service.turbopuffer_client.upsert_vectors.assert_called_once()

    async def test_store_embeddings_deletion_failure_continues_with_upsert(
        self,
        vector_storage_service: VectorStorageService,
        sample_embeddings: List[List[float]],
        sample_chunks: List[CodeChunk],
    ):
        """Test that store_embeddings continues with upsert even if deletion fails."""
        project_name = "test_project"
        file_path = "/test/file.py"
        
        # Mock deletion failure
        vector_storage_service.turbopuffer_client.delete_vectors_for_file.side_effect = RuntimeError(
            "Deletion failed"
        )
        
        # Store embeddings should still succeed despite deletion failure
        await vector_storage_service.store_embeddings(
            sample_embeddings, sample_chunks, project_name, file_path
        )
        
        # Verify deletion was attempted
        vector_storage_service.turbopuffer_client.delete_vectors_for_file.assert_called_once()
        
        # Verify upsert was still called even though deletion failed
        vector_storage_service.turbopuffer_client.upsert_vectors.assert_called_once()

    async def test_get_file_metadata_all_files(
        self, vector_storage_service: VectorStorageService
    ):
        """Test retrieving file metadata for all files in a project."""
        project_name = "test_project"
        
        # Create mock Row objects matching the real interface
        class MockRow:
            def __init__(self, file_path: str, file_mtime: str, **kwargs):
                self.file_path = file_path
                self.file_mtime = file_mtime
                # Add other attributes that might be present
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        mock_rows = [
            MockRow("/test/file1.py", "1751984561.6831303", 
                   id="test_project_0_123", chunk_type="function"),
            MockRow("/test/file2.py", "1751984562.5", 
                   id="test_project_1_456", chunk_type="class"),
            MockRow("/test/file1.py", "1751984563.0",  # newer timestamp for same file
                   id="test_project_2_789", chunk_type="method"),
        ]
        
        # Mock the search_vectors call
        vector_storage_service.turbopuffer_client.search_vectors.return_value = mock_rows
        
        result = await vector_storage_service.get_file_metadata(project_name)
        
        # Should return the most recent mtime for each file
        expected_result = {
            "/test/file1.py": 1751984563.0,  # Most recent timestamp
            "/test/file2.py": 1751984562.5,
        }
        assert result == expected_result
        
        # Verify correct search_vectors call
        vector_storage_service.turbopuffer_client.search_vectors.assert_called_once_with(
            query_vector=[0.0] * 1536,  # dummy vector with embedding_dimension
            top_k=1200,
            namespace="mcp_code_test_project",
            filters={"project_id": project_name},
        )

    async def test_get_file_metadata_specific_files(
        self, vector_storage_service: VectorStorageService
    ):
        """Test retrieving file metadata for specific files."""
        project_name = "test_project"
        file_paths = ["/test/file1.py", "/test/file2.py"]
        
        class MockRow:
            def __init__(self, file_path: str, file_mtime: str, **kwargs):
                self.file_path = file_path
                self.file_mtime = file_mtime
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        # Mock responses for each file path query
        mock_rows_file1 = [
            MockRow("/test/file1.py", "1751984561.0", id="test_project_0_123"),
            MockRow("/test/file1.py", "1751984562.0", id="test_project_1_456"),  # newer
        ]
        mock_rows_file2 = [
            MockRow("/test/file2.py", "1751984563.0", id="test_project_2_789"),
        ]
        
        # Mock search_vectors to return different results for different calls
        vector_storage_service.turbopuffer_client.search_vectors.side_effect = [
            mock_rows_file1, mock_rows_file2
        ]
        
        result = await vector_storage_service.get_file_metadata(project_name, file_paths)
        
        expected_result = {
            "/test/file1.py": 1751984562.0,  # Most recent from file1
            "/test/file2.py": 1751984563.0,
        }
        assert result == expected_result
        
        # Verify correct number of search_vectors calls (one per file)
        assert vector_storage_service.turbopuffer_client.search_vectors.call_count == 2

    async def test_get_file_metadata_no_namespace(
        self, vector_storage_service: VectorStorageService
    ):
        """Test get_file_metadata when namespace doesn't exist."""
        project_name = "nonexistent_project"
        
        # Mock _ensure_namespace_exists to return None (namespace doesn't exist)
        with patch.object(vector_storage_service, '_ensure_namespace_exists', return_value=None):
            result = await vector_storage_service.get_file_metadata(project_name)
            
        assert result == {}

    async def test_get_file_metadata_no_rows_found(
        self, vector_storage_service: VectorStorageService
    ):
        """Test get_file_metadata when no rows are found."""
        project_name = "test_project"
        
        # Mock search_vectors to return None (no rows found)
        vector_storage_service.turbopuffer_client.search_vectors.return_value = None
        
        result = await vector_storage_service.get_file_metadata(project_name)
        
        assert result == {}

    async def test_get_file_metadata_empty_rows(
        self, vector_storage_service: VectorStorageService
    ):
        """Test get_file_metadata when empty rows are returned."""
        project_name = "test_project"
        
        # Mock search_vectors to return empty list
        vector_storage_service.turbopuffer_client.search_vectors.return_value = []
        
        result = await vector_storage_service.get_file_metadata(project_name)
        
        assert result == {}

    async def test_get_file_metadata_missing_attributes(
        self, vector_storage_service: VectorStorageService
    ):
        """Test get_file_metadata with rows missing file_path or file_mtime attributes."""
        project_name = "test_project"
        
        class MockRow:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        mock_rows = [
            MockRow(file_path="/test/file1.py", file_mtime="1751984561.0"),  # Complete row
            MockRow(file_path="/test/file2.py"),  # Missing file_mtime
            MockRow(file_mtime="1751984562.0"),   # Missing file_path
            MockRow(id="test_project_3_xyz"),     # Missing both
        ]
        
        vector_storage_service.turbopuffer_client.search_vectors.return_value = mock_rows
        
        result = await vector_storage_service.get_file_metadata(project_name)
        
        # Should only include rows with both attributes
        expected_result = {
            "/test/file1.py": 1751984561.0,
        }
        assert result == expected_result

    async def test_get_file_metadata_invalid_mtime_format(
        self, vector_storage_service: VectorStorageService
    ):
        """Test get_file_metadata handles invalid mtime format gracefully."""
        project_name = "test_project"
        
        class MockRow:
            def __init__(self, file_path: str, file_mtime: str):
                self.file_path = file_path
                self.file_mtime = file_mtime
        
        mock_rows = [
            MockRow("/test/file1.py", "1751984561.0"),     # Valid
            MockRow("/test/file2.py", "invalid_timestamp"), # Invalid
            MockRow("/test/file3.py", "1751984562.5"),     # Valid
        ]
        
        vector_storage_service.turbopuffer_client.search_vectors.return_value = mock_rows
        
        # The method should handle the invalid timestamp gracefully and return empty dict
        result = await vector_storage_service.get_file_metadata(project_name)
        
        # Should return empty dict due to exception handling
        assert result == {}

    async def test_get_file_metadata_exception_handling(
        self, vector_storage_service: VectorStorageService
    ):
        """Test get_file_metadata exception handling."""
        project_name = "test_project"
        
        # Mock search_vectors to raise an exception
        vector_storage_service.turbopuffer_client.search_vectors.side_effect = RuntimeError("Search failed")
        
        result = await vector_storage_service.get_file_metadata(project_name)
        
        # Should return empty dict on exception
        assert result == {}

    async def test_get_file_metadata_specific_files_with_empty_list(
        self, vector_storage_service: VectorStorageService
    ):
        """Test get_file_metadata with empty file_paths list."""
        project_name = "test_project"
        
        class MockRow:
            def __init__(self, file_path: str, file_mtime: str):
                self.file_path = file_path
                self.file_mtime = file_mtime
        
        mock_rows = [
            MockRow("/test/file1.py", "1751984561.0"),
        ]
        
        vector_storage_service.turbopuffer_client.search_vectors.return_value = mock_rows
        
        # Empty list should be treated like None (query all files)
        result = await vector_storage_service.get_file_metadata(project_name, [])
        
        expected_result = {
            "/test/file1.py": 1751984561.0,
        }
        assert result == expected_result
        
        # Verify it called search_vectors with project-wide filters (not per-file)
        vector_storage_service.turbopuffer_client.search_vectors.assert_called_once_with(
            query_vector=[0.0] * 1536,
            top_k=1200,
            namespace="mcp_code_test_project",
            filters={"project_id": project_name},
        )
