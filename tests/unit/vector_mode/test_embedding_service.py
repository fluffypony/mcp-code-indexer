"""
Unit tests for EmbeddingService.

Tests the embedding service functionality including text preparation,
batching, and VoyageClient integration.
"""

from pathlib import Path
from typing import List
from unittest.mock import Mock, patch
import pytest
import pytest_asyncio

from mcp_code_indexer.vector_mode.services.embedding_service import EmbeddingService
from mcp_code_indexer.vector_mode.config import VectorConfig
from mcp_code_indexer.vector_mode.chunking.ast_chunker import CodeChunk
from mcp_code_indexer.database.models import ChunkType


class TestEmbeddingService:
    """Test EmbeddingService functionality."""

    @pytest.fixture
    def mock_voyage_client(self):
        """Create a mock VoyageClient."""
        return Mock()

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return VectorConfig(
            voyage_api_key="test-key",
            embedding_model="voyage-code-2",
            batch_size=32
        )

    @pytest.fixture
    def embedding_service(self, mock_voyage_client, config):
        """Create an EmbeddingService with mocked client."""
        # Mock the validate_api_access method to avoid actual API calls during testing
        mock_voyage_client.validate_api_access = Mock()
        return EmbeddingService(mock_voyage_client, config)

    @pytest.fixture
    def sample_chunks(self):
        """Create sample code chunks."""
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

    async def test_generate_embeddings_success(
        self, embedding_service, mock_voyage_client, sample_chunks
    ):
        """Test successful embedding generation."""
        project_name = "test_project"
        file_path = Path("/test/file.py")

        # Mock VoyageClient response
        mock_voyage_client.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3] * 512,  # 1536 dimensions
            [0.4, 0.5, 0.6] * 512
        ]

        embeddings = await embedding_service.generate_embeddings_for_chunks(
            sample_chunks, project_name, file_path
        )

        # Verify embeddings were returned
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536

        # Verify VoyageClient was called correctly
        mock_voyage_client.generate_embeddings.assert_called_once()
        call_args = mock_voyage_client.generate_embeddings.call_args
        texts = call_args[0][0]
        input_type = call_args[1]["input_type"]

        assert len(texts) == 2
        assert input_type == "document"
        # Verify context enhancement
        assert "# hello_world\ndef hello_world():" in texts[0]
        assert "# TestClass\nclass TestClass:" in texts[1]

    async def test_generate_embeddings_empty_chunks(
        self, embedding_service, mock_voyage_client
    ):
        """Test handling of empty chunk list."""
        project_name = "test_project"
        file_path = Path("/test/file.py")

        embeddings = await embedding_service.generate_embeddings_for_chunks(
            [], project_name, file_path
        )

        # Should return empty list
        assert embeddings == []
        # VoyageClient should not be called
        mock_voyage_client.generate_embeddings.assert_not_called()

    async def test_generate_embeddings_batching(
        self, mock_voyage_client, config
    ):
        """Test batching logic with large chunk list."""
        # Create service with small batch size for testing
        config.batch_size = 3
        embedding_service = EmbeddingService(mock_voyage_client, config)

        # Create 5 chunks (more than batch size of 3)
        chunks = []
        for i in range(5):
            chunk = CodeChunk(
                content=f"def function_{i}():\n    pass",
                chunk_type=ChunkType.FUNCTION,
                name=f"function_{i}",
                file_path="/test/file.py",
                start_line=i*2,
                end_line=i*2+1,
                content_hash=f"hash_{i}",
                language="python"
            )
            chunks.append(chunk)

        project_name = "test_project"
        file_path = Path("/test/file.py")

        # Mock VoyageClient for batched calls
        mock_voyage_client.generate_embeddings.side_effect = [
            [[0.1] * 1536] * 3,  # First batch (3 chunks)
            [[0.2] * 1536] * 2   # Second batch (2 chunks)
        ]

        embeddings = await embedding_service.generate_embeddings_for_chunks(
            chunks, project_name, file_path
        )

        # Verify all embeddings returned
        assert len(embeddings) == 5

        # Verify batching occurred (2 calls)
        assert mock_voyage_client.generate_embeddings.call_count == 2

        # Verify batch sizes
        first_call_texts = mock_voyage_client.generate_embeddings.call_args_list[0][0][0]
        second_call_texts = mock_voyage_client.generate_embeddings.call_args_list[1][0][0]
        assert len(first_call_texts) == 3  # First batch
        assert len(second_call_texts) == 2  # Second batch

    async def test_generate_embeddings_api_error(
        self, embedding_service, mock_voyage_client, sample_chunks
    ):
        """Test handling of VoyageClient API errors."""
        project_name = "test_project"
        file_path = Path("/test/file.py")

        # Mock VoyageClient to raise exception
        mock_voyage_client.generate_embeddings.side_effect = RuntimeError("API Error")

        # Should propagate the error
        with pytest.raises(RuntimeError, match="API Error"):
            await embedding_service.generate_embeddings_for_chunks(
                sample_chunks, project_name, file_path
            )

    async def test_prepare_chunk_texts(self, embedding_service, sample_chunks):
        """Test text preparation from chunks."""
        texts = embedding_service._prepare_chunk_texts(sample_chunks)

        assert len(texts) == 2
        # Verify context enhancement for named chunks
        assert texts[0] == "# hello_world\ndef hello_world():\n    print('Hello, World!')"
        assert texts[1] == "# TestClass\nclass TestClass:\n    def __init__(self):\n        pass"

    async def test_prepare_chunk_texts_no_name(self, embedding_service):
        """Test text preparation for chunks without names."""
        chunk_without_name = CodeChunk(
            content="x = 42",
            chunk_type=ChunkType.VARIABLE,
            name=None,
            file_path="/test/file.py",
            start_line=1,
            end_line=1,
            content_hash="xyz789",
            language="python"
        )

        texts = embedding_service._prepare_chunk_texts([chunk_without_name])

        assert len(texts) == 1
        # No context prefix for unnamed chunks
        assert texts[0] == "x = 42"

    async def test_generate_embeddings_redacted_chunks(
        self, embedding_service, mock_voyage_client
    ):
        """Test embedding generation with redacted chunks."""
        redacted_chunk = CodeChunk(
            content="def secret_function():\n    api_key = '[REDACTED]'\n    return api_key",
            chunk_type=ChunkType.FUNCTION,
            name="secret_function",
            file_path="/test/file.py",
            start_line=1,
            end_line=3,
            content_hash="redacted123",
            language="python",
            redacted=True
        )

        project_name = "test_project"
        file_path = Path("/test/file.py")

        mock_voyage_client.generate_embeddings.return_value = [[0.1] * 1536]

        embeddings = await embedding_service.generate_embeddings_for_chunks(
            [redacted_chunk], project_name, file_path
        )

        # Should process redacted chunks normally
        assert len(embeddings) == 1

        # Verify redacted content was sent
        call_args = mock_voyage_client.generate_embeddings.call_args[0]
        texts = call_args[0]
        assert "[REDACTED]" in texts[0]

    async def test_service_initialization_validates_api_access(self, mock_voyage_client, config):
        """Test that API access is validated during service initialization."""
        # Mock successful validation
        mock_voyage_client.validate_api_access = Mock()
        
        # Create service (should call validate_api_access)
        service = EmbeddingService(mock_voyage_client, config)
        
        # Verify validation was called
        mock_voyage_client.validate_api_access.assert_called_once()

    async def test_service_initialization_validation_failure(self, mock_voyage_client, config):
        """Test that service initialization fails when API validation fails."""
        # Mock validation failure
        mock_voyage_client.validate_api_access = Mock(
            side_effect=RuntimeError("API validation failed")
        )
        
        # Service creation should fail
        with pytest.raises(RuntimeError, match="API validation failed"):
            EmbeddingService(mock_voyage_client, config)
