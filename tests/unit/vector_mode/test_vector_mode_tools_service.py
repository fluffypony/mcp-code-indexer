"""
Unit tests for VectorModeToolsService.

Tests the find_similar_code functionality including input validation,
code chunking, embedding generation, and similarity search.
"""

from pathlib import Path
from typing import List
from unittest.mock import Mock, patch, AsyncMock
import pytest
import pytest_asyncio

from turbopuffer.types import Row

from mcp_code_indexer.vector_mode.services.vector_mode_tools_service import (
    VectorModeToolsService,
)
from mcp_code_indexer.vector_mode.providers.voyage_client import VoyageClient
from mcp_code_indexer.vector_mode.providers.turbopuffer_client import TurbopufferClient
from mcp_code_indexer.vector_mode.config import VectorConfig
from mcp_code_indexer.vector_mode.chunking.ast_chunker import CodeChunk, ASTChunker
from mcp_code_indexer.database.models import ChunkType


class TestVectorModeToolsService:
    """Test VectorModeToolsService functionality."""

    @pytest_asyncio.fixture
    async def mock_voyage_client(self) -> Mock:
        """Create a mock VoyageClient."""
        client = Mock(spec=VoyageClient)
        client.validate_api_access = AsyncMock()
        client.generate_embeddings = AsyncMock()
        return client

    @pytest_asyncio.fixture
    async def mock_turbopuffer_client(self) -> Mock:
        """Create a mock TurbopufferClient."""
        client = Mock(spec=TurbopufferClient)
        client.validate_api_access = AsyncMock()
        return client

    @pytest.fixture
    def mock_config(self) -> VectorConfig:
        """Create a test configuration."""
        return VectorConfig(
            voyage_api_key="test-voyage-key",
            turbopuffer_api_key="test-turbopuffer-key",
            embedding_model="voyage-code-2",
            batch_size=32,
            similarity_threshold=0.8,
            max_search_results=10,
        )

    @pytest.fixture
    def sample_code_chunks(self) -> List[CodeChunk]:
        """Create sample code chunks."""
        return [
            CodeChunk(
                content="def hello_world():\n    print('Hello, World!')",
                chunk_type=ChunkType.FUNCTION,
                name="hello_world",
                file_path="test.py",
                start_line=1,
                end_line=2,
                content_hash="hash1",
                language="python",
            ),
            CodeChunk(
                content="class TestClass:\n    def __init__(self):\n        pass",
                chunk_type=ChunkType.CLASS,
                name="TestClass",
                file_path="test.py",
                start_line=4,
                end_line=6,
                content_hash="hash2",
                language="python",
            ),
        ]

    @pytest.fixture
    def sample_search_results(self) -> List[Row]:
        """Create sample search results."""
        return [
            Row(
                id="result1",
                **{"$dist": 0.1},  # Distance of 0.1 = similarity of 0.9
                file_path="src/utils.py",
                start_line=10,
                end_line=12,
                content="def similar_function():\n    return 'similar'",
                content_hash="hash123",
                chunk_type="function",
            ),
            Row(
                id="result2",
                **{"$dist": 0.3},  # Distance of 0.3 = similarity of 0.7
                file_path="src/helpers.py",
                start_line=20,
                end_line=25,
                content="class SimilarClass:\n    def method(self):\n        pass",
                content_hash="hash456",
                chunk_type="class",
            ),
        ]

    @pytest_asyncio.fixture
    async def service(self) -> VectorModeToolsService:
        """Create a VectorModeToolsService instance."""
        return VectorModeToolsService()

    @pytest_asyncio.fixture
    async def initialized_service(
        self,
        service: VectorModeToolsService,
        mock_voyage_client: Mock,
        mock_turbopuffer_client: Mock,
        mock_config: VectorConfig,
    ) -> VectorModeToolsService:
        """Create an initialized VectorModeToolsService."""
        with (
            patch(
                "mcp_code_indexer.vector_mode.services.vector_mode_tools_service.is_vector_mode_available"
            ) as mock_available,
            patch(
                "mcp_code_indexer.vector_mode.services.vector_mode_tools_service.load_vector_config"
            ) as mock_load_config,
            patch(
                "mcp_code_indexer.vector_mode.services.vector_mode_tools_service.VoyageClient"
            ) as mock_voyage_class,
            patch(
                "mcp_code_indexer.vector_mode.services.vector_mode_tools_service.TurbopufferClient"
            ) as mock_turbo_class,
            patch(
                "mcp_code_indexer.vector_mode.services.vector_mode_tools_service.ASTChunker"
            ) as mock_ast_chunker_class,
            patch(
                "mcp_code_indexer.vector_mode.services.vector_mode_tools_service.EmbeddingService"
            ) as mock_embedding_service_class,
            patch(
                "mcp_code_indexer.vector_mode.services.vector_mode_tools_service.VectorStorageService"
            ) as mock_vector_storage_class,
        ):

            # Configure mocks
            mock_available.return_value = True
            mock_load_config.return_value = mock_config
            mock_voyage_class.return_value = mock_voyage_client
            mock_turbo_class.return_value = mock_turbopuffer_client

            # Mock services
            mock_ast_chunker = Mock(spec=ASTChunker)
            mock_embedding_service = Mock()
            mock_vector_storage = Mock()

            mock_ast_chunker_class.return_value = mock_ast_chunker
            mock_embedding_service_class.return_value = mock_embedding_service
            mock_vector_storage_class.return_value = mock_vector_storage

            # Initialize the service
            service._ensure_initialized()

            # Store mocked services for access in tests
            service._mock_ast_chunker = mock_ast_chunker
            service._mock_embedding_service = mock_embedding_service
            service._mock_vector_storage = mock_vector_storage

            return service

    @pytest.mark.asyncio
    async def test_input_validation_both_code_snippet_and_file_path(
        self, service: VectorModeToolsService
    ) -> None:
        """Test that providing both code_snippet and file_path raises ValueError."""
        with pytest.raises(
            ValueError, match="Cannot specify both code_snippet and file_path"
        ):
            await service.find_similar_code(
                project_name="test_project",
                folder_path="/test/path",
                code_snippet="def test():\n    pass",
                file_path="test.py",
                line_start=1,
                line_end=2,
            )

    @pytest.mark.asyncio
    async def test_input_validation_neither_code_snippet_nor_file_path(
        self, service: VectorModeToolsService
    ) -> None:
        """Test that providing neither code_snippet nor file_path raises ValueError."""
        with pytest.raises(
            ValueError, match="Must specify either code_snippet or file_path"
        ):
            await service.find_similar_code(
                project_name="test_project",
                folder_path="/test/path",
            )

    @pytest.mark.asyncio
    async def test_input_validation_file_path_without_line_range(
        self, service: VectorModeToolsService
    ) -> None:
        """Test that providing file_path without line_start/line_end raises ValueError."""
        with pytest.raises(
            ValueError, match="file_path requires both line_start and line_end"
        ):
            await service.find_similar_code(
                project_name="test_project",
                folder_path="/test/path",
                file_path="test.py",
            )

        with pytest.raises(
            ValueError, match="file_path requires both line_start and line_end"
        ):
            await service.find_similar_code(
                project_name="test_project",
                folder_path="/test/path",
                file_path="test.py",
                line_start=1,
            )

    @pytest.mark.asyncio
    async def test_find_similar_code_with_code_snippet_success(
        self,
        initialized_service: VectorModeToolsService,
        sample_code_chunks: List[CodeChunk],
        sample_search_results: List[Row],
    ) -> None:
        """Test successful find_similar_code with code snippet."""
        service = initialized_service

        # Mock the chunking
        service._mock_ast_chunker.chunk_content.return_value = sample_code_chunks

        # Mock embedding generation
        sample_embeddings = [
            [0.1] * 1024,
            [0.2] * 1024,
        ]  # Mock embeddings for each chunk
        service._mock_embedding_service.generate_embeddings_for_chunks = AsyncMock(
            return_value=sample_embeddings
        )

        # Mock vector storage search - return results for each query chunk
        service._mock_vector_storage.search_similar_chunks = AsyncMock(
            side_effect=[
                sample_search_results[:1],
                sample_search_results[1:],
            ]  # Different results per chunk
        )

        result = await service.find_similar_code(
            project_name="test_project",
            folder_path="/test/path",
            code_snippet="def hello():\n    print('hi')",
            similarity_threshold=0.6,
            max_results=5,
        )

        # Verify the result structure
        assert "results" in result
        assert "total_results" in result
        assert "query_info" in result

        # Check query info
        query_info = result["query_info"]
        assert query_info["source"] == "code_snippet"
        assert query_info["chunks_generated"] == 2
        assert query_info["similarity_threshold"] == 0.6
        assert query_info["max_results"] == 5

        # Verify results are filtered and formatted correctly
        results = result["results"]
        assert len(results) == 2  # Two unique results

        # Check first result (higher similarity)
        first_result = results[0]
        assert first_result["score"] == 0.9  # 1 - 0.1 distance
        assert first_result["file_name"] == "utils.py"
        assert first_result["start_line"] == 10
        assert first_result["end_line"] == 12
        assert "content" in first_result
        assert "metadata" in first_result

        # Check second result (lower similarity but above threshold)
        second_result = results[1]
        assert second_result["score"] == 0.7  # 1 - 0.3 distance
        assert second_result["file_name"] == "helpers.py"

        # Verify service calls
        service._mock_ast_chunker.chunk_content.assert_called_once_with(
            content="def hello():\n    print('hi')",
            file_path="code_snippet",
            language="python",
        )

        service._mock_embedding_service.generate_embeddings_for_chunks.assert_called_once()
        assert service._mock_vector_storage.search_similar_chunks.call_count == 2

    @pytest.mark.asyncio
    async def test_find_similar_code_with_file_section_success(
        self,
        initialized_service: VectorModeToolsService,
        tmp_path: Path,
        sample_code_chunks: List[CodeChunk],
    ) -> None:
        """Test successful find_similar_code with file section."""
        service = initialized_service

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\ndef hello():\n    print('test')\nline4\n")

        # Mock chunking and search
        service._mock_ast_chunker.chunk_content.return_value = sample_code_chunks
        sample_embeddings = [[0.1] * 1024]
        service._mock_embedding_service.generate_embeddings_for_chunks = AsyncMock(
            return_value=sample_embeddings
        )
        service._mock_vector_storage.search_similar_chunks = AsyncMock(return_value=[])

        result = await service.find_similar_code(
            project_name="test_project",
            folder_path=str(tmp_path),
            file_path="test.py",
            line_start=2,
            line_end=3,
        )

        # Verify the result
        assert result["query_info"]["source"] == "test.py:2-3"

        # Verify file reading worked correctly
        # The chunker should be called with the extracted file section
        call_args = service._mock_ast_chunker.chunk_content.call_args
        assert call_args[1]["content"] == "def hello():\n    print('test')"
        assert call_args[1]["file_path"] == "test.py:2-3"

    @pytest.mark.asyncio
    async def test_find_similar_code_no_chunks_generated(
        self, initialized_service: VectorModeToolsService
    ) -> None:
        """Test handling when no chunks are generated from input code."""
        service = initialized_service

        # Mock empty chunks
        service._mock_ast_chunker.chunk_content.return_value = []

        result = await service.find_similar_code(
            project_name="test_project",
            folder_path="/test/path",
            code_snippet="# just a comment",
        )

        assert result["results"] == []
        assert result["total_results"] == 0
        assert result["query_info"]["chunks_generated"] == 0
        assert "No valid code chunks could be generated" in result["message"]

    @pytest.mark.asyncio
    async def test_find_similar_code_filters_by_similarity_threshold(
        self,
        initialized_service: VectorModeToolsService,
        sample_code_chunks: List[CodeChunk],
    ) -> None:
        """Test that results below similarity threshold are filtered out."""
        service = initialized_service

        # Mock chunking
        service._mock_ast_chunker.chunk_content.return_value = sample_code_chunks[
            :1
        ]  # Single chunk
        service._mock_embedding_service.generate_embeddings_for_chunks = AsyncMock(
            return_value=[[0.1] * 1024]
        )

        # Create results with different similarity scores
        low_similarity_results = [
            Row(
                id="low_sim",
                **{"$dist": 0.9},  # Distance 0.9 = similarity 0.1 (below 0.8 threshold)
                file_path="low_sim.py",
                start_line=1,
                end_line=2,
                content="low similarity content",
                content_hash="low_hash",
                chunk_type="function",
            ),
            Row(
                id="high_sim",
                **{"$dist": 0.1},  # Distance 0.1 = similarity 0.9 (above 0.8 threshold)
                file_path="high_sim.py",
                start_line=1,
                end_line=2,
                content="high similarity content",
                content_hash="high_hash",
                chunk_type="function",
            ),
        ]

        service._mock_vector_storage.search_similar_chunks = AsyncMock(
            return_value=low_similarity_results
        )

        result = await service.find_similar_code(
            project_name="test_project",
            folder_path="/test/path",
            code_snippet="def test():\n    pass",
            similarity_threshold=0.8,
        )

        # Only the high similarity result should be returned
        assert len(result["results"]) == 1
        assert result["results"][0]["score"] == 0.9
        assert result["results"][0]["file_name"] == "high_sim.py"

    @pytest.mark.asyncio
    async def test_find_similar_code_deduplicates_results(
        self,
        initialized_service: VectorModeToolsService,
        sample_code_chunks: List[CodeChunk],
    ) -> None:
        """Test that duplicate results are properly deduplicated."""
        service = initialized_service

        # Mock chunking
        service._mock_ast_chunker.chunk_content.return_value = sample_code_chunks
        service._mock_embedding_service.generate_embeddings_for_chunks = AsyncMock(
            return_value=[[0.1] * 1024, [0.2] * 1024]
        )

        # Create duplicate results (same file_path + content_hash)
        duplicate_results = [
            Row(
                id="dup1",
                **{"$dist": 0.1},
                file_path="same_file.py",
                start_line=1,
                end_line=2,
                content="same content",
                content_hash="same_hash",
                chunk_type="function",
            ),
            Row(
                id="dup2",
                **{"$dist": 0.2},  # Different similarity but same file+hash
                file_path="same_file.py",
                start_line=1,
                end_line=2,
                content="same content",
                content_hash="same_hash",
                chunk_type="function",
            ),
        ]

        service._mock_vector_storage.search_similar_chunks = AsyncMock(
            side_effect=[duplicate_results[:1], duplicate_results[1:]]
        )

        result = await service.find_similar_code(
            project_name="test_project",
            folder_path="/test/path",
            code_snippet="def test():\n    pass",
        )

        # Only one result should remain after deduplication
        assert len(result["results"]) == 1
        assert (
            result["results"][0]["score"] == 0.9
        )  # Higher score (lower distance) kept

    @pytest.mark.asyncio
    async def test_find_similar_code_respects_max_results(
        self,
        initialized_service: VectorModeToolsService,
        sample_code_chunks: List[CodeChunk],
    ) -> None:
        """Test that max_results parameter limits the number of returned results."""
        service = initialized_service

        # Mock chunking
        service._mock_ast_chunker.chunk_content.return_value = sample_code_chunks[:1]
        service._mock_embedding_service.generate_embeddings_for_chunks = AsyncMock(
            return_value=[[0.1] * 1024]
        )

        # Create more results than max_results
        many_results = []
        for i in range(5):
            many_results.append(
                Row(
                    id=f"result{i}",
                    **{"$dist": 0.1},
                    file_path=f"file{i}.py",
                    start_line=1,
                    end_line=2,
                    content=f"content {i}",
                    content_hash=f"hash{i}",
                    chunk_type="function",
                )
            )

        service._mock_vector_storage.search_similar_chunks = AsyncMock(
            return_value=many_results
        )

        result = await service.find_similar_code(
            project_name="test_project",
            folder_path="/test/path",
            code_snippet="def test():\n    pass",
            max_results=2,
        )

        # Should be limited to max_results
        assert len(result["results"]) == 2
        assert result["query_info"]["max_results"] == 2

    @pytest.mark.asyncio
    async def test_find_similar_code_uses_config_defaults(
        self,
        initialized_service: VectorModeToolsService,
        sample_code_chunks: List[CodeChunk],
    ) -> None:
        """Test that config defaults are used when parameters are not specified."""
        service = initialized_service

        # Mock chunking
        service._mock_ast_chunker.chunk_content.return_value = sample_code_chunks[:1]
        service._mock_embedding_service.generate_embeddings_for_chunks = AsyncMock(
            return_value=[[0.1] * 1024]
        )
        service._mock_vector_storage.search_similar_chunks = AsyncMock(return_value=[])

        result = await service.find_similar_code(
            project_name="test_project",
            folder_path="/test/path",
            code_snippet="def test():\n    pass",
        )

        # Should use config defaults
        query_info = result["query_info"]
        assert query_info["similarity_threshold"] == 0.8  # From mock_config
        assert query_info["max_results"] == 10  # From mock_config

    @pytest.mark.asyncio
    async def test_read_file_section_file_not_found(
        self, initialized_service: VectorModeToolsService
    ) -> None:
        """Test file reading error handling for non-existent file."""
        service = initialized_service

        with pytest.raises(ValueError, match="File not found"):
            await service._read_file_section(
                folder_path="/test/path",
                file_path="nonexistent.py",
                line_start=1,
                line_end=2,
            )

    @pytest.mark.asyncio
    async def test_read_file_section_invalid_line_numbers(
        self, initialized_service: VectorModeToolsService, tmp_path: Path
    ) -> None:
        """Test file reading with invalid line numbers."""
        service = initialized_service

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\n")

        # Test line_start < 1
        with pytest.raises(ValueError, match="Line numbers must be >= 1"):
            await service._read_file_section(
                folder_path=str(tmp_path),
                file_path="test.py",
                line_start=0,
                line_end=2,
            )

        # Test line_start beyond file length
        with pytest.raises(ValueError, match="line_start 5 exceeds file length"):
            await service._read_file_section(
                folder_path=str(tmp_path),
                file_path="test.py",
                line_start=5,
                line_end=6,
            )

    @pytest.mark.asyncio
    async def test_read_file_section_clamps_line_end(
        self, initialized_service: VectorModeToolsService, tmp_path: Path
    ) -> None:
        """Test that line_end is clamped to file length."""
        service = initialized_service

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        content = await service._read_file_section(
            folder_path=str(tmp_path),
            file_path="test.py",
            line_start=2,
            line_end=10,  # Beyond file length
        )

        assert content == "line2\nline3"

    @pytest.mark.asyncio
    async def test_read_file_section_security_check(
        self, initialized_service: VectorModeToolsService, tmp_path: Path
    ) -> None:
        """Test security check prevents reading files outside project folder."""
        service = initialized_service

        with pytest.raises(ValueError, match="is outside project folder"):
            await service._read_file_section(
                folder_path=str(tmp_path),
                file_path="../outside.py",
                line_start=1,
                line_end=2,
            )

    @pytest.mark.asyncio
    async def test_chunk_code_error_handling(
        self, initialized_service: VectorModeToolsService
    ) -> None:
        """Test error handling in code chunking."""
        service = initialized_service

        # Mock chunking to raise an exception
        service._mock_ast_chunker.chunk_content.side_effect = Exception(
            "Chunking failed"
        )

        with pytest.raises(RuntimeError, match="Code chunking failed"):
            await service._chunk_code("invalid code", "test.py")

    @pytest.mark.asyncio
    async def test_find_similar_code_search_error_handling(
        self,
        initialized_service: VectorModeToolsService,
        sample_code_chunks: List[CodeChunk],
    ) -> None:
        """Test error handling when vector search fails."""
        service = initialized_service

        # Mock successful chunking but failed search
        service._mock_ast_chunker.chunk_content.return_value = sample_code_chunks[:1]
        service._mock_embedding_service.generate_embeddings_for_chunks = AsyncMock(
            return_value=[[0.1] * 1024]
        )
        service._mock_vector_storage.search_similar_chunks = AsyncMock(
            side_effect=Exception("Search failed")
        )

        with pytest.raises(RuntimeError, match="Similarity search failed"):
            await service.find_similar_code(
                project_name="test_project",
                folder_path="/test/path",
                code_snippet="def test():\n    pass",
            )

    @pytest.mark.asyncio
    async def test_ensure_initialized_vector_mode_not_available(
        self, service: VectorModeToolsService
    ) -> None:
        """Test initialization failure when vector mode is not available."""
        with patch(
            "mcp_code_indexer.vector_mode.services.vector_mode_tools_service.is_vector_mode_available"
        ) as mock_available:
            mock_available.return_value = False

            with pytest.raises(
                RuntimeError, match="Vector mode dependencies are not available"
            ):
                service._ensure_initialized()

    @pytest.mark.asyncio
    async def test_filter_and_deduplicate_results_handles_none_rows(
        self, initialized_service: VectorModeToolsService
    ) -> None:
        """Test that filter_and_deduplicate_results handles None values in results."""
        service = initialized_service

        # Include None values in results
        results_with_none = [
            None,
            Row(
                id="valid",
                **{"$dist": 0.1},
                file_path="test.py",
                start_line=1,
                end_line=2,
                content="test content",
                content_hash="hash123",
                chunk_type="function",
            ),
            None,
        ]

        filtered = service._filter_and_deduplicate_results(
            results_with_none, similarity_threshold=0.5, max_results=10
        )

        # Should only return the valid result
        assert len(filtered) == 1
        assert filtered[0]["file_name"] == "test.py"

    @pytest.mark.asyncio
    async def test_filter_and_deduplicate_results_sorts_by_similarity(
        self, initialized_service: VectorModeToolsService
    ) -> None:
        """Test that results are sorted by similarity score in descending order."""
        service = initialized_service

        results = [
            Row(
                id="low",
                **{"$dist": 0.4},  # Similarity 0.6
                file_path="low.py",
                start_line=1,
                end_line=2,
                content="low similarity",
                content_hash="hash1",
                chunk_type="function",
            ),
            Row(
                id="high",
                **{"$dist": 0.1},  # Similarity 0.9
                file_path="high.py",
                start_line=1,
                end_line=2,
                content="high similarity",
                content_hash="hash2",
                chunk_type="function",
            ),
            Row(
                id="medium",
                **{"$dist": 0.2},  # Similarity 0.8
                file_path="medium.py",
                start_line=1,
                end_line=2,
                content="medium similarity",
                content_hash="hash3",
                chunk_type="function",
            ),
        ]

        filtered = service._filter_and_deduplicate_results(
            results, similarity_threshold=0.5, max_results=10
        )

        # Should be sorted by similarity (highest first)
        assert len(filtered) == 3
        assert filtered[0]["score"] == 0.9  # high similarity
        assert filtered[1]["score"] == 0.8  # medium similarity
        assert filtered[2]["score"] == 0.6  # low similarity
