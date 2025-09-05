"""
Unit tests for VectorDaemon monitored_projects management and file change processing.

This module tests the VectorDaemon._get_project_monitoring_status method and
_process_file_change_task method including:
- Monitoring status when no projects exist
- Monitoring new vector-enabled projects
- Unmonitoring projects when vector mode is disabled
- Handling projects without aliases
- Processing file changes with chunking
- Error handling for file processing
"""

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch, AsyncMock

import pytest
import pytest_asyncio

from mcp_code_indexer.database.database import DatabaseManager
from mcp_code_indexer.database.models import Project
from mcp_code_indexer.vector_mode.daemon import VectorDaemon
from mcp_code_indexer.vector_mode.monitoring.change_detector import (
    ChangeType,
    FileChange,
)

from mcp_code_indexer.vector_mode.config import VectorConfig
from mcp_code_indexer.vector_mode.chunking.ast_chunker import ASTChunker


class TestVectorDaemonMonitoringStatus:
    """Test VectorDaemon monitored_projects management."""

    @pytest.fixture
    async def vector_daemon(self, db_manager: DatabaseManager) -> VectorDaemon:
        """Create a VectorDaemon for testing."""
        config = VectorConfig(
            voyage_api_key="test-voyage-key",
            turbopuffer_api_key="test-turbopuffer-key"
        )
        cache_dir = Path("/tmp/test_cache")
        
        # Mock API validation to avoid real API calls during testing
        with patch('mcp_code_indexer.vector_mode.providers.voyage_client.VoyageClient.validate_api_access'), \
             patch('mcp_code_indexer.vector_mode.providers.turbopuffer_client.TurbopufferClient.validate_api_access'):
            daemon = VectorDaemon(config, db_manager, cache_dir)
        return daemon

    async def test_get_project_monitoring_status_empty(
        self, db_manager: DatabaseManager, vector_daemon: VectorDaemon
    ) -> None:
        """Test monitoring status when no projects exist."""
        status = await vector_daemon._get_project_monitoring_status()

        assert status["monitored"] == []
        assert status["unmonitored"] == []
        assert len(vector_daemon.monitored_projects) == 0

    async def test_get_project_monitoring_status_new_project(
        self,
        db_manager: DatabaseManager,
        sample_project: Project,
        vector_daemon: VectorDaemon,
    ) -> None:
        """Test monitoring status with a new vector-enabled project."""
        # Enable vector mode for the project
        await db_manager.set_project_vector_mode(sample_project.id, True)

        # Project already has aliases from fixture, so it should be monitorable
        status = await vector_daemon._get_project_monitoring_status()

        # Should be marked for monitoring
        assert len(status["monitored"]) == 1
        assert status["monitored"][0].name == sample_project.name
        assert len(status["unmonitored"]) == 0

    async def test_get_project_monitoring_status_unmonitor_project(
        self,
        db_manager: DatabaseManager,
        sample_project: Project,
        vector_daemon: VectorDaemon,
    ) -> None:
        """Test monitoring status when project should be unmonitored."""
        # Setup: enable vector mode (project already has aliases from fixture)
        await db_manager.set_project_vector_mode(sample_project.id, True)

        # Simulate project being monitored
        vector_daemon.monitored_projects.add(sample_project.name)

        # Now disable vector mode
        await db_manager.set_project_vector_mode(sample_project.id, False)

        status = await vector_daemon._get_project_monitoring_status()

        # Should be marked for unmonitoring
        assert len(status["monitored"]) == 0
        assert len(status["unmonitored"]) == 1
        assert status["unmonitored"][0].name == sample_project.name

    async def test_get_project_monitoring_status_no_aliases(
        self, db_manager: DatabaseManager, vector_daemon: VectorDaemon
    ) -> None:
        """Test that projects without aliases are not monitored."""
        # Create a project without aliases

        project_no_aliases = Project(
            id="no_aliases_project",
            name="no aliases project",
            aliases=[],  # No aliases
            created=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            vector_mode=False,
        )
        await db_manager.create_project(project_no_aliases)

        # Enable vector mode
        await db_manager.set_project_vector_mode(project_no_aliases.id, True)

        status = await vector_daemon._get_project_monitoring_status()

        # Should not be monitored due to missing aliases
        assert len(status["monitored"]) == 0
        assert len(status["unmonitored"]) == 0


class TestVectorDaemonFileChangeProcessing:
    """Test VectorDaemon file change processing functionality."""

    @pytest.fixture
    async def vector_daemon(
        self, db_manager: DatabaseManager, tmp_path: Path
    ) -> VectorDaemon:
        """Create a VectorDaemon for testing.""" 
        from mcp_code_indexer.vector_mode.config import VectorConfig

        config = VectorConfig(
            voyage_api_key="test-voyage-key",
            turbopuffer_api_key="test-turbopuffer-key",
            batch_size=32
        )
        cache_dir = tmp_path / "test_cache"
        
        # Mock API validation to avoid real API calls during testing
        with patch('mcp_code_indexer.vector_mode.providers.voyage_client.VoyageClient.validate_api_access'), \
             patch('mcp_code_indexer.vector_mode.providers.turbopuffer_client.TurbopufferClient.validate_api_access'):
            daemon = VectorDaemon(config, db_manager, cache_dir)
        # Initialize stats
        daemon.stats = {
            "files_processed": 0, 
            "errors_count": 0, 
            "last_activity": 0.0,
            "embeddings_generated": 0
        }
        return daemon

    @pytest.fixture
    def sample_python_file(self, tmp_path: Path) -> Path:
        """Create a sample Python file for testing."""
        file_path = tmp_path / "test_file.py"
        content = '''"""Sample Python file for testing."""

def hello_world():
    """Print hello world message."""
    print("Hello, World!")
    return "Hello, World!"

class TestClass:
    """A simple test class."""
    
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    hello_world()
'''
        file_path.write_text(content)
        return file_path

    @pytest.fixture
    def sample_binary_file(self, tmp_path: Path) -> Path:
        """Create a sample binary file for testing."""
        file_path = tmp_path / "test_file.bin"
        file_path.write_bytes(b"\x00\x01\x02\x03\x04\x05")
        return file_path

    async def test_process_file_change_task_created_file(
        self, vector_daemon: VectorDaemon, sample_python_file: Path
    ) -> None:
        """Test processing a CREATED file change."""
        change = FileChange(
            path=str(sample_python_file),
            change_type=ChangeType.CREATED,
            timestamp=datetime.utcnow(),
            size=len(sample_python_file.read_text()),
        )

        task = {"project_name": "test_project", "change": change}

        worker_id = "worker_1"

        # Mock EmbeddingService, VectorStorageService, and ASTChunker
        with patch("mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file") as mock_chunk_file, \
             patch.object(vector_daemon, '_embedding_service') as mock_service, \
             patch.object(vector_daemon, '_vector_storage_service') as mock_vector_storage_service:
            
            # Mock return value to ensure processing continues
            mock_chunk_file.return_value = [
                type('Chunk', (), {
                    'chunk_type': type('ChunkType', (), {'value': 'function'}),
                    'redacted': False,
                    'name': 'test_function',
                    'start_line': 1,
                    'end_line': 5,
                    'content': 'def test_function():\n    pass'
                })
            ]
            
            # Mock embeddings generation
            mock_service.generate_embeddings_for_chunks = AsyncMock(return_value=[[0.1] * 1536])
            
            # Mock vector storage
            mock_vector_storage_service.store_embeddings = AsyncMock()

            await vector_daemon._process_file_change_task(task, worker_id)

            # Verify stats were updated
            assert vector_daemon.stats["files_processed"] == 1
            assert vector_daemon.stats["errors_count"] == 0

            # Verify ASTChunker.chunk_file was called with the correct file path
            mock_chunk_file.assert_called_once_with(str(sample_python_file))

    async def test_process_file_change_task_modified_file(
        self, vector_daemon: VectorDaemon, sample_python_file: Path
    ) -> None:
        """Test processing a MODIFIED file change."""
        change = FileChange(
            path=str(sample_python_file),
            change_type=ChangeType.MODIFIED,
            timestamp=datetime.utcnow(),
            size=len(sample_python_file.read_text()),
        )

        task = {"project_name": "test_project", "change": change}

        worker_id = "worker_2"

        # Mock EmbeddingService, VectorStorageService, and ASTChunker
        with patch("mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file") as mock_chunk_file, \
             patch.object(vector_daemon, '_embedding_service') as mock_service, \
             patch.object(vector_daemon, '_vector_storage_service') as mock_vector_storage_service:
            # Make chunk_file return some test chunks
            mock_chunk_file.return_value = [
                type(
                    "Chunk",
                    (),
                    {
                        "chunk_type": type("ChunkType", (), {"value": "class"}),
                        "redacted": False,
                        "name": "TestClass",
                        "start_line": 1,
                        "end_line": 10,
                        "content": "class TestClass:\n    pass",
                    },
                )
            ]
            
            # Mock embeddings generation
            mock_service.generate_embeddings_for_chunks = AsyncMock(return_value=[[0.1] * 1536])
            
            # Mock vector storage
            mock_vector_storage_service.store_embeddings = AsyncMock()

            await vector_daemon._process_file_change_task(task, worker_id)

            # Verify stats were updated
            assert vector_daemon.stats["files_processed"] == 1
            assert vector_daemon.stats["errors_count"] == 0

            # Verify chunking occurred
            mock_chunk_file.assert_called_once_with(str(sample_python_file))

    async def test_process_file_change_task_deleted_file(
        self, vector_daemon: VectorDaemon, sample_python_file: Path
    ) -> None:
        """Test processing a DELETED file change (should delete vectors from database)."""
        change = FileChange(
            path=str(sample_python_file),
            change_type=ChangeType.DELETED,
            timestamp=datetime.utcnow(),
            size=0,
        )

        task = {"project_name": "test_project", "change": change}

        worker_id = "worker_3"

        # Mock the vector storage service's delete_vectors_for_file method
        with patch.object(
            vector_daemon._vector_storage_service,
            "delete_vectors_for_file"
        ) as mock_delete_vectors, patch(
            "mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file"
        ) as mock_chunk_file:

            await vector_daemon._process_file_change_task(task, worker_id)

            # Verify no file processing stats changed (deleted files don't count as processed)
            assert vector_daemon.stats["files_processed"] == 0
            assert vector_daemon.stats["errors_count"] == 0

            # Verify ASTChunker.chunk_file was NOT called (deleted files don't need chunking)
            mock_chunk_file.assert_not_called()
            
            # Verify delete_vectors_for_file was called to clean up vectors
            mock_delete_vectors.assert_called_once_with("test_project", str(sample_python_file))

    async def test_process_file_change_task_deleted_file_error(
        self, vector_daemon: VectorDaemon, sample_python_file: Path
    ) -> None:
        """Test processing a DELETED file change when vector deletion fails."""
        change = FileChange(
            path=str(sample_python_file),
            change_type=ChangeType.DELETED,
            timestamp=datetime.utcnow(),
            size=0,
        )

        task = {"project_name": "test_project", "change": change}

        worker_id = "worker_3"

        # Mock the vector storage service's delete_vectors_for_file method to fail
        with patch.object(
            vector_daemon._vector_storage_service,
            "delete_vectors_for_file"
        ) as mock_delete_vectors, patch(
            "mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file"
        ) as mock_chunk_file:

            # Make deletion fail
            mock_delete_vectors.side_effect = RuntimeError("Vector deletion failed")

            await vector_daemon._process_file_change_task(task, worker_id)

            # Verify no file processing stats changed (error handling is graceful)
            assert vector_daemon.stats["files_processed"] == 0
            assert vector_daemon.stats["errors_count"] == 0

            # Verify ASTChunker.chunk_file was NOT called (deleted files don't need chunking)
            mock_chunk_file.assert_not_called()
            
            # Verify delete_vectors_for_file was called
            mock_delete_vectors.assert_called_once_with("test_project", str(sample_python_file))

    async def test_process_file_change_task_binary_file(
        self, vector_daemon: VectorDaemon, sample_binary_file: Path
    ) -> None:
        """Test processing a binary file (ASTChunker handles with errors='ignore')."""
        change = FileChange(
            path=str(sample_binary_file),
            change_type=ChangeType.CREATED,
            timestamp=datetime.utcnow(),
            size=6,
        )

        task = {"project_name": "test_project", "change": change}

        worker_id = "worker_4"

        # Mock EmbeddingService, VectorStorageService, and ASTChunker
        with patch("mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file") as mock_chunk_file, \
             patch.object(vector_daemon, '_embedding_service') as mock_service, \
             patch.object(vector_daemon, '_vector_storage_service') as mock_vector_storage_service:
            # Make chunk_file return some test chunks (binary files can produce chunks with errors='ignore')
            mock_chunk_file.return_value = [
                type(
                    "Chunk",
                    (),
                    {
                        "chunk_type": type("ChunkType", (), {"value": "text"}),
                        "redacted": False,
                        "name": None,
                        "start_line": 1,
                        "end_line": 1,
                        "content": "binary content",
                    },
                )
            ]
            
            # Mock embeddings generation
            mock_service.generate_embeddings_for_chunks = AsyncMock(return_value=[[0.1] * 1536])
            
            # Mock vector storage
            mock_vector_storage_service.store_embeddings = AsyncMock()

            await vector_daemon._process_file_change_task(task, worker_id)

            # ASTChunker processes binary files using errors='ignore'
            # This may produce chunks, so files_processed gets incremented
            assert vector_daemon.stats["files_processed"] == 1
            assert vector_daemon.stats["errors_count"] == 0

            # Verify chunking occurred
            mock_chunk_file.assert_called_once_with(str(sample_binary_file))

    async def test_process_file_change_task_nonexistent_file(
        self, vector_daemon: VectorDaemon, tmp_path: Path
    ) -> None:
        """Test processing a non-existent file (should handle gracefully)."""
        nonexistent_file = tmp_path / "nonexistent.py"

        change = FileChange(
            path=str(nonexistent_file),
            change_type=ChangeType.CREATED,
            timestamp=datetime.utcnow(),
            size=100,
        )

        task = {"project_name": "test_project", "change": change}

        worker_id = "worker_5"

        # Spy on ASTChunker.chunk_file to verify it was called but returned no chunks
        with patch(
            "mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file"
        ) as mock_chunk_file:
            # Make chunk_file return empty list for nonexistent file
            mock_chunk_file.return_value = []

            await vector_daemon._process_file_change_task(task, worker_id)

            # ASTChunker handles nonexistent files gracefully by returning empty list
            # No chunks produced, so no files_processed increment
            assert vector_daemon.stats["files_processed"] == 0
            assert vector_daemon.stats["errors_count"] == 0

            # Verify ASTChunker.chunk_file was called with the nonexistent file
            mock_chunk_file.assert_called_once_with(str(nonexistent_file))

    async def test_process_file_change_task_chunking_details(
        self, vector_daemon: VectorDaemon, sample_python_file: Path
    ) -> None:
        """Test that chunking details are logged correctly."""
        change = FileChange(
            path=str(sample_python_file),
            change_type=ChangeType.CREATED,
            timestamp=datetime.utcnow(),
            size=len(sample_python_file.read_text()),
        )

        task = {"project_name": "test_project", "change": change}

        worker_id = "worker_6"

        # Mock EmbeddingService, VectorStorageService, and ASTChunker
        with patch("mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file") as mock_chunk_file, \
             patch.object(vector_daemon, '_embedding_service') as mock_service, \
             patch.object(vector_daemon, '_vector_storage_service') as mock_vector_storage_service:
            # Make chunk_file return multiple chunks with different types for detailed testing
            mock_chunks = [
                type(
                    "Chunk",
                    (),
                    {
                        "chunk_type": type("ChunkType", (), {"value": "function"}),
                        "redacted": False,
                        "name": "test_function",
                        "start_line": 1,
                        "end_line": 5,
                        "content": "def test_function():\n    pass",
                    },
                ),
                type(
                    "Chunk",
                    (),
                    {
                        "chunk_type": type("ChunkType", (), {"value": "class"}),
                        "redacted": True,
                        "name": "TestClass",
                        "start_line": 7,
                        "end_line": 15,
                        "content": "class TestClass:\n    def __init__(self):\n        pass",
                    },
                ),
            ]
            mock_chunk_file.return_value = mock_chunks
            
            # Mock embeddings generation
            mock_service.generate_embeddings_for_chunks = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
            
            # Mock vector storage
            mock_vector_storage_service.store_embeddings = AsyncMock()

            await vector_daemon._process_file_change_task(task, worker_id)

            # Verify chunking occurred
            mock_chunk_file.assert_called_once_with(str(sample_python_file))

            # Verify that chunks were processed (stats updated)
            assert vector_daemon.stats["files_processed"] == 1
            assert vector_daemon.stats["errors_count"] == 0

    async def test_process_file_change_task_stats_update(
        self, vector_daemon: VectorDaemon, sample_python_file: Path
    ) -> None:
        """Test that daemon stats are updated correctly."""
        initial_files_processed = vector_daemon.stats["files_processed"]
        initial_last_activity = vector_daemon.stats["last_activity"]

        change = FileChange(
            path=str(sample_python_file),
            change_type=ChangeType.MODIFIED,
            timestamp=datetime.utcnow(),
            size=len(sample_python_file.read_text()),
        )

        task = {"project_name": "test_project", "change": change}

        # Mock EmbeddingService and VectorStorageService for stats test
        with patch.object(vector_daemon, '_embedding_service') as mock_service, \
             patch.object(vector_daemon, '_vector_storage_service') as mock_vector_storage_service:
            mock_service.generate_embeddings_for_chunks = AsyncMock(return_value=[[0.1] * 1536])
            mock_vector_storage_service.store_embeddings = AsyncMock()
            
            await vector_daemon._process_file_change_task(task, "worker_7")

            # Verify stats were updated
            assert vector_daemon.stats["files_processed"] == initial_files_processed + 1
            assert vector_daemon.stats["last_activity"] > initial_last_activity

    async def test_process_file_change_task_empty_file(
        self, vector_daemon: VectorDaemon, tmp_path: Path
    ) -> None:
        """Test processing an empty file."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        change = FileChange(
            path=str(empty_file),
            change_type=ChangeType.CREATED,
            timestamp=datetime.utcnow(),
            size=0,
        )

        task = {"project_name": "test_project", "change": change}

        worker_id = "worker_8"

        # Spy on ASTChunker.chunk_file to verify it was called but returned no chunks for empty file
        with patch(
            "mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file"
        ) as mock_chunk_file:
            # Make chunk_file return empty list for empty file
            mock_chunk_file.return_value = []

            await vector_daemon._process_file_change_task(task, worker_id)

            # Empty files should be processed but result in 0 chunks
            # This depends on ASTChunker behavior - it should skip empty files
            # So no processing should occur
            assert vector_daemon.stats["files_processed"] == 0
            assert vector_daemon.stats["errors_count"] == 0

            # Verify ASTChunker.chunk_file was called with the empty file
            mock_chunk_file.assert_called_once_with(str(empty_file))


class TestVectorDaemonEmbeddingGeneration:
    """Test VectorDaemon embedding generation functionality."""

    @pytest.fixture
    async def vector_daemon_with_voyage(
        self, db_manager: DatabaseManager, tmp_path: Path
    ) -> VectorDaemon:
        """Create a VectorDaemon with VoyageClient for testing."""
        from mcp_code_indexer.vector_mode.config import VectorConfig

        config = VectorConfig(
            voyage_api_key="test-voyage-key",
            turbopuffer_api_key="test-turbopuffer-key",
            batch_size=32
        )
        cache_dir = tmp_path / "test_cache"
        
        # Mock API validation to avoid real API calls during testing
        with patch('mcp_code_indexer.vector_mode.providers.voyage_client.VoyageClient.validate_api_access'), \
             patch('mcp_code_indexer.vector_mode.providers.turbopuffer_client.TurbopufferClient.validate_api_access'):
            daemon = VectorDaemon(config, db_manager, cache_dir)
        # Initialize stats
        daemon.stats = {
            "files_processed": 0, 
            "errors_count": 0, 
            "last_activity": 0.0,
            "embeddings_generated": 0
        }
        return daemon

    @pytest.fixture
    def sample_chunks(self):
        """Create sample code chunks for testing."""
        from mcp_code_indexer.vector_mode.chunking.ast_chunker import CodeChunk
        from mcp_code_indexer.database.models import ChunkType
        
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
        self, vector_daemon_with_voyage: VectorDaemon, sample_chunks
    ):
        """Test successful embedding generation."""
        project_name = "test_project"
        file_path = Path("/test/file.py")
        
        # Mock EmbeddingService
        with patch.object(vector_daemon_with_voyage, '_embedding_service') as mock_service:
            mock_service.generate_embeddings_for_chunks = AsyncMock(return_value=[
                [0.1, 0.2, 0.3] * 512,  # 1536 dimensions for voyage-code-2
                [0.4, 0.5, 0.6] * 512
            ])
            
            embeddings = await vector_daemon_with_voyage._generate_embeddings(
                sample_chunks, project_name, file_path
            )
            
            # Verify embeddings were returned
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 1536  # voyage-code-2 dimension
            assert len(embeddings[1]) == 1536
            
            # Verify EmbeddingService was called correctly
            mock_service.generate_embeddings_for_chunks.assert_called_once_with(
                sample_chunks, project_name, file_path
            )
            
            # Verify stats were updated
            assert vector_daemon_with_voyage.stats["embeddings_generated"] == 2

    async def test_generate_embeddings_empty_chunks(
        self, vector_daemon_with_voyage: VectorDaemon
    ):
        """Test embedding generation with empty chunks list."""
        project_name = "test_project"
        file_path = Path("/test/file.py")
        
        embeddings = await vector_daemon_with_voyage._generate_embeddings(
            [], project_name, file_path
        )
        
        # Should return empty list for no chunks
        assert embeddings == []
        assert vector_daemon_with_voyage.stats["embeddings_generated"] == 0

    async def test_generate_embeddings_batch_processing(
        self, vector_daemon_with_voyage: VectorDaemon
    ):
        """Test embedding generation with large batch requiring splitting."""
        from mcp_code_indexer.vector_mode.chunking.ast_chunker import CodeChunk
        from mcp_code_indexer.database.models import ChunkType
        
        # Create more chunks than batch size
        large_chunk_list = []
        for i in range(50):  # More than default batch size of 32
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
            large_chunk_list.append(chunk)
        
        project_name = "test_project"
        file_path = Path("/test/file.py")
        
        # Mock EmbeddingService (batching is handled internally)
        with patch.object(vector_daemon_with_voyage, '_embedding_service') as mock_service:
            mock_service.generate_embeddings_for_chunks = AsyncMock(return_value=[[0.1] * 1536] * 50)
            
            embeddings = await vector_daemon_with_voyage._generate_embeddings(
                large_chunk_list, project_name, file_path
            )
            
            # Verify all embeddings returned
            assert len(embeddings) == 50
            
            # Verify EmbeddingService was called once with all chunks
            mock_service.generate_embeddings_for_chunks.assert_called_once_with(
                large_chunk_list, project_name, file_path
            )
            
            # Verify stats
            assert vector_daemon_with_voyage.stats["embeddings_generated"] == 50

    async def test_generate_embeddings_api_error(
        self, vector_daemon_with_voyage: VectorDaemon, sample_chunks
    ):
        """Test handling of EmbeddingService API errors."""
        project_name = "test_project"
        file_path = Path("/test/file.py")
        
        # Mock EmbeddingService to raise exception
        with patch.object(vector_daemon_with_voyage, '_embedding_service') as mock_service:
            mock_service.generate_embeddings_for_chunks = AsyncMock(side_effect=RuntimeError("API Error"))
            
            # Should raise the error and update error stats
            with pytest.raises(RuntimeError, match="API Error"):
                await vector_daemon_with_voyage._generate_embeddings(
                    sample_chunks, project_name, file_path
                )
            
            # Verify no embeddings stats were updated on error
            assert vector_daemon_with_voyage.stats["embeddings_generated"] == 0
            # Error stats should be incremented
            assert vector_daemon_with_voyage.stats["errors_count"] == 1

    async def test_generate_embeddings_redacted_chunks(
        self, vector_daemon_with_voyage: VectorDaemon
    ):
        """Test embedding generation with redacted chunks."""
        from mcp_code_indexer.vector_mode.chunking.ast_chunker import CodeChunk
        from mcp_code_indexer.database.models import ChunkType
        
        chunks = [
            CodeChunk(
                content="def normal_function():\n    return 'hello'",
                chunk_type=ChunkType.FUNCTION,
                name="normal_function",
                file_path="/test/file.py",
                start_line=1,
                end_line=2,
                content_hash="normal123",
                language="python",
                redacted=False
            ),
            CodeChunk(
                content="def secret_function():\n    api_key = '[REDACTED]'\n    return api_key",
                chunk_type=ChunkType.FUNCTION,
                name="secret_function", 
                file_path="/test/file.py",
                start_line=4,
                end_line=6,
                content_hash="redacted456",
                language="python",
                redacted=True
            )
        ]
        
        project_name = "test_project"
        file_path = Path("/test/file.py")
        
        with patch.object(vector_daemon_with_voyage, '_embedding_service') as mock_service:
            mock_service.generate_embeddings_for_chunks = AsyncMock(return_value=[
                [0.1] * 1536,  # Normal chunk embedding
                [0.2] * 1536   # Redacted chunk embedding
            ])
            
            embeddings = await vector_daemon_with_voyage._generate_embeddings(
                chunks, project_name, file_path
            )
            
            # Should generate embeddings for both chunks (redacted content included)
            assert len(embeddings) == 2
            
            # Verify EmbeddingService was called with both chunks
            mock_service.generate_embeddings_for_chunks.assert_called_once_with(
                chunks, project_name, file_path
            )

    async def test_voyage_client_initialization(
        self, db_manager: DatabaseManager, tmp_path: Path
    ):
        """Test that VoyageClient is properly initialized."""
        from mcp_code_indexer.vector_mode.config import VectorConfig
        
        # Test client creation with mocked create_voyage_client and mocked validation
        with patch('mcp_code_indexer.vector_mode.daemon.create_voyage_client') as mock_create, \
             patch('mcp_code_indexer.vector_mode.providers.voyage_client.VoyageClient.validate_api_access'), \
             patch('mcp_code_indexer.vector_mode.providers.turbopuffer_client.TurbopufferClient.validate_api_access'):
            mock_client = mock_create.return_value
            
            config = VectorConfig(
                voyage_api_key="test-voyage-key",
                turbopuffer_api_key="test-turbopuffer-key",
                batch_size=32
            )
            cache_dir = tmp_path / "test_cache"
            daemon = VectorDaemon(config, db_manager, cache_dir)
            
            # Verify client creation was called with correct config
            mock_create.assert_called_once_with(config)
            assert daemon._voyage_client == mock_client

    async def test_voyage_client_initialization_no_api_key(
        self, db_manager: DatabaseManager, tmp_path: Path
    ):
        """Test VoyageClient initialization without API key should raise ValueError."""
        from mcp_code_indexer.vector_mode.config import VectorConfig
        
        config = VectorConfig(
            voyage_api_key=None,  # No API key
            batch_size=32
        )
        cache_dir = tmp_path / "test_cache"
        
        # Should raise ValueError when no API key provided
        with pytest.raises(ValueError, match="VOYAGE_API_KEY is required for embedding generation"):
            VectorDaemon(config, db_manager, cache_dir)

    async def test_generate_embeddings_input_type_document(
        self, vector_daemon_with_voyage: VectorDaemon, sample_chunks
    ):
        """Test that embeddings are generated via EmbeddingService."""
        project_name = "test_project"
        file_path = Path("/test/file.py")
        
        with patch.object(vector_daemon_with_voyage, '_embedding_service') as mock_service:
            mock_service.generate_embeddings_for_chunks = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
            
            await vector_daemon_with_voyage._generate_embeddings(
                sample_chunks, project_name, file_path
            )
            
            # Verify EmbeddingService was called correctly (input_type handling is internal)
            mock_service.generate_embeddings_for_chunks.assert_called_once_with(
                sample_chunks, project_name, file_path
            )


class TestVectorDaemonVectorStorageIntegration:
    """Test VectorDaemon integration with VectorStorageService for vector storage."""

    @pytest.fixture
    async def vector_daemon_with_turbopuffer(
        self, db_manager: DatabaseManager, tmp_path: Path
    ) -> VectorDaemon:
        """Create a VectorDaemon with VectorStorageService for testing."""
        from mcp_code_indexer.vector_mode.config import VectorConfig

        config = VectorConfig(
            voyage_api_key="test-voyage-key",
            turbopuffer_api_key="test-turbopuffer-key",
            batch_size=32,
            turbopuffer_region="gcp-europe-west3"
        )
        cache_dir = tmp_path / "test_cache"
        
        # Mock API validation to avoid real API calls during testing
        with patch('mcp_code_indexer.vector_mode.providers.voyage_client.VoyageClient.validate_api_access'), \
             patch('mcp_code_indexer.vector_mode.providers.turbopuffer_client.TurbopufferClient.validate_api_access'):
            daemon = VectorDaemon(config, db_manager, cache_dir)
        daemon.stats = {
            "files_processed": 0, 
            "errors_count": 0, 
            "last_activity": 0.0,
            "embeddings_generated": 0
        }
        return daemon

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return [
            [0.1, 0.2, 0.3] * 512,  # 1536 dimensions
            [0.4, 0.5, 0.6] * 512
        ]

    @pytest.fixture
    def sample_chunks_with_metadata(self):
        """Create sample code chunks with metadata for testing."""
        from mcp_code_indexer.vector_mode.chunking.ast_chunker import CodeChunk
        from mcp_code_indexer.database.models import ChunkType
        
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

    async def test_turbopuffer_client_initialization(
        self, db_manager: DatabaseManager, tmp_path: Path
    ):
        """Test that TurbopufferClient is properly initialized."""
        from mcp_code_indexer.vector_mode.config import VectorConfig
        
        with patch('mcp_code_indexer.vector_mode.daemon.create_turbopuffer_client') as mock_create_client, \
             patch('mcp_code_indexer.vector_mode.providers.voyage_client.VoyageClient.validate_api_access'), \
             patch('mcp_code_indexer.vector_mode.providers.turbopuffer_client.TurbopufferClient.validate_api_access'):
            mock_client = mock_create_client.return_value
            
            config = VectorConfig(
                voyage_api_key="test-voyage-key",
                turbopuffer_api_key="test-turbopuffer-key",
                turbopuffer_region="gcp-europe-west3"
            )
            cache_dir = tmp_path / "test_cache"
            daemon = VectorDaemon(config, db_manager, cache_dir)
            
            # Verify client creation was called with correct config
            mock_create_client.assert_called_once_with(config)
            assert daemon._turbopuffer_client == mock_client

    async def test_vector_storage_service_initialization(
        self, vector_daemon_with_turbopuffer: VectorDaemon
    ):
        """Test that VectorStorageService is properly initialized."""
        # VectorStorageService should be initialized in constructor
        assert hasattr(vector_daemon_with_turbopuffer, '_vector_storage_service')
        assert vector_daemon_with_turbopuffer._vector_storage_service is not None

    async def test_store_embeddings_success(
        self, 
        vector_daemon_with_turbopuffer: VectorDaemon, 
        sample_embeddings,
        sample_chunks_with_metadata
    ):
        """Test successful storage of embeddings using VectorStorageService."""
        project_name = "test_project"
        file_path = "/test/file.py"
        
        # Mock VectorStorageService
        with patch.object(vector_daemon_with_turbopuffer, '_vector_storage_service') as mock_service:
            mock_service.store_embeddings = AsyncMock()
            
            await vector_daemon_with_turbopuffer._store_embeddings(
                sample_embeddings, sample_chunks_with_metadata, project_name, file_path
            )
            
            # Verify VectorStorageService was called correctly
            mock_service.store_embeddings.assert_called_once_with(
                sample_embeddings, sample_chunks_with_metadata, project_name, file_path
            )

    async def test_store_embeddings_empty_list(
        self, vector_daemon_with_turbopuffer: VectorDaemon
    ):
        """Test storing empty embeddings list."""
        project_name = "test_project"
        file_path = "/test/file.py"
        
        with patch.object(vector_daemon_with_turbopuffer, '_vector_storage_service') as mock_service:
            mock_service.store_embeddings = AsyncMock()
            
            await vector_daemon_with_turbopuffer._store_embeddings(
                [], [], project_name, file_path
            )
            
            # Should still call service (let service handle empty case)
            mock_service.store_embeddings.assert_called_once_with(
                [], [], project_name, file_path
            )

    async def test_store_embeddings_service_error(
        self, 
        vector_daemon_with_turbopuffer: VectorDaemon, 
        sample_embeddings,
        sample_chunks_with_metadata
    ):
        """Test handling of VectorStorageService errors."""
        project_name = "test_project"
        file_path = "/test/file.py"
        
        # Get initial error count
        initial_error_count = vector_daemon_with_turbopuffer.stats["errors_count"]
        
        # Mock service to raise exception
        with patch.object(vector_daemon_with_turbopuffer, '_vector_storage_service') as mock_service:
            mock_service.store_embeddings = AsyncMock(side_effect=RuntimeError("Turbopuffer API Error"))
            
            # Should propagate the error and increment error count
            with pytest.raises(RuntimeError, match="Turbopuffer API Error"):
                await vector_daemon_with_turbopuffer._store_embeddings(
                    sample_embeddings, sample_chunks_with_metadata, project_name, file_path
                )
            
            # Verify error count was incremented
            assert vector_daemon_with_turbopuffer.stats["errors_count"] == initial_error_count + 1

    async def test_store_embeddings_dimension_mismatch(
        self, vector_daemon_with_turbopuffer: VectorDaemon, sample_chunks_with_metadata
    ):
        """Test handling of mismatched embeddings and chunks count."""
        project_name = "test_project"
        file_path = "/test/file.py"
        
        # Mismatched counts - 3 embeddings vs 2 chunks
        mismatched_embeddings = [
            [0.1] * 1536,
            [0.2] * 1536,
            [0.3] * 1536
        ]
        
        with patch.object(vector_daemon_with_turbopuffer, '_vector_storage_service') as mock_service:
            mock_service.store_embeddings = AsyncMock(
                side_effect=ValueError("Embeddings and chunks count mismatch")
            )
            
            with pytest.raises(ValueError, match="Embeddings and chunks count mismatch"):
                await vector_daemon_with_turbopuffer._store_embeddings(
                    mismatched_embeddings, sample_chunks_with_metadata, project_name, file_path
                )

    async def test_turbopuffer_client_no_api_key_error(
        self, db_manager: DatabaseManager, tmp_path: Path
    ):
        """Test TurbopufferClient initialization without API key should raise ValueError."""
        from mcp_code_indexer.vector_mode.config import VectorConfig
        
        config = VectorConfig(
            voyage_api_key="test-voyage-key",
            turbopuffer_api_key=None  # No API key
        )
        cache_dir = tmp_path / "test_cache"
        
        # Mock Voyage validation to pass, should fail on Turbopuffer validation  
        with patch('mcp_code_indexer.vector_mode.providers.voyage_client.VoyageClient.validate_api_access'):
            # Should raise ValueError when no API key provided
            with pytest.raises(ValueError, match="TURBOPUFFER_API_KEY is required for vector storage"):
                VectorDaemon(config, db_manager, cache_dir)
