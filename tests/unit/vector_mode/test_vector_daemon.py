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
from unittest.mock import patch

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
        config = VectorConfig()
        cache_dir = Path("/tmp/test_cache")
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

        config = VectorConfig()
        cache_dir = tmp_path / "test_cache"
        daemon = VectorDaemon(config, db_manager, cache_dir)
        # Initialize stats
        daemon.stats = {"files_processed": 0, "errors_count": 0, "last_activity": 0.0}
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

        # Spy on ASTChunker.chunk_file to verify it was called
        # NOTE: Ideally we'd use real chunker results, but for simplicity using mock return
        with patch("mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file") as mock_chunk_file:
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

        # Spy on ASTChunker.chunk_file to verify it was called
        with patch(
            "mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file"
        ) as mock_chunk_file:
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

            await vector_daemon._process_file_change_task(task, worker_id)

            # Verify stats were updated
            assert vector_daemon.stats["files_processed"] == 1
            assert vector_daemon.stats["errors_count"] == 0

            # Verify chunking occurred
            mock_chunk_file.assert_called_once_with(str(sample_python_file))

    async def test_process_file_change_task_deleted_file(
        self, vector_daemon: VectorDaemon, sample_python_file: Path
    ) -> None:
        """Test processing a DELETED file change (should be skipped)."""
        change = FileChange(
            path=str(sample_python_file),
            change_type=ChangeType.DELETED,
            timestamp=datetime.utcnow(),
            size=0,
        )

        task = {"project_name": "test_project", "change": change}

        worker_id = "worker_3"

        # Spy on ASTChunker.chunk_file to verify it was NOT called for deleted files
        with patch(
            "mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file"
        ) as mock_chunk_file:

            await vector_daemon._process_file_change_task(task, worker_id)

            # Verify no processing occurred
            assert vector_daemon.stats["files_processed"] == 0
            assert vector_daemon.stats["errors_count"] == 0

            # Verify ASTChunker.chunk_file was NOT called (deleted files are skipped)
            mock_chunk_file.assert_not_called()

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

        # Spy on ASTChunker.chunk_file to verify it was called
        with patch(
            "mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file"
        ) as mock_chunk_file:
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

        # Spy on ASTChunker.chunk_file to verify it was called and verify chunk details
        with patch(
            "mcp_code_indexer.vector_mode.chunking.ast_chunker.ASTChunker.chunk_file"
        ) as mock_chunk_file:
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
