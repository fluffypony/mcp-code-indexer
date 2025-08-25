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

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from mcp_code_indexer.database.database import DatabaseManager
from mcp_code_indexer.database.models import Project
from mcp_code_indexer.vector_mode.daemon import VectorDaemon
from mcp_code_indexer.vector_mode.monitoring.change_detector import ChangeType, FileChange


class TestVectorDaemonMonitoringStatus:
    """Test VectorDaemon monitored_projects management."""
    
    @pytest.fixture
    async def vector_daemon(self, db_manager: DatabaseManager) -> VectorDaemon:
        """Create a VectorDaemon for testing."""
        from mcp_code_indexer.vector_mode.config import VectorConfig
        from pathlib import Path
        
        config = VectorConfig()
        cache_dir = Path("/tmp/test_cache")
        daemon = VectorDaemon(config, db_manager, cache_dir)
        return daemon

    async def test_get_project_monitoring_status_empty(self, db_manager: DatabaseManager, vector_daemon: VectorDaemon) -> None:
        """Test monitoring status when no projects exist."""
        status = await vector_daemon._get_project_monitoring_status()
        
        assert status["monitored"] == []
        assert status["unmonitored"] == []
        assert len(vector_daemon.monitored_projects) == 0

    async def test_get_project_monitoring_status_new_project(self, db_manager: DatabaseManager, sample_project: Project, vector_daemon: VectorDaemon) -> None:
        """Test monitoring status with a new vector-enabled project."""
        # Enable vector mode for the project
        await db_manager.set_project_vector_mode(sample_project.id, True)
        
        # Project already has aliases from fixture, so it should be monitorable
        status = await vector_daemon._get_project_monitoring_status()
        
        # Should be marked for monitoring
        assert len(status["monitored"]) == 1
        assert status["monitored"][0].name == sample_project.name
        assert len(status["unmonitored"]) == 0

    async def test_get_project_monitoring_status_unmonitor_project(self, db_manager: DatabaseManager, sample_project: Project, vector_daemon: VectorDaemon) -> None:
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

    async def test_get_project_monitoring_status_no_aliases(self, db_manager: DatabaseManager, vector_daemon: VectorDaemon) -> None:
        """Test that projects without aliases are not monitored."""
        # Create a project without aliases
        from mcp_code_indexer.database.models import Project
        from datetime import datetime
        
        project_no_aliases = Project(
            id="no_aliases_project",
            name="no aliases project", 
            aliases=[],  # No aliases
            created=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            vector_mode=False
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
    async def vector_daemon(self, db_manager: DatabaseManager, tmp_path: Path) -> VectorDaemon:
        """Create a VectorDaemon for testing."""
        from mcp_code_indexer.vector_mode.config import VectorConfig

        config = VectorConfig()
        cache_dir = tmp_path / "test_cache"
        daemon = VectorDaemon(config, db_manager, cache_dir)
        # Initialize stats
        daemon.stats = {
            "files_processed": 0,
            "errors_count": 0,
            "last_activity": 0.0
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
            size=len(sample_python_file.read_text())
        )
        
        task = {
            "project_name": "test_project",
            "change": change
        }
        
        worker_id = "worker_1"
        
        # Mock the logger and debug writer
        with patch("mcp_code_indexer.vector_mode.daemon.logger") as mock_logger, \
             patch("mcp_code_indexer.vector_mode.daemon._write_debug_log") as mock_debug_log:
            
            await vector_daemon._process_file_change_task(task, worker_id)
            
            # Verify stats were updated
            assert vector_daemon.stats["files_processed"] == 1
            assert vector_daemon.stats["errors_count"] == 0
            
            # Verify logging was called
            mock_logger.info.assert_called()
            mock_debug_log.assert_called()
            
            # Check the info log call for chunking results
            info_calls = mock_logger.info.call_args_list
            assert len(info_calls) >= 2  # Initial file change + chunking results
            
            # Find the chunking results log call
            chunking_call = None
            for call in info_calls:
                if "Chunked" in str(call[0][0]):
                    chunking_call = call
                    break
            
            assert chunking_call is not None
            assert "chunks" in str(chunking_call[0][0])

    async def test_process_file_change_task_modified_file(
        self, vector_daemon: VectorDaemon, sample_python_file: Path
    ) -> None:
        """Test processing a MODIFIED file change."""
        change = FileChange(
            path=str(sample_python_file),
            change_type=ChangeType.MODIFIED,
            timestamp=datetime.utcnow(),
            size=len(sample_python_file.read_text())
        )
        
        task = {
            "project_name": "test_project",
            "change": change
        }
        
        worker_id = "worker_2"
        
        with patch("mcp_code_indexer.vector_mode.daemon.logger") as mock_logger, \
             patch("mcp_code_indexer.vector_mode.daemon._write_debug_log") as mock_debug_log:
            
            await vector_daemon._process_file_change_task(task, worker_id)
            
            # Verify stats were updated
            assert vector_daemon.stats["files_processed"] == 1
            assert vector_daemon.stats["errors_count"] == 0
            
            # Verify chunking occurred
            mock_logger.info.assert_called()
            mock_debug_log.assert_called()

    async def test_process_file_change_task_deleted_file(
        self, vector_daemon: VectorDaemon, sample_python_file: Path
    ) -> None:
        """Test processing a DELETED file change (should be skipped)."""
        change = FileChange(
            path=str(sample_python_file),
            change_type=ChangeType.DELETED,
            timestamp=datetime.utcnow(),
            size=0
        )
        
        task = {
            "project_name": "test_project",
            "change": change
        }
        
        worker_id = "worker_3"
        
        with patch("mcp_code_indexer.vector_mode.daemon.logger") as mock_logger, \
             patch("mcp_code_indexer.vector_mode.daemon._write_debug_log") as mock_debug_log:
            
            await vector_daemon._process_file_change_task(task, worker_id)
            
            # Verify no processing occurred
            assert vector_daemon.stats["files_processed"] == 0
            assert vector_daemon.stats["errors_count"] == 0
            
            # Verify debug log about skipping was called
            mock_logger.debug.assert_called_with(f"Worker {worker_id}: Skipping deleted file {sample_python_file}")

    async def test_process_file_change_task_binary_file(
        self, vector_daemon: VectorDaemon, sample_binary_file: Path
    ) -> None:
        """Test processing a binary file (ASTChunker handles with errors='ignore')."""
        change = FileChange(
            path=str(sample_binary_file),
            change_type=ChangeType.CREATED,
            timestamp=datetime.utcnow(),
            size=6
        )
        
        task = {
            "project_name": "test_project",
            "change": change
        }
        
        worker_id = "worker_4"
        
        with patch("mcp_code_indexer.vector_mode.daemon.logger") as mock_logger, \
             patch("mcp_code_indexer.vector_mode.daemon._write_debug_log") as mock_debug_log:
            
            await vector_daemon._process_file_change_task(task, worker_id)
            
            # ASTChunker processes binary files using errors='ignore'
            # This may produce chunks, so files_processed gets incremented
            assert vector_daemon.stats["files_processed"] == 1
            assert vector_daemon.stats["errors_count"] == 0
            
            # Verify logging occurred
            mock_logger.info.assert_called()
            mock_debug_log.assert_called()

    async def test_process_file_change_task_nonexistent_file(
        self, vector_daemon: VectorDaemon, tmp_path: Path
    ) -> None:
        """Test processing a non-existent file (should handle gracefully)."""
        nonexistent_file = tmp_path / "nonexistent.py"
        
        change = FileChange(
            path=str(nonexistent_file),
            change_type=ChangeType.CREATED,
            timestamp=datetime.utcnow(),
            size=100
        )
        
        task = {
            "project_name": "test_project",
            "change": change
        }
        
        worker_id = "worker_5"
        
        with patch("mcp_code_indexer.vector_mode.daemon.logger") as mock_logger, \
             patch("mcp_code_indexer.vector_mode.daemon._write_debug_log") as mock_debug_log:
            
            await vector_daemon._process_file_change_task(task, worker_id)
            
            # ASTChunker handles nonexistent files gracefully by returning empty list
            # No chunks produced, so no files_processed increment
            assert vector_daemon.stats["files_processed"] == 0
            assert vector_daemon.stats["errors_count"] == 0
            
            # Verify debug log about no chunks was called
            mock_logger.debug.assert_called_with(f"Worker {worker_id}: No chunks produced for {nonexistent_file}")

    async def test_process_file_change_task_chunking_details(
        self, vector_daemon: VectorDaemon, sample_python_file: Path
    ) -> None:
        """Test that chunking details are logged correctly."""
        change = FileChange(
            path=str(sample_python_file),
            change_type=ChangeType.CREATED,
            timestamp=datetime.utcnow(),
            size=len(sample_python_file.read_text())
        )
        
        task = {
            "project_name": "test_project",
            "change": change
        }
        
        worker_id = "worker_6"
        
        with patch("mcp_code_indexer.vector_mode.daemon.logger") as mock_logger, \
             patch("mcp_code_indexer.vector_mode.daemon._write_debug_log") as mock_debug_log:
            
            await vector_daemon._process_file_change_task(task, worker_id)
            
            # Verify debug log was called with chunk details
            debug_calls = mock_debug_log.call_args_list
            assert len(debug_calls) >= 2  # Initial processing + chunk details
            
            # Check that chunk details are in the debug log
            chunk_details_call = None
            for call in debug_calls:
                if "chunking details:" in str(call[0][0]):
                    chunk_details_call = call
                    break
            
            assert chunk_details_call is not None
            chunk_details_text = str(chunk_details_call[0][0])
            assert "Total chunks:" in chunk_details_text
            assert "Chunk types:" in chunk_details_text
            assert "Redacted chunks:" in chunk_details_text
            assert "Sample chunks:" in chunk_details_text

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
            size=len(sample_python_file.read_text())
        )
        
        task = {
            "project_name": "test_project",
            "change": change
        }
        
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
            size=0
        )
        
        task = {
            "project_name": "test_project", 
            "change": change
        }
        
        worker_id = "worker_8"
        
        with patch("mcp_code_indexer.vector_mode.daemon.logger") as mock_logger, \
             patch("mcp_code_indexer.vector_mode.daemon._write_debug_log") as mock_debug_log:
            
            await vector_daemon._process_file_change_task(task, worker_id)
            
            # Empty files should be processed but result in 0 chunks
            # This depends on ASTChunker behavior - it should skip empty files
            # So no processing should occur
            assert vector_daemon.stats["files_processed"] == 0
            assert vector_daemon.stats["errors_count"] == 0
