"""
Tests for the enabled_vector_mode tool functionality.

This module tests the vector mode enable/disable functionality including:
- DatabaseManager.set_project_vector_mode method
- MCP server handler _handle_enabled_vector_mode
- Error handling for invalid projects
- Integration between handler and database operations
- VectorDaemon monitored_projects management
"""

import pytest
import pytest_asyncio

from mcp_code_indexer.database.models import Project
from mcp_code_indexer.server.mcp_server import MCPCodeIndexServer
from mcp_code_indexer.vector_mode.daemon import VectorDaemon


class TestDatabaseManagerVectorMode:
    """Test DatabaseManager vector mode operations."""

    async def test_set_project_vector_mode_enable(self, db_manager, sample_project):
        """Test enabling vector mode for a project."""
        # Enable vector mode
        await db_manager.set_project_vector_mode(sample_project.id, True)
        
        # Verify the change
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project is not None
        assert project.vector_mode is True

    async def test_set_project_vector_mode_disable(self, db_manager, sample_project):
        """Test disabling vector mode for a project."""
        # First enable it
        await db_manager.set_project_vector_mode(sample_project.id, True)
        
        # Then disable it
        await db_manager.set_project_vector_mode(sample_project.id, False)
        
        # Verify the change
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project is not None
        assert project.vector_mode is False

    async def test_set_project_vector_mode_nonexistent_project(self, db_manager):
        """Test error handling for nonexistent project."""
        from mcp_code_indexer.database.exceptions import DatabaseError
        with pytest.raises(DatabaseError, match="Project not found"):
            await db_manager.set_project_vector_mode("nonexistent_id", True)

    async def test_set_project_vector_mode_toggle(self, db_manager, sample_project):
        """Test toggling vector mode multiple times."""
        # Default should be False
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is False
        
        # Enable
        await db_manager.set_project_vector_mode(sample_project.id, True)
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is True
        
        # Disable
        await db_manager.set_project_vector_mode(sample_project.id, False)
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is False
        
        # Enable again
        await db_manager.set_project_vector_mode(sample_project.id, True)
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is True


class TestMCPHandlerVectorMode:
    """Test MCP server handler for vector mode operations."""

    async def test_handle_enabled_vector_mode_enable_success(self, mcp_server, tmp_path):
        """Test successful vector mode enable via MCP handler."""
        arguments = {
            "projectName": "Test Project",
            "folderPath": str(tmp_path),
            "enabled": True,
        }
        
        result = await mcp_server._handle_enabled_vector_mode(arguments)
        
        assert result["success"] is True
        assert result["vector_mode"] is True
        assert "enabled" in result["message"]
        assert "project_id" in result

    async def test_handle_enabled_vector_mode_disable_success(self, mcp_server, tmp_path):
        """Test successful vector mode disable via MCP handler."""
        # First enable it
        enable_arguments = {
            "projectName": "Test Project",
            "folderPath": str(tmp_path),
            "enabled": True,
        }
        await mcp_server._handle_enabled_vector_mode(enable_arguments)
        
        # Then disable it
        disable_arguments = {
            "projectName": "Test Project",
            "folderPath": str(tmp_path),
            "enabled": False,
        }
        
        result = await mcp_server._handle_enabled_vector_mode(disable_arguments)
        
        assert result["success"] is True
        assert result["vector_mode"] is False
        assert "disabled" in result["message"]
        assert "project_id" in result

    async def test_handle_enabled_vector_mode_new_project(self, mcp_server, tmp_path):
        """Test that handler creates new project if it doesn't exist."""
        arguments = {
            "projectName": "New Project",
            "folderPath": str(tmp_path),
            "enabled": True,
        }
        
        result = await mcp_server._handle_enabled_vector_mode(arguments)
        
        assert result["success"] is True
        assert result["vector_mode"] is True
        assert result["project_id"] is not None

    async def test_handle_enabled_vector_mode_integration(self, mcp_server, tmp_path):
        """Test integration between handler and database operations."""
        project_name = "Integration Test Project"
        arguments = {
            "projectName": project_name,
            "folderPath": str(tmp_path),
            "enabled": True,
        }
        
        # Enable vector mode via handler
        result = await mcp_server._handle_enabled_vector_mode(arguments)
        project_id = result["project_id"]
        
        # Verify in database directly
        db_manager = await mcp_server.db_factory.get_database_manager(str(tmp_path))
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == project_id), None)
        
        assert project is not None
        assert project.vector_mode is True
        # Project names are normalized to lowercase in the database
        assert project.name == project_name.lower()

    async def test_handle_enabled_vector_mode_boolean_validation(self, mcp_server, tmp_path):
        """Test that handler properly handles boolean values."""
        # Test with explicit True
        result = await mcp_server._handle_enabled_vector_mode({
            "projectName": "Bool Test",
            "folderPath": str(tmp_path),
            "enabled": True,
        })
        assert result["success"] is True
        assert result["vector_mode"] is True
        
        # Test with explicit False
        result = await mcp_server._handle_enabled_vector_mode({
            "projectName": "Bool Test",
            "folderPath": str(tmp_path),
            "enabled": False,
        })
        assert result["success"] is True
        assert result["vector_mode"] is False


class TestVectorModeErrorHandling:
    """Test error handling scenarios for vector mode operations."""

    async def test_database_error_handling(self, mcp_server, tmp_path):
        """Test that database errors are handled gracefully."""
        # Close the database to simulate an error
        if hasattr(mcp_server, "db_manager") and mcp_server.db_manager:
            await mcp_server.db_manager.close_pool()
        
        arguments = {
            "projectName": "Error Test",
            "folderPath": str(tmp_path),
            "enabled": True,
        }
        
        # This should handle the database error gracefully
        # Note: Depending on the implementation, this might reconnect automatically
        # or return an error response
        result = await mcp_server._handle_enabled_vector_mode(arguments)
        
        # The exact behavior depends on the error handling implementation
        # At minimum, it shouldn't crash
        assert isinstance(result, dict)
        assert "success" in result


class TestVectorModeEdgeCases:
    """Test edge cases for vector mode operations."""

    async def test_set_same_value_twice(self, db_manager, sample_project):
        """Test setting the same vector mode value twice."""
        # Enable twice
        await db_manager.set_project_vector_mode(sample_project.id, True)
        await db_manager.set_project_vector_mode(sample_project.id, True)
        
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is True
        
        # Disable twice
        await db_manager.set_project_vector_mode(sample_project.id, False)
        await db_manager.set_project_vector_mode(sample_project.id, False)
        
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is False

    async def test_concurrent_vector_mode_operations(self, db_manager, sample_project):
        """Test concurrent vector mode operations."""
        import asyncio
        
        # Run multiple operations concurrently
        tasks = [
            db_manager.set_project_vector_mode(sample_project.id, True),
            db_manager.set_project_vector_mode(sample_project.id, False),
            db_manager.set_project_vector_mode(sample_project.id, True),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Final state should be deterministic (last operation wins)
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project is not None
        # The final state will be True since that's the last operation
        assert project.vector_mode is True


class TestVectorDaemonMonitoringStatus:
    """Test VectorDaemon monitored_projects management."""
    
    @pytest.fixture
    async def vector_daemon(self, db_manager):
        """Create a VectorDaemon for testing."""
        from mcp_code_indexer.vector_mode.config import VectorConfig
        from pathlib import Path
        
        config = VectorConfig()
        cache_dir = Path("/tmp/test_cache")
        daemon = VectorDaemon(config, db_manager, cache_dir)
        return daemon

    async def test_get_project_monitoring_status_empty(self, db_manager, vector_daemon):
        """Test monitoring status when no projects exist."""
        status = await vector_daemon._get_project_monitoring_status()
        
        assert status["monitored"] == []
        assert status["unmonitored"] == []
        assert len(vector_daemon.monitored_projects) == 0

    async def test_get_project_monitoring_status_new_project(self, db_manager, sample_project, vector_daemon):
        """Test monitoring status with a new vector-enabled project."""
        # Enable vector mode for the project
        await db_manager.set_project_vector_mode(sample_project.id, True)
        
        # Project already has aliases from fixture, so it should be monitorable
        status = await vector_daemon._get_project_monitoring_status()
        
        # Should be marked for monitoring
        assert len(status["monitored"]) == 1
        assert status["monitored"][0].name == sample_project.name
        assert len(status["unmonitored"]) == 0

    async def test_get_project_monitoring_status_unmonitor_project(self, db_manager, sample_project, vector_daemon):
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

    async def test_get_project_monitoring_status_no_aliases(self, db_manager, vector_daemon):
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
