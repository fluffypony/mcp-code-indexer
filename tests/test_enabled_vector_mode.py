"""
Tests for the enabled_vector_mode tool functionality.

This module tests the vector mode enable/disable functionality including:
- DatabaseManager.set_project_vector_mode method
- MCP server handler _handle_enabled_vector_mode
- Error handling for invalid projects
- Integration between handler and database operations
"""

import pytest
import pytest_asyncio

from mcp_code_indexer.database.models import Project
from mcp_code_indexer.server.mcp_server import MCPCodeIndexServer


@pytest_asyncio.fixture
async def mcp_server(temp_db, mock_file_system):
    """Create an MCP server for testing."""
    server = MCPCodeIndexServer(
        token_limit=1000,
        db_path=temp_db,
        cache_dir=mock_file_system / "cache",
    )
    await server.initialize()
    yield server
    
    # Cleanup
    if hasattr(server, "db_manager") and server.db_manager:
        await server.db_manager.close_pool()


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

    async def test_handle_enabled_vector_mode_enable_success(self, mcp_server, mock_file_system):
        """Test successful vector mode enable via MCP handler."""
        arguments = {
            "projectName": "Test Project",
            "folderPath": str(mock_file_system),
            "enabled": True,
        }
        
        result = await mcp_server._handle_enabled_vector_mode(arguments)
        
        assert result["success"] is True
        assert result["vector_mode"] is True
        assert "enabled" in result["message"]
        assert "project_id" in result

    async def test_handle_enabled_vector_mode_disable_success(self, mcp_server, mock_file_system):
        """Test successful vector mode disable via MCP handler."""
        # First enable it
        enable_arguments = {
            "projectName": "Test Project",
            "folderPath": str(mock_file_system),
            "enabled": True,
        }
        await mcp_server._handle_enabled_vector_mode(enable_arguments)
        
        # Then disable it
        disable_arguments = {
            "projectName": "Test Project",
            "folderPath": str(mock_file_system),
            "enabled": False,
        }
        
        result = await mcp_server._handle_enabled_vector_mode(disable_arguments)
        
        assert result["success"] is True
        assert result["vector_mode"] is False
        assert "disabled" in result["message"]
        assert "project_id" in result

    async def test_handle_enabled_vector_mode_new_project(self, mcp_server, mock_file_system):
        """Test that handler creates new project if it doesn't exist."""
        arguments = {
            "projectName": "New Project",
            "folderPath": str(mock_file_system),
            "enabled": True,
        }
        
        result = await mcp_server._handle_enabled_vector_mode(arguments)
        
        assert result["success"] is True
        assert result["vector_mode"] is True
        assert result["project_id"] is not None

    async def test_handle_enabled_vector_mode_integration(self, mcp_server, mock_file_system):
        """Test integration between handler and database operations."""
        project_name = "Integration Test Project"
        arguments = {
            "projectName": project_name,
            "folderPath": str(mock_file_system),
            "enabled": True,
        }
        
        # Enable vector mode via handler
        result = await mcp_server._handle_enabled_vector_mode(arguments)
        project_id = result["project_id"]
        
        # Verify in database directly
        db_manager = await mcp_server.db_factory.get_database_manager(str(mock_file_system))
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == project_id), None)
        
        assert project is not None
        assert project.vector_mode is True
        # Project names are normalized to lowercase in the database
        assert project.name == project_name.lower()

    async def test_handle_enabled_vector_mode_boolean_validation(self, mcp_server, mock_file_system):
        """Test that handler properly handles boolean values."""
        # Test with explicit True
        result = await mcp_server._handle_enabled_vector_mode({
            "projectName": "Bool Test",
            "folderPath": str(mock_file_system),
            "enabled": True,
        })
        assert result["success"] is True
        assert result["vector_mode"] is True
        
        # Test with explicit False
        result = await mcp_server._handle_enabled_vector_mode({
            "projectName": "Bool Test",
            "folderPath": str(mock_file_system),
            "enabled": False,
        })
        assert result["success"] is True
        assert result["vector_mode"] is False


class TestVectorModeErrorHandling:
    """Test error handling scenarios for vector mode operations."""

    async def test_database_error_handling(self, mcp_server, mock_file_system):
        """Test that database errors are handled gracefully."""
        # Close the database to simulate an error
        if hasattr(mcp_server, "db_manager") and mcp_server.db_manager:
            await mcp_server.db_manager.close_pool()
        
        arguments = {
            "projectName": "Error Test",
            "folderPath": str(mock_file_system),
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
