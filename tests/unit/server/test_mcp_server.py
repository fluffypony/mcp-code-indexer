"""
Unit tests for MCP server vector mode handler operations.

This module tests the MCPCodeIndexServer._handle_enabled_vector_mode method including:
- Successful enable/disable operations
- New project creation handling
- Integration with database operations
- Boolean value validation
- Error handling scenarios
"""

from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from mcp_code_indexer.server.mcp_server import MCPCodeIndexServer


class TestMCPHandlerVectorMode:
    """Test MCP server handler for vector mode operations."""

    async def test_handle_enabled_vector_mode_enable_success(self, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
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

    async def test_handle_enabled_vector_mode_disable_success(self, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
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

    async def test_handle_enabled_vector_mode_new_project(self, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
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

    async def test_handle_enabled_vector_mode_integration(self, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
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

    async def test_handle_enabled_vector_mode_boolean_validation(self, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
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

    async def test_database_error_handling(self, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
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
