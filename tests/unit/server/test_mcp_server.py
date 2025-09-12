"""
Unit tests for MCP server vector mode handler operations.

This module tests the MCPCodeIndexServer vector mode methods including:
- _handle_enabled_vector_mode method for enabling/disabling vector mode
- _handle_find_similar_code method for semantic code similarity search
- New project creation handling
- Integration with database operations
- Boolean value validation
- Error handling scenarios
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestMCPHandlerFindSimilarCode:
    """Test MCP server handler for find_similar_code operations."""

    @patch('mcp_code_indexer.vector_mode.services.vector_mode_tools_service.VectorModeToolsService')
    async def test_handle_find_similar_code_with_snippet_success(self, mock_service_class: MagicMock, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
        """Test successful find_similar_code with code snippet."""
        # Mock service instance and response
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        
        mock_service.find_similar_code.return_value = {
            "results": [
                {
                    "file_path": "src/utils/helpers.py",
                    "code_section": "def calculate_sum(numbers):\n    return sum(numbers)",
                    "similarity_score": 0.85,
                    "start_line": 10,
                    "end_line": 11,
                    "context": "Math utility functions"
                }
            ],
            "search_input": {
                "type": "snippet",
                "content": "def add_numbers(nums): return sum(nums)"
            },
            "total_results": 1,
            "similarity_threshold": 0.7
        }
        
        arguments = {
            "projectName": "test-project",
            "folderPath": str(tmp_path),
            "code_snippet": "def add_numbers(nums): return sum(nums)",
            "similarity_threshold": 0.7,
            "max_results": 10
        }
        
        result = await mcp_server._handle_find_similar_code(arguments)
        
        # Verify result structure
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["similarity_score"] == 0.85
        assert result["total_results"] == 1
        assert result["search_input"]["type"] == "snippet"
        
        # Verify service was called with correct parameters
        mock_service.find_similar_code.assert_called_once_with(
            project_name="test-project",
            folder_path=str(tmp_path),
            code_snippet="def add_numbers(nums): return sum(nums)",
            file_path=None,
            line_start=None,
            line_end=None,
            similarity_threshold=0.7,
            max_results=10
        )

    @patch('mcp_code_indexer.vector_mode.services.vector_mode_tools_service.VectorModeToolsService')
    async def test_handle_find_similar_code_with_file_section_success(self, mock_service_class: MagicMock, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
        """Test successful find_similar_code with file section."""
        # Mock service instance and response
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        
        mock_service.find_similar_code.return_value = {
            "results": [
                {
                    "file_path": "src/controllers/user_controller.py",
                    "code_section": "async def create_user(req):\n    user = await user_service.create(req.data)\n    return response.json(user)",
                    "similarity_score": 0.92,
                    "start_line": 25,
                    "end_line": 27,
                    "context": "User CRUD operations"
                }
            ],
            "search_input": {
                "type": "file_section",
                "content": "async def create_product(request):\n    product = await product_service.create(request.body)\n    return jsonify(product)",
                "source": "src/controllers/product_controller.py"
            },
            "total_results": 1,
            "similarity_threshold": 0.6
        }
        
        arguments = {
            "projectName": "api-project", 
            "folderPath": str(tmp_path),
            "file_path": "src/controllers/product_controller.py",
            "line_start": 15,
            "line_end": 20,
            "similarity_threshold": 0.6
        }
        
        result = await mcp_server._handle_find_similar_code(arguments)
        
        # Verify result structure
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["similarity_score"] == 0.92
        assert result["search_input"]["type"] == "file_section"
        assert result["search_input"]["source"] == "src/controllers/product_controller.py"
        
        # Verify service was called with correct parameters
        mock_service.find_similar_code.assert_called_once_with(
            project_name="api-project",
            folder_path=str(tmp_path),
            code_snippet=None,
            file_path="src/controllers/product_controller.py",
            line_start=15,
            line_end=20,
            similarity_threshold=0.6,
            max_results=None
        )

    @patch('mcp_code_indexer.vector_mode.services.vector_mode_tools_service.VectorModeToolsService')
    async def test_handle_find_similar_code_no_results(self, mock_service_class: MagicMock, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
        """Test find_similar_code when no similar code is found."""
        # Mock service instance with no results
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        
        mock_service.find_similar_code.return_value = {
            "results": [],
            "search_input": {
                "type": "snippet",
                "content": "very_unique_function_name_12345()"
            },
            "total_results": 0,
            "similarity_threshold": 0.8
        }
        
        arguments = {
            "projectName": "empty-project",
            "folderPath": str(tmp_path),
            "code_snippet": "very_unique_function_name_12345()",
            "similarity_threshold": 0.8
        }
        
        result = await mcp_server._handle_find_similar_code(arguments)
        
        # Verify empty results are handled correctly
        assert result["success"] is True
        assert result["results"] == []
        assert result["total_results"] == 0

    @patch('mcp_code_indexer.vector_mode.services.vector_mode_tools_service.VectorModeToolsService')
    async def test_handle_find_similar_code_service_error(self, mock_service_class: MagicMock, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
        """Test find_similar_code when service throws an error."""
        # Mock service instance that raises an exception
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        mock_service.find_similar_code.side_effect = Exception("Vector mode not enabled for project")
        
        arguments = {
            "projectName": "test-project",
            "folderPath": str(tmp_path),
            "code_snippet": "def test_function(): pass"
        }
        
        result = await mcp_server._handle_find_similar_code(arguments)
        
        # Verify error is handled gracefully
        assert result["success"] is False
        assert "Vector mode not enabled for project" in result["error"]
        assert result["results"] == []
        assert result["total_results"] == 0

    @patch('mcp_code_indexer.vector_mode.services.vector_mode_tools_service.VectorModeToolsService')
    async def test_handle_find_similar_code_minimal_arguments(self, mock_service_class: MagicMock, mcp_server: MCPCodeIndexServer, tmp_path: Path) -> None:
        """Test find_similar_code with only required arguments."""
        # Mock service instance 
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        
        mock_service.find_similar_code.return_value = {
            "results": [],
            "search_input": {"type": "snippet", "content": "print('hello')"},
            "total_results": 0,
            "similarity_threshold": 0.7  # Default threshold from service
        }
        
        # Only provide required arguments
        arguments = {
            "projectName": "minimal-test",
            "folderPath": str(tmp_path),
            "code_snippet": "print('hello')"
        }
        
        result = await mcp_server._handle_find_similar_code(arguments)
        
        # Verify it works with minimal arguments
        assert result["success"] is True
        
        # Verify service was called with None for optional parameters
        mock_service.find_similar_code.assert_called_once_with(
            project_name="minimal-test",
            folder_path=str(tmp_path),
            code_snippet="print('hello')",
            file_path=None,
            line_start=None,
            line_end=None,
            similarity_threshold=None,
            max_results=None
        )
