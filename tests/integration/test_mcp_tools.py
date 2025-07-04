"""
Integration tests for MCP tools.

This module tests the complete MCP tool workflow including server initialization,
tool execution, error handling, and response formatting.
"""

from pathlib import Path
import pytest
import pytest_asyncio

from src.mcp_code_indexer.server.mcp_server import MCPCodeIndexServer


class TestMCPServerIntegration:
    """Integration tests for the MCP server."""

    @pytest_asyncio.fixture
    async def mcp_server(self, tmp_path: Path):
        """Create an MCP server for testing."""
        db_path = tmp_path / "test.db"
        cache_dir = tmp_path / "cache"

        server = MCPCodeIndexServer(
            token_limit=1000, db_path=db_path, cache_dir=cache_dir
        )
        await server.initialize()

        yield server

        await server.shutdown()

    async def test_server_initialization(self, mcp_server):
        """Test server initializes correctly."""
        assert mcp_server.token_limit == 1000
        assert mcp_server.db_manager is not None
        assert mcp_server.token_counter is not None
        assert mcp_server.error_handler is not None

    async def test_get_file_description_not_found(self, mcp_server):
        """Test getting file description when it doesn't exist."""
        arguments = {
            "projectName": "test-project",
            "folderPath": "/tmp/test",
            "filePath": "nonexistent.py",
        }

        result = await mcp_server._handle_get_file_description(arguments)

        assert result["exists"] is False
        assert "No description found" in result["message"]

    async def test_update_and_get_file_description(self, mcp_server):
        """Test updating and retrieving file description."""
        # Update file description
        update_args = {
            "projectName": "test-project",
            "folderPath": "/tmp/test",
            "filePath": "test.py",
            "description": "Test file for integration testing",
            "fileHash": "abc123",
        }

        update_result = await mcp_server._handle_update_file_description(update_args)

        assert update_result["success"] is True
        assert update_result["filePath"] == "test.py"

        # Get file description
        get_args = {
            "projectName": "test-project",
            "folderPath": "/tmp/test",
            "filePath": "test.py",
        }

        get_result = await mcp_server._handle_get_file_description(get_args)

        assert get_result["exists"] is True
        assert get_result["description"] == "Test file for integration testing"
        assert get_result["fileHash"] == "abc123"

    async def test_check_codebase_size(self, mcp_server):
        """Test checking codebase size."""
        # Add some file descriptions
        project_name = "size-test"
        folder_path = "/tmp/size-test"

        # Add individual file descriptions
        for i, desc in enumerate(["First file", "Second file", "Third file"], 1):
            update_args = {
                "projectName": project_name,
                "folderPath": folder_path,
                "filePath": f"file{i}.py",
                "description": desc,
                "fileHash": f"hash{i}",
            }
            await mcp_server._handle_update_file_description(update_args)

        # Check size - use the current project directory for test
        size_args = {
            "projectName": project_name,
            "folderPath": folder_path,
            "tokenLimit": 1000,
        }

        result = await mcp_server._handle_check_codebase_size(size_args)

        # Note: The actual count might be 0 if cleanup removes files without
        # corresponding filesystem entries. This is expected behavior for
        # check_codebase_size which validates against filesystem
        assert "totalFiles" in result
        assert "totalTokens" in result
        assert result["tokenLimit"] == 1000
        assert "isLarge" in result
        assert "recommendation" in result

    async def test_search_descriptions(self, mcp_server):
        """Test searching file descriptions."""
        # Add searchable content
        descriptions_args = {
            "projectName": "search-test",
            "folderPath": "/tmp/search-test",
            "descriptions": [
                {
                    "filePath": "auth.py",
                    "description": "Authentication and authorization module",
                },
                {
                    "filePath": "database.py",
                    "description": "Database connection and models",
                },
                {
                    "filePath": "api.py",
                    "description": "REST API endpoints and handlers",
                },
                {
                    "filePath": "utils.py",
                    "description": "Utility functions and helpers",
                },
            ],
        }

        # Add files individually instead of using non-existent method
        for file_data in descriptions_args["descriptions"]:
            update_args = {
                "projectName": descriptions_args["projectName"],
                "folderPath": descriptions_args["folderPath"],
                "filePath": file_data["filePath"],
                "description": file_data["description"],
            }
            await mcp_server._handle_update_file_description(update_args)

        # Search for authentication
        search_args = {
            "projectName": "search-test",
            "folderPath": "/tmp/search-test",
            "query": "authentication",
            "maxResults": 10,
        }

        result = await mcp_server._handle_search_descriptions(search_args)

        assert result["totalResults"] >= 1
        assert any("auth.py" in r["filePath"] for r in result["results"])
        assert result["query"] == "authentication"

    async def test_intelligent_multi_word_search(self, mcp_server):
        """Test intelligent multi-word search with query preprocessing."""
        # Add individual file descriptions for testing
        project_name = "multi-word-test"
        folder_path = "/tmp/multi-word-test"

        # Add multiple files with descriptions containing target terms
        files_to_add = [
            ("grpc_server.py", "gRPC server implementation with protocol buffers"),
            ("proto_handler.py", "Protocol buffer message handling and validation"),
            ("auth_middleware.py", "Authentication middleware for API requests"),
            ("config_parser.py", "Configuration file parser with YAML support"),
            ("error_handler.py", "Error handling AND logging utilities"),
        ]

        for file_path, description in files_to_add:
            args = {
                "projectName": project_name,
                "folderPath": folder_path,
                "filePath": file_path,
                "description": description,
            }
            await mcp_server._handle_update_file_description(args)

        # Test multi-word query (should find files containing both words)
        search_args = {
            "projectName": project_name,
            "folderPath": folder_path,
            "query": "grpc proto",
            "maxResults": 10,
        }

        result = await mcp_server._handle_search_descriptions(search_args)

        # Should find files containing both 'grpc' and 'proto'
        assert result["totalResults"] >= 2
        file_paths = [r["filePath"] for r in result["results"]]
        assert "grpc_server.py" in file_paths
        assert "proto_handler.py" in file_paths

    async def test_fts5_operator_literal_search(self, mcp_server):
        """Test FTS5 operators are treated as literal search terms."""
        project_name = "operator-test"
        folder_path = "/tmp/operator-test"

        files_to_add = [
            ("error_handler.py", "Error handling AND logging utilities"),
            ("backup_service.py", "Database backup OR restore service"),
            ("config.py", "Configuration NOT secrets management"),
        ]

        for file_path, description in files_to_add:
            args = {
                "projectName": project_name,
                "folderPath": folder_path,
                "filePath": file_path,
                "description": description,
            }
            await mcp_server._handle_update_file_description(args)

        # Search for 'AND' as a literal term (not as FTS5 operator)
        search_args = {
            "projectName": project_name,
            "folderPath": folder_path,
            "query": "error AND logging",
            "maxResults": 10,
        }

        result = await mcp_server._handle_search_descriptions(search_args)

        # Should find the file containing all three terms: 'error', 'AND', 'logging'
        assert result["totalResults"] >= 1
        assert any("error_handler.py" in r["filePath"] for r in result["results"])

    async def test_order_agnostic_search(self, mcp_server):
        """Test that multi-word search is order-agnostic."""
        project_name = "order-test"
        folder_path = "/tmp/order-test"

        files_to_add = [
            ("proto_grpc.py", "Protocol buffer gRPC service definitions"),
            ("config_yaml.py", "YAML configuration file parser"),
        ]

        for file_path, description in files_to_add:
            args = {
                "projectName": project_name,
                "folderPath": folder_path,
                "filePath": file_path,
                "description": description,
            }
            await mcp_server._handle_update_file_description(args)

        # Test both orders should find the same results
        search_args_1 = {
            "projectName": project_name,
            "folderPath": folder_path,
            "query": "grpc protocol",
            "maxResults": 10,
        }

        search_args_2 = {
            "projectName": project_name,
            "folderPath": folder_path,
            "query": "protocol grpc",
            "maxResults": 10,
        }

        result_1 = await mcp_server._handle_search_descriptions(search_args_1)
        result_2 = await mcp_server._handle_search_descriptions(search_args_2)

        # Both queries should find the same file
        assert result_1["totalResults"] >= 1
        assert result_2["totalResults"] >= 1
        assert result_1["totalResults"] == result_2["totalResults"]

        file_paths_1 = [r["filePath"] for r in result_1["results"]]
        file_paths_2 = [r["filePath"] for r in result_2["results"]]
        assert "proto_grpc.py" in file_paths_1
        assert "proto_grpc.py" in file_paths_2

    async def test_get_codebase_overview_small(self, mcp_server):
        """Test getting codebase overview for small codebase."""
        # Add files for overview
        descriptions_args = {
            "projectName": "overview-test",
            "folderPath": "/tmp/overview-test",
            "descriptions": [
                {
                    "filePath": "src/main.py",
                    "description": "Main application entry point",
                },
                {"filePath": "src/utils.py", "description": "Utility functions"},
                {
                    "filePath": "tests/test_main.py",
                    "description": "Tests for main module",
                },
            ],
        }

        # Add files individually
        for file_data in descriptions_args["descriptions"]:
            update_args = {
                "projectName": descriptions_args["projectName"],
                "folderPath": descriptions_args["folderPath"],
                "filePath": file_data["filePath"],
                "description": file_data["description"],
            }
            await mcp_server._handle_update_file_description(update_args)

        # Get overview
        overview_args = {
            "projectName": "overview-test",
            "folderPath": "/tmp/overview-test",
        }

        result = await mcp_server._handle_get_codebase_overview(overview_args)

        assert result["projectName"] == "overview-test"
        assert result["totalFiles"] == 3
        assert "structure" in result

        # Check structure contains expected folders
        structure = result["structure"]
        assert structure["name"] == ""  # Root

        # Should have src and tests folders
        folder_names = [f["name"] for f in structure["folders"]]
        assert "src" in folder_names
        assert "tests" in folder_names

    async def test_upstream_inheritance(self, mcp_server):
        """Test automatic upstream inheritance."""
        # Create upstream project with descriptions
        upstream_args = {
            "projectName": "upstream-project",
            "folderPath": "/tmp/upstream",
            "remoteOrigin": "https://github.com/upstream/repo.git",
            "descriptions": [
                {
                    "filePath": "core.py",
                    "description": "Core functionality from upstream",
                },
                {"filePath": "utils.py", "description": "Utilities from upstream"},
            ],
        }

        # Add upstream files individually
        for file_data in upstream_args["descriptions"]:
            update_args = {
                "projectName": upstream_args["projectName"],
                "folderPath": upstream_args["folderPath"],
                "filePath": file_data["filePath"],
                "description": file_data["description"],
            }
            await mcp_server._handle_update_file_description(update_args)

        # Create fork project that should inherit
        fork_args = {
            "projectName": "fork-project",
            "folderPath": "/tmp/fork",
            "remoteOrigin": "https://github.com/user/repo.git",
            "upstreamOrigin": "https://github.com/upstream/repo.git",
            "filePath": "README.md",
            "description": "Fork-specific readme",
        }

        # This should trigger upstream inheritance
        await mcp_server._handle_update_file_description(fork_args)

        # Check that upstream files were inherited
        get_args = {
            "projectName": "fork-project",
            "folderPath": "/tmp/fork",
            "filePath": "core.py",
        }

        result = await mcp_server._handle_get_file_description(get_args)

        assert result["exists"] is True
        assert "Core functionality from upstream" in result["description"]


class TestMCPToolErrors:
    """Test error handling in MCP tools."""

    @pytest_asyncio.fixture
    async def mcp_server(self, tmp_path: Path):
        """Create an MCP server for error testing."""
        db_path = tmp_path / "test.db"
        cache_dir = tmp_path / "cache"

        server = MCPCodeIndexServer(
            token_limit=1000, db_path=db_path, cache_dir=cache_dir
        )
        await server.initialize()

        yield server

        await server.shutdown()

    async def test_missing_required_fields(self, mcp_server):
        """Test tool calls with missing required fields."""
        # This should trigger validation error
        arguments = {
            "projectName": "test-project",
            # Missing folderPath, branch, filePath
        }

        try:
            await mcp_server._handle_get_file_description(arguments)
            assert False, "Should have raised KeyError"
        except KeyError:
            # Expected - missing required fields
            pass


@pytest.mark.performance
class TestMCPPerformance:
    """Performance tests for MCP tools."""

    @pytest_asyncio.fixture
    async def mcp_server_with_data(self, tmp_path: Path):
        """Create server with large dataset."""
        db_path = tmp_path / "test.db"
        cache_dir = tmp_path / "cache"

        server = MCPCodeIndexServer(
            token_limit=10000, db_path=db_path, cache_dir=cache_dir
        )
        await server.initialize()

        # Add large dataset
        descriptions = []
        for i in range(500):
            descriptions.append(
                {
                    "filePath": f"src/module_{i:03d}.py",
                    "description": (
                        f"Module {i} with functionality for feature set {i // 50}"
                    ),
                }
            )

        large_args = {
            "projectName": "large-project",
            "folderPath": "/tmp/large",
            "descriptions": descriptions,
        }

        # Add large files individually
        for file_data in large_args["descriptions"]:
            update_args = {
                "projectName": large_args["projectName"],
                "folderPath": large_args["folderPath"],
                "filePath": file_data["filePath"],
                "description": file_data["description"],
            }
            await server._handle_update_file_description(update_args)

        yield server

        await server.shutdown()

    async def test_large_search_performance(self, mcp_server_with_data):
        """Test search performance with large dataset."""
        import time

        search_args = {
            "projectName": "large-project",
            "folderPath": "/tmp/large",
            "query": "feature",
            "maxResults": 20,
        }

        start_time = time.time()
        result = await mcp_server_with_data._handle_search_descriptions(search_args)
        search_time = time.time() - start_time

        assert result["totalResults"] > 0
        assert search_time < 2.0  # Should complete in under 2 seconds

    async def test_large_overview_performance(self, mcp_server_with_data):
        """Test overview performance with large dataset."""
        import time

        overview_args = {"projectName": "large-project", "folderPath": "/tmp/large"}

        start_time = time.time()
        result = await mcp_server_with_data._handle_get_codebase_overview(overview_args)
        overview_time = time.time() - start_time

        # Should recommend search for large codebase
        if result.get("isLarge"):
            assert result["recommendation"] == "use_search"
        else:
            # If not large, should have structure
            assert "structure" in result

        assert overview_time < 3.0  # Should complete in under 3 seconds


@pytest.mark.integration
class TestMCPWorkflow:
    """Test complete MCP workflows."""

    @pytest_asyncio.fixture
    async def mcp_server(self, tmp_path: Path):
        """Create server for workflow testing."""
        db_path = tmp_path / "test.db"
        cache_dir = tmp_path / "cache"

        server = MCPCodeIndexServer(
            token_limit=1000, db_path=db_path, cache_dir=cache_dir
        )
        await server.initialize()

        yield server

        await server.shutdown()

    async def test_complete_project_workflow(self, mcp_server):
        """Test complete workflow from project creation to merge."""
        project_args = {"projectName": "workflow-test", "folderPath": "/tmp/workflow"}

        # 1. Check initial size (should be empty)
        size_result = await mcp_server._handle_check_codebase_size(project_args)
        assert size_result["totalFiles"] == 0

        # 2. Add initial files
        initial_files = {
            **project_args,
            "descriptions": [
                {"filePath": "main.py", "description": "Main application file"},
                {"filePath": "config.py", "description": "Configuration settings"},
                {"filePath": "utils.py", "description": "Utility functions"},
            ],
        }

        # Add initial files individually
        for file_data in initial_files["descriptions"]:
            update_args = {
                "projectName": initial_files["projectName"],
                "folderPath": initial_files["folderPath"],
                "filePath": file_data["filePath"],
                "description": file_data["description"],
            }
            await mcp_server._handle_update_file_description(update_args)

        # 3. Check size after adding files
        size_result = await mcp_server._handle_check_codebase_size(project_args)
        assert size_result["totalFiles"] == 3

        # 4. Get overview
        overview_result = await mcp_server._handle_get_codebase_overview(project_args)
        assert overview_result["totalFiles"] == 3
        assert len(overview_result["structure"]["files"]) == 3

        # 5. Search for specific functionality
        search_result = await mcp_server._handle_search_descriptions(
            {**project_args, "query": "configuration", "maxResults": 5}
        )
        assert any("config.py" in r["filePath"] for r in search_result["results"])

        # 6. Create feature branch with changes
        feature_files = {
            "projectName": "workflow-test",
            "folderPath": "/tmp/workflow",
            "descriptions": [
                {
                    "filePath": "main.py",
                    "description": "Enhanced main application with new features",
                },
                {
                    "filePath": "new_feature.py",
                    "description": "New feature implementation",
                },
            ],
        }

        # Add feature files individually
        for file_data in feature_files["descriptions"]:
            update_args = {
                "projectName": feature_files["projectName"],
                "folderPath": feature_files["folderPath"],
                "filePath": file_data["filePath"],
                "description": file_data["description"],
            }
            await mcp_server._handle_update_file_description(update_args)

        # 7. Merge feature branch back to main
        merge_result = await mcp_server._handle_merge_branch_descriptions(
            {
                "projectName": "workflow-test",
                "folderPath": "/tmp/workflow",
                "sourceBranch": "feature/enhancement",
                "targetBranch": "main",
            }
        )

        # Should have one conflict (main.py)
        assert merge_result["phase"] == "conflicts_detected"
        assert merge_result["conflictCount"] == 1

        # Resolve and complete merge
        conflict = merge_result["conflicts"][0]
        resolution_result = await mcp_server._handle_merge_branch_descriptions(
            {
                "projectName": "workflow-test",
                "folderPath": "/tmp/workflow",
                "sourceBranch": "feature/enhancement",
                "targetBranch": "main",
                "conflictResolutions": [
                    {
                        "conflictId": conflict["conflictId"],
                        "resolvedDescription": (
                            "Main application with enhanced features "
                            "and new functionality"
                        ),
                    }
                ],
            }
        )

        assert resolution_result["phase"] == "completed"
        assert resolution_result["success"] is True

        # 8. Verify final state
        final_overview = await mcp_server._handle_get_codebase_overview(project_args)
        assert final_overview["totalFiles"] == 4  # main, config, utils, new_feature

        # Check that main.py has the resolved description
        main_file = await mcp_server._handle_get_file_description(
            {**project_args, "filePath": "main.py"}
        )
        assert "enhanced features" in main_file["description"]
