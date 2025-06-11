"""
MCP Server implementation for the Code Indexer.

This module provides the main MCP server that handles JSON-RPC communication
for file description management tools.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..database.database import DatabaseManager
from ..file_scanner import FileScanner
from ..token_counter import TokenCounter
from ..database.models import (
    Project, FileDescription, CodebaseOverview, SearchResult,
    CodebaseSizeInfo, FolderNode, FileNode
)

logger = logging.getLogger(__name__)


class MCPCodeIndexServer:
    """
    MCP Code Index Server.
    
    Provides file description tracking and codebase navigation tools
    through the Model Context Protocol.
    """
    
    def __init__(
        self,
        token_limit: int = 32000,
        db_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the MCP Code Index Server.
        
        Args:
            token_limit: Maximum tokens before recommending search over overview
            db_path: Path to SQLite database
            cache_dir: Directory for caching
        """
        self.token_limit = token_limit
        self.db_path = db_path or Path.home() / ".mcp-code-index" / "tracker.db"
        self.cache_dir = cache_dir or Path.home() / ".mcp-code-index" / "cache"
        
        # Initialize components
        self.db_manager = DatabaseManager(self.db_path)
        self.token_counter = TokenCounter(token_limit)
        
        # Create MCP server
        self.server = Server("mcp-code-indexer")
        
        # Register handlers
        self._register_handlers()
        
        logger.info(f"MCP Code Index Server initialized with token limit: {token_limit}")
    
    async def initialize(self) -> None:
        """Initialize database and other resources."""
        await self.db_manager.initialize()
        logger.info("Server initialized successfully")
    
    def _register_handlers(self) -> None:
        """Register MCP tool and resource handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """Return list of available tools."""
            return [
                types.Tool(
                    name="get_file_description",
                    description="Retrieves the stored description for a specific file in a codebase. Use this to quickly understand what a file contains without reading its full contents.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "projectName": {
                                "type": "string",
                                "description": "The name of the project"
                            },
                            "folderPath": {
                                "type": "string", 
                                "description": "Absolute path to the project folder on disk"
                            },
                            "branch": {
                                "type": "string",
                                "description": "Git branch name (e.g., 'main', 'develop')"
                            },
                            "remoteOrigin": {
                                "type": ["string", "null"],
                                "description": "Git remote origin URL if available"
                            },
                            "upstreamOrigin": {
                                "type": ["string", "null"],
                                "description": "Upstream repository URL if this is a fork"
                            },
                            "filePath": {
                                "type": "string",
                                "description": "Relative path to the file from project root"
                            }
                        },
                        "required": ["projectName", "folderPath", "branch", "filePath"]
                    }
                ),
                types.Tool(
                    name="update_file_description",
                    description="Creates or updates the description for a file. Use this after analyzing a file's contents to store a detailed summary.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "projectName": {"type": "string", "description": "The name of the project"},
                            "folderPath": {"type": "string", "description": "Absolute path to the project folder on disk"},
                            "branch": {"type": "string", "description": "Git branch name"},
                            "remoteOrigin": {"type": ["string", "null"], "description": "Git remote origin URL if available"},
                            "upstreamOrigin": {"type": ["string", "null"], "description": "Upstream repository URL if this is a fork"},
                            "filePath": {"type": "string", "description": "Relative path to the file from project root"},
                            "description": {"type": "string", "description": "Detailed description of the file's contents"},
                            "fileHash": {"type": ["string", "null"], "description": "SHA-256 hash of the file contents (optional)"}
                        },
                        "required": ["projectName", "folderPath", "branch", "filePath", "description"]
                    }
                ),
                types.Tool(
                    name="check_codebase_size",
                    description="Checks the total token count of a codebase's file structure and descriptions. Returns whether the codebase is 'large' and recommends using search instead of the full overview.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "projectName": {"type": "string", "description": "The name of the project"},
                            "folderPath": {"type": "string", "description": "Absolute path to the project folder on disk"},
                            "branch": {"type": "string", "description": "Git branch name"},
                            "remoteOrigin": {"type": ["string", "null"], "description": "Git remote origin URL if available"},
                            "upstreamOrigin": {"type": ["string", "null"], "description": "Upstream repository URL if this is a fork"}
                        },
                        "required": ["projectName", "folderPath", "branch"]
                    }
                ),
                types.Tool(
                    name="find_missing_descriptions",
                    description="Scans the project folder to find files that don't have descriptions yet. This is stage 1 of a two-stage process for updating missing descriptions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "projectName": {"type": "string", "description": "The name of the project"},
                            "folderPath": {"type": "string", "description": "Absolute path to the project folder on disk"},
                            "branch": {"type": "string", "description": "Git branch name"},
                            "remoteOrigin": {"type": ["string", "null"], "description": "Git remote origin URL if available"},
                            "upstreamOrigin": {"type": ["string", "null"], "description": "Upstream repository URL if this is a fork"}
                        },
                        "required": ["projectName", "folderPath", "branch"]
                    }
                ),
                types.Tool(
                    name="update_missing_descriptions",
                    description="Batch updates descriptions for multiple files at once. This is stage 2 after find_missing_descriptions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "projectName": {"type": "string", "description": "The name of the project"},
                            "folderPath": {"type": "string", "description": "Absolute path to the project folder on disk"},
                            "branch": {"type": "string", "description": "Git branch name"},
                            "remoteOrigin": {"type": ["string", "null"], "description": "Git remote origin URL if available"},
                            "upstreamOrigin": {"type": ["string", "null"], "description": "Upstream repository URL if this is a fork"},
                            "descriptions": {
                                "type": "array",
                                "description": "Array of file paths and their descriptions",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "filePath": {"type": "string", "description": "Relative path to the file"},
                                        "description": {"type": "string", "description": "Detailed description of the file"}
                                    },
                                    "required": ["filePath", "description"]
                                }
                            }
                        },
                        "required": ["projectName", "folderPath", "branch", "descriptions"]
                    }
                ),
                types.Tool(
                    name="search_descriptions",
                    description="Searches through all file descriptions in a project to find files related to specific functionality. Use this for large codebases instead of loading the entire structure.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "projectName": {"type": "string", "description": "The name of the project"},
                            "folderPath": {"type": "string", "description": "Absolute path to the project folder on disk"},
                            "branch": {"type": "string", "description": "Git branch to search in"},
                            "remoteOrigin": {"type": ["string", "null"], "description": "Git remote origin URL if available"},
                            "upstreamOrigin": {"type": ["string", "null"], "description": "Upstream repository URL if this is a fork"},
                            "query": {"type": "string", "description": "Search query (e.g., 'authentication middleware', 'database models')"},
                            "maxResults": {"type": "integer", "default": 20, "description": "Maximum number of results to return"}
                        },
                        "required": ["projectName", "folderPath", "branch", "query"]
                    }
                ),
                types.Tool(
                    name="get_codebase_overview",
                    description="Returns the complete file and folder structure of a codebase with all descriptions. For large codebases, this will recommend using search_descriptions instead.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "projectName": {"type": "string", "description": "The name of the project"},
                            "folderPath": {"type": "string", "description": "Absolute path to the project folder on disk"},
                            "branch": {"type": "string", "description": "Git branch name"},
                            "remoteOrigin": {"type": ["string", "null"], "description": "Git remote origin URL if available"},
                            "upstreamOrigin": {"type": ["string", "null"], "description": "Upstream repository URL if this is a fork"}
                        },
                        "required": ["projectName", "folderPath", "branch"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls."""
            try:
                if name == "get_file_description":
                    result = await self._handle_get_file_description(arguments)
                elif name == "update_file_description":
                    result = await self._handle_update_file_description(arguments)
                elif name == "check_codebase_size":
                    result = await self._handle_check_codebase_size(arguments)
                elif name == "find_missing_descriptions":
                    result = await self._handle_find_missing_descriptions(arguments)
                elif name == "update_missing_descriptions":
                    result = await self._handle_update_missing_descriptions(arguments)
                elif name == "search_descriptions":
                    result = await self._handle_search_descriptions(arguments)
                elif name == "get_codebase_overview":
                    result = await self._handle_get_codebase_overview(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]
                
            except Exception as e:
                logger.error(f"Error handling tool {name}: {e}")
                error_result = {
                    "error": str(e),
                    "tool": name,
                    "arguments": arguments
                }
                return [types.TextContent(
                    type="text", 
                    text=json.dumps(error_result, indent=2)
                )]
    
    async def _get_or_create_project_id(self, arguments: Dict[str, Any]) -> str:
        """Get or create a project ID from tool arguments."""
        project_name = arguments["projectName"]
        remote_origin = arguments.get("remoteOrigin")
        upstream_origin = arguments.get("upstreamOrigin")
        folder_path = arguments["folderPath"]
        
        # Create project ID from identifiers
        id_source = f"{project_name}:{remote_origin}:{upstream_origin}:{folder_path}"
        project_id = hashlib.sha256(id_source.encode()).hexdigest()[:16]
        
        # Check if project exists, create if not
        project = await self.db_manager.get_project(project_id)
        if not project:
            project = Project(
                id=project_id,
                name=project_name,
                remote_origin=remote_origin,
                upstream_origin=upstream_origin,
                aliases=[folder_path],
                created=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            await self.db_manager.create_project(project)
        else:
            # Update last accessed time
            await self.db_manager.update_project_access_time(project_id)
        
        return project_id
    
    async def _handle_get_file_description(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_file_description tool calls."""
        project_id = await self._get_or_create_project_id(arguments)
        
        file_desc = await self.db_manager.get_file_description(
            project_id=project_id,
            branch=arguments["branch"],
            file_path=arguments["filePath"]
        )
        
        if file_desc:
            return {
                "exists": True,
                "description": file_desc.description,
                "lastModified": file_desc.last_modified.isoformat(),
                "fileHash": file_desc.file_hash,
                "version": file_desc.version
            }
        else:
            return {
                "exists": False,
                "message": f"No description found for {arguments['filePath']}"
            }
    
    async def _handle_update_file_description(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update_file_description tool calls."""
        project_id = await self._get_or_create_project_id(arguments)
        
        file_desc = FileDescription(
            project_id=project_id,
            branch=arguments["branch"],
            file_path=arguments["filePath"],
            description=arguments["description"],
            file_hash=arguments.get("fileHash"),
            last_modified=datetime.utcnow(),
            version=1
        )
        
        await self.db_manager.create_file_description(file_desc)
        
        return {
            "success": True,
            "message": f"Description updated for {arguments['filePath']}",
            "filePath": arguments["filePath"],
            "lastModified": file_desc.last_modified.isoformat()
        }
    
    async def _handle_check_codebase_size(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle check_codebase_size tool calls."""
        project_id = await self._get_or_create_project_id(arguments)
        
        # Get all file descriptions for this project/branch
        file_descriptions = await self.db_manager.get_all_file_descriptions(
            project_id=project_id,
            branch=arguments["branch"]
        )
        
        # Calculate total tokens
        total_tokens = self.token_counter.calculate_codebase_tokens(file_descriptions)
        is_large = self.token_counter.is_large_codebase(total_tokens)
        recommendation = self.token_counter.get_recommendation(total_tokens)
        
        return {
            "totalTokens": total_tokens,
            "isLarge": is_large,
            "recommendation": recommendation,
            "tokenLimit": self.token_counter.token_limit,
            "totalFiles": len(file_descriptions)
        }
    
    async def _handle_find_missing_descriptions(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle find_missing_descriptions tool calls."""
        project_id = await self._get_or_create_project_id(arguments)
        folder_path = Path(arguments["folderPath"])
        
        # Get existing file descriptions
        existing_descriptions = await self.db_manager.get_all_file_descriptions(
            project_id=project_id,
            branch=arguments["branch"]
        )
        existing_paths = {desc.file_path for desc in existing_descriptions}
        
        # Scan directory for files
        scanner = FileScanner(folder_path)
        if not scanner.is_valid_project_directory():
            return {
                "error": f"Invalid or inaccessible project directory: {folder_path}"
            }
        
        missing_files = scanner.find_missing_files(existing_paths)
        missing_paths = [scanner.get_relative_path(f) for f in missing_files]
        
        # Get project stats
        stats = scanner.get_project_stats()
        
        return {
            "missingFiles": missing_paths,
            "totalMissing": len(missing_paths),
            "existingDescriptions": len(existing_paths),
            "projectStats": stats
        }
    
    async def _handle_update_missing_descriptions(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle update_missing_descriptions tool calls."""
        project_id = await self._get_or_create_project_id(arguments)
        descriptions_data = arguments["descriptions"]
        
        # Create FileDescription objects
        file_descriptions = []
        for desc_data in descriptions_data:
            file_desc = FileDescription(
                project_id=project_id,
                branch=arguments["branch"],
                file_path=desc_data["filePath"],
                description=desc_data["description"],
                file_hash=None,  # Hash not provided in batch operations
                last_modified=datetime.utcnow(),
                version=1
            )
            file_descriptions.append(file_desc)
        
        # Batch create descriptions
        await self.db_manager.batch_create_file_descriptions(file_descriptions)
        
        return {
            "success": True,
            "updatedFiles": len(file_descriptions),
            "files": [desc["filePath"] for desc in descriptions_data],
            "message": f"Successfully updated descriptions for {len(file_descriptions)} files"
        }
    
    async def _handle_search_descriptions(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_descriptions tool calls."""
        project_id = await self._get_or_create_project_id(arguments)
        max_results = arguments.get("maxResults", 20)
        
        # Perform search
        search_results = await self.db_manager.search_file_descriptions(
            project_id=project_id,
            branch=arguments["branch"],
            query=arguments["query"],
            max_results=max_results
        )
        
        # Format results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "filePath": result.file_path,
                "description": result.description,
                "relevanceScore": result.relevance_score
            })
        
        return {
            "results": formatted_results,
            "totalResults": len(formatted_results),
            "query": arguments["query"],
            "maxResults": max_results
        }
    
    async def _handle_get_codebase_overview(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_codebase_overview tool calls."""
        project_id = await self._get_or_create_project_id(arguments)
        
        # Get all file descriptions
        file_descriptions = await self.db_manager.get_all_file_descriptions(
            project_id=project_id,
            branch=arguments["branch"]
        )
        
        # Calculate total tokens
        total_tokens = self.token_counter.calculate_codebase_tokens(file_descriptions)
        is_large = self.token_counter.is_large_codebase(total_tokens)
        
        # If large, recommend search instead
        if is_large:
            return {
                "isLarge": True,
                "totalTokens": total_tokens,
                "tokenLimit": self.token_counter.token_limit,
                "totalFiles": len(file_descriptions),
                "recommendation": "use_search",
                "message": f"Codebase has {total_tokens} tokens (limit: {self.token_counter.token_limit}). Use search_descriptions instead for better performance."
            }
        
        # Build folder structure
        structure = self._build_folder_structure(file_descriptions)
        
        return {
            "projectName": arguments["projectName"],
            "branch": arguments["branch"],
            "totalFiles": len(file_descriptions),
            "totalTokens": total_tokens,
            "isLarge": is_large,
            "tokenLimit": self.token_counter.token_limit,
            "structure": structure
        }
    
    def _build_folder_structure(self, file_descriptions: List[FileDescription]) -> Dict[str, Any]:
        """Build hierarchical folder structure from file descriptions."""
        root = {"name": "", "path": "", "files": [], "folders": {}}
        
        for file_desc in file_descriptions:
            path_parts = Path(file_desc.file_path).parts
            current = root
            
            # Navigate/create folder structure
            for i, part in enumerate(path_parts[:-1]):
                folder_path = "/".join(path_parts[:i+1])
                if part not in current["folders"]:
                    current["folders"][part] = {
                        "name": part,
                        "path": folder_path,
                        "files": [],
                        "folders": {}
                    }
                current = current["folders"][part]
            
            # Add file to current folder
            if path_parts:  # Handle empty paths
                current["files"].append({
                    "name": path_parts[-1],
                    "path": file_desc.file_path,
                    "description": file_desc.description
                })
        
        # Convert nested dict structure to list format
        def convert_structure(node):
            return {
                "name": node["name"],
                "path": node["path"],
                "files": node["files"],
                "folders": [convert_structure(folder) for folder in node["folders"].values()]
            }
        
        return convert_structure(root)
    
    async def run(self) -> None:
        """Run the MCP server."""
        await self.initialize()
        
        async with stdio_server() as (read_stream, write_stream):
            initialization_options = self.server.create_initialization_options()
            await self.server.run(
                read_stream,
                write_stream, 
                initialization_options
            )


async def main():
    """Main entry point for the MCP server."""
    import sys
    
    # Setup logging to stderr (stdout is used for MCP communication)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    
    # Create and run server
    server = MCPCodeIndexServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
