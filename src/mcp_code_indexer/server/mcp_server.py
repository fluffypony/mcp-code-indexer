"""
MCP Server implementation for the Code Indexer.

This module provides the main MCP server that handles JSON-RPC communication
for file description management tools.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import ValidationError

from mcp_code_indexer.database.database import DatabaseManager
from mcp_code_indexer.file_scanner import FileScanner
from mcp_code_indexer.token_counter import TokenCounter
from mcp_code_indexer.database.models import (
    Project, FileDescription, CodebaseOverview, SearchResult,
    CodebaseSizeInfo, FolderNode, FileNode
)
from mcp_code_indexer.error_handler import setup_error_handling, ErrorHandler
from mcp_code_indexer.middleware.error_middleware import create_tool_middleware, AsyncTaskManager
from mcp_code_indexer.logging_config import get_logger
from mcp_code_indexer.merge_handler import MergeHandler

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
        self.merge_handler = MergeHandler(self.db_manager)
        
        # Setup error handling
        self.logger = get_logger(__name__)
        self.error_handler = setup_error_handling(self.logger)
        self.middleware = create_tool_middleware(self.error_handler)
        self.task_manager = AsyncTaskManager(self.error_handler)
        
        # Create MCP server
        self.server = Server("mcp-code-indexer")
        
        # Register handlers
        self._register_handlers()
        
        # Add debug logging for server events
        self.logger.debug("MCP server instance created and handlers registered")
        
        self.logger.info(
            "MCP Code Index Server initialized", 
            extra={"structured_data": {"initialization": {"token_limit": token_limit}}}
        )
    
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
                    description="Scans the project folder to find files that don't have descriptions yet. Use update_file_description to add descriptions for individual files.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "projectName": {"type": "string", "description": "The name of the project"},
                            "folderPath": {"type": "string", "description": "Absolute path to the project folder on disk"},
                            "branch": {"type": "string", "description": "Git branch name"},
                            "remoteOrigin": {"type": ["string", "null"], "description": "Git remote origin URL if available"},
                            "upstreamOrigin": {"type": ["string", "null"], "description": "Upstream repository URL if this is a fork"},
                            "limit": {"type": "integer", "description": "Maximum number of missing files to return (optional)"}
                        },
                        "required": ["projectName", "folderPath", "branch"]
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
                ),
                types.Tool(
                    name="merge_branch_descriptions",
                    description="Merges file descriptions from one branch to another. This is a two-stage process: first call without resolutions returns conflicts where the same file has different descriptions in each branch. Second call with resolutions completes the merge.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "projectName": {"type": "string", "description": "The name of the project"},
                            "folderPath": {"type": "string", "description": "Absolute path to the project folder"},
                            "remoteOrigin": {"type": ["string", "null"], "description": "Git remote origin URL"},
                            "upstreamOrigin": {"type": ["string", "null"], "description": "Upstream repository URL if this is a fork"},
                            "sourceBranch": {"type": "string", "description": "Branch to merge from (e.g., 'feature/new-ui')"},
                            "targetBranch": {"type": "string", "description": "Branch to merge into (e.g., 'main')"},
                            "conflictResolutions": {
                                "type": ["array", "null"],
                                "description": "Array of resolved conflicts (only for second stage)",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "conflictId": {"type": "string", "description": "ID of the conflict to resolve"},
                                        "resolvedDescription": {"type": "string", "description": "Final description to use after merge"}
                                    },
                                    "required": ["conflictId", "resolvedDescription"]
                                }
                            }
                        },
                        "required": ["projectName", "folderPath", "sourceBranch", "targetBranch"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls with middleware."""
            # Map tool names to handler methods
            tool_handlers = {
                "get_file_description": self._handle_get_file_description,
                "update_file_description": self._handle_update_file_description,
                "check_codebase_size": self._handle_check_codebase_size,
                "find_missing_descriptions": self._handle_find_missing_descriptions,
                "search_descriptions": self._handle_search_descriptions,
                "get_codebase_overview": self._handle_get_codebase_overview,
                "merge_branch_descriptions": self._handle_merge_branch_descriptions,
            }
            
            if name not in tool_handlers:
                from ..error_handler import ValidationError
                raise ValidationError(f"Unknown tool: {name}")
            
            # Wrap handler with middleware
            wrapped_handler = self.middleware.wrap_tool_handler(name)(
                lambda args: self._execute_tool_handler(tool_handlers[name], args)
            )
            
            return await wrapped_handler(arguments)
    
    async def _execute_tool_handler(self, handler, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Execute a tool handler and format the result."""
        result = await handler(arguments)
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
    
    async def _get_or_create_project_id(self, arguments: Dict[str, Any]) -> str:
        """
        Get or create a project ID using intelligent matching.
        
        Matches projects based on 2+ out of 4 identification factors:
        1. Project name (normalized, case-insensitive)
        2. Remote origin URL
        3. Upstream origin URL  
        4. Any folder path in aliases
        
        If only 1 factor matches, uses file similarity to determine if it's the same project.
        """
        project_name = arguments["projectName"]
        remote_origin = arguments.get("remoteOrigin")
        upstream_origin = arguments.get("upstreamOrigin")
        folder_path = arguments["folderPath"]
        branch = arguments.get("branch", "main")
        
        # Normalize project name for case-insensitive matching
        normalized_name = project_name.lower()
        
        # Find potential project matches
        project = await self._find_matching_project(
            normalized_name, remote_origin, upstream_origin, folder_path
        )
        if project:
            # Update project metadata and aliases
            await self._update_existing_project(project, normalized_name, remote_origin, upstream_origin, folder_path)
            
            # Check if upstream inheritance is needed
            if upstream_origin and await self.db_manager.check_upstream_inheritance_needed(project):
                try:
                    inherited_count = await self.db_manager.inherit_from_upstream(project, branch)
                    if inherited_count > 0:
                        logger.info(f"Auto-inherited {inherited_count} descriptions from upstream for {normalized_name}")
                except Exception as e:
                    logger.warning(f"Failed to inherit from upstream: {e}")
        else:
            # Create new project with UUID
            project_id = str(uuid.uuid4())
            project = Project(
                id=project_id,
                name=normalized_name,
                remote_origin=remote_origin,
                upstream_origin=upstream_origin,
                aliases=[folder_path],
                created=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            await self.db_manager.create_project(project)
            logger.info(f"Created new project: {normalized_name} ({project_id})")
            
            # Auto-inherit from upstream if needed
            if upstream_origin:
                try:
                    inherited_count = await self.db_manager.inherit_from_upstream(project, branch)
                    if inherited_count > 0:
                        logger.info(f"Auto-inherited {inherited_count} descriptions from upstream for {normalized_name}")
                except Exception as e:
                    logger.warning(f"Failed to inherit from upstream: {e}")
        
        return project.id
    
    async def _find_matching_project(
        self, 
        normalized_name: str, 
        remote_origin: Optional[str], 
        upstream_origin: Optional[str], 
        folder_path: str
    ) -> Optional[Project]:
        """
        Find a matching project using intelligent 2-out-of-4 matching logic.
        
        Returns the best matching project or None if no sufficient match is found.
        """
        all_projects = await self.db_manager.get_all_projects()
        
        best_match = None
        best_score = 0
        
        for project in all_projects:
            score = 0
            match_factors = []
            
            # Factor 1: Project name match
            if project.name.lower() == normalized_name:
                score += 1
                match_factors.append("name")
            
            # Factor 2: Remote origin match
            if remote_origin and project.remote_origin == remote_origin:
                score += 1
                match_factors.append("remote_origin")
                
            # Factor 3: Upstream origin match  
            if upstream_origin and project.upstream_origin == upstream_origin:
                score += 1
                match_factors.append("upstream_origin")
                
            # Factor 4: Folder path in aliases
            project_aliases = json.loads(project.aliases) if isinstance(project.aliases, str) else project.aliases
            if folder_path in project_aliases:
                score += 1
                match_factors.append("folder_path")
            
            # If we have 2+ matches, this is a strong candidate
            if score >= 2:
                if score > best_score:
                    best_score = score
                    best_match = project
                    logger.info(f"Strong match for project {project.name} (score: {score}, factors: {match_factors})")
            
            # If only 1 match, check file similarity for potential matches
            elif score == 1:
                if await self._check_file_similarity(project, folder_path):
                    logger.info(f"File similarity match for project {project.name} (factor: {match_factors[0]})")
                    if score > best_score:
                        best_score = score
                        best_match = project
        
        return best_match
    
    async def _check_file_similarity(self, project: Project, folder_path: str) -> bool:
        """
        Check if the files in the folder are similar to files already indexed for this project.
        Returns True if 80%+ of files match.
        """
        try:
            # Get files currently in the folder
            scanner = FileScanner(Path(folder_path))
            if not scanner.is_valid_project_directory():
                return False
            
            current_files = scanner.scan_files()
            current_basenames = {Path(f).name for f in current_files}
            
            if not current_basenames:
                return False
            
            # Get files already indexed for this project
            indexed_files = await self.db_manager.get_all_file_descriptions(project.id, "main")
            indexed_basenames = {Path(fd.file_path).name for fd in indexed_files}
            
            if not indexed_basenames:
                return False
            
            # Calculate similarity
            intersection = current_basenames & indexed_basenames
            similarity = len(intersection) / len(current_basenames)
            
            logger.debug(f"File similarity for {project.name}: {similarity:.2%} ({len(intersection)}/{len(current_basenames)} files match)")
            
            return similarity >= 0.8
        except Exception as e:
            logger.warning(f"Error checking file similarity: {e}")
            return False
    
    async def _update_existing_project(
        self, 
        project: Project, 
        normalized_name: str,
        remote_origin: Optional[str], 
        upstream_origin: Optional[str], 
        folder_path: str
    ) -> None:
        """Update an existing project with new metadata and folder alias."""
        # Update last accessed time
        await self.db_manager.update_project_access_time(project.id)
        
        should_update = False
        
        # Update name if different
        if project.name != normalized_name:
            project.name = normalized_name
            should_update = True
            
        # Update remote/upstream origins if provided and different
        if remote_origin and project.remote_origin != remote_origin:
            project.remote_origin = remote_origin
            should_update = True
            
        if upstream_origin and project.upstream_origin != upstream_origin:
            project.upstream_origin = upstream_origin
            should_update = True
        
        # Add folder path to aliases if not already present
        project_aliases = json.loads(project.aliases) if isinstance(project.aliases, str) else project.aliases
        if folder_path not in project_aliases:
            project_aliases.append(folder_path)
            project.aliases = project_aliases
            should_update = True
            logger.info(f"Added new folder alias to project {project.name}: {folder_path}")
        
        if should_update:
            await self.db_manager.update_project(project)
            logger.debug(f"Updated project metadata for {project.name}")
    
    async def _find_best_branch(self, project_id: str, requested_branch: str) -> Optional[str]:
        """
        Find the best available branch for a project when the requested branch has no files.
        Returns the branch with the most files, or None if no branches have files.
        """
        try:
            # Get all branches and their file counts for this project
            branch_counts = await self.db_manager.get_branch_file_counts(project_id)
            
            if not branch_counts:
                return None
            
            # First try common branch name variations
            common_variations = {
                'main': ['master', 'develop', 'development', 'dev'],
                'master': ['main', 'develop', 'development', 'dev'], 
                'develop': ['development', 'main', 'master', 'dev'],
                'development': ['develop', 'main', 'master', 'dev'],
                'dev': ['develop', 'development', 'main', 'master']
            }
            
            # Try variations of the requested branch
            if requested_branch.lower() in common_variations:
                for variation in common_variations[requested_branch.lower()]:
                    if variation in branch_counts and branch_counts[variation] > 0:
                        return variation
            
            # Fall back to the branch with the most files
            best_branch = max(branch_counts.items(), key=lambda x: x[1])
            return best_branch[0] if best_branch[1] > 0 else None
            
        except Exception as e:
            logger.warning(f"Error finding best branch: {e}")
            return None
    
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
        requested_branch = arguments["branch"]
        
        # Get file descriptions for this project/branch
        file_descriptions = await self.db_manager.get_all_file_descriptions(
            project_id=project_id,
            branch=requested_branch
        )
        
        # If no files found for requested branch, try to find the best available branch
        if not file_descriptions:
            available_branch = await self._find_best_branch(project_id, requested_branch)
            if available_branch and available_branch != requested_branch:
                file_descriptions = await self.db_manager.get_all_file_descriptions(
                    project_id=project_id,
                    branch=available_branch
                )
                logger.info(f"No files found for branch '{requested_branch}', using '{available_branch}' instead")
        
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
        
        # Apply limit if specified
        limit = arguments.get("limit")
        total_missing = len(missing_paths)
        if limit is not None and isinstance(limit, int) and limit > 0:
            missing_paths = missing_paths[:limit]
        
        # Get project stats
        stats = scanner.get_project_stats()
        
        return {
            "missingFiles": missing_paths,
            "totalMissing": total_missing,
            "returnedCount": len(missing_paths),
            "existingDescriptions": len(existing_paths),
            "projectStats": stats
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
    
    async def _handle_merge_branch_descriptions(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle merge_branch_descriptions tool calls."""
        project_id = await self._get_or_create_project_id(arguments)
        source_branch = arguments["sourceBranch"]
        target_branch = arguments["targetBranch"]
        conflict_resolutions = arguments.get("conflictResolutions")
        
        if conflict_resolutions is None:
            # Phase 1: Detect conflicts
            session = await self.merge_handler.start_merge_phase1(
                project_id, source_branch, target_branch
            )
            
            if session.get_conflict_count() == 0:
                # No conflicts, can merge immediately
                return {
                    "phase": "completed",
                    "conflicts": [],
                    "message": f"No conflicts detected. Merge from {source_branch} to {target_branch} can proceed automatically.",
                    "sourceBranch": source_branch,
                    "targetBranch": target_branch,
                    "conflictCount": 0
                }
            else:
                # Return conflicts for resolution
                return {
                    "phase": "conflicts_detected",
                    "sessionId": session.session_id,
                    "conflicts": [conflict.to_dict() for conflict in session.conflicts],
                    "conflictCount": session.get_conflict_count(),
                    "sourceBranch": source_branch,
                    "targetBranch": target_branch,
                    "message": f"Found {session.get_conflict_count()} conflicts that need resolution."
                }
        else:
            # Phase 2: Apply resolutions
            # Find the session ID from conflict resolutions
            if not conflict_resolutions:
                from ..error_handler import ValidationError
                raise ValidationError("Conflict resolutions required for phase 2")
            
            # For simplicity, create a new session and resolve immediately
            # In a production system, you'd want to track session IDs properly
            session = await self.merge_handler.start_merge_phase1(
                project_id, source_branch, target_branch
            )
            
            if session.get_conflict_count() == 0:
                return {
                    "phase": "completed",
                    "message": "No conflicts to resolve",
                    "sourceBranch": source_branch,
                    "targetBranch": target_branch
                }
            
            result = await self.merge_handler.complete_merge_phase2(
                session.session_id, conflict_resolutions
            )
            
            return {
                "phase": "completed",
                **result
            }
    
    async def _run_session_with_retry(self, read_stream, write_stream, initialization_options) -> None:
        """Run a single MCP session with error handling and retry logic."""
        max_retries = 3
        base_delay = 1.0  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Starting MCP server protocol session (attempt {attempt + 1})...")
                await self.server.run(
                    read_stream,
                    write_stream, 
                    initialization_options
                )
                logger.info("MCP server session completed normally")
                return  # Success, exit retry loop
                
            except ValidationError as e:
                # Handle malformed requests gracefully
                logger.warning(f"Received malformed request (attempt {attempt + 1}): {e}", extra={
                    "structured_data": {
                        "error_type": "ValidationError",
                        "validation_errors": e.errors() if hasattr(e, 'errors') else str(e),
                        "attempt": attempt + 1,
                        "max_retries": max_retries
                    }
                })
                
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries exceeded for validation errors. Server will continue but this session failed.")
                    return
                    
            except (ConnectionError, BrokenPipeError, EOFError) as e:
                # Handle client disconnection gracefully
                logger.info(f"Client disconnected: {e}")
                return
                
            except Exception as e:
                # Handle other exceptions with full logging
                import traceback
                if "unhandled errors in a TaskGroup" in str(e) and "ValidationError" in str(e):
                    # This is likely a ValidationError wrapped in a TaskGroup exception
                    logger.warning(f"Detected wrapped validation error (attempt {attempt + 1}): {e}", extra={
                        "structured_data": {
                            "error_type": type(e).__name__, 
                            "error_message": str(e),
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "likely_validation_error": True
                        }
                    })
                    
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.info(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error("Max retries exceeded for validation errors. Server will continue but this session failed.")
                        return
                else:
                    # This is a genuine error, log and re-raise
                    logger.error(f"MCP server session error: {e}", extra={
                        "structured_data": {
                            "error_type": type(e).__name__, 
                            "error_message": str(e),
                            "traceback": traceback.format_exc()
                        }
                    })
                    raise

    async def run(self) -> None:
        """Run the MCP server with robust error handling."""
        logger.info("Starting server initialization...")
        await self.initialize()
        logger.info("Server initialization completed, starting MCP protocol...")
        
        max_retries = 5
        base_delay = 2.0  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                async with stdio_server() as (read_stream, write_stream):
                    logger.info(f"stdio_server context established (attempt {attempt + 1})")
                    initialization_options = self.server.create_initialization_options()
                    logger.debug(f"Initialization options: {initialization_options}")
                    
                    await self._run_session_with_retry(read_stream, write_stream, initialization_options)
                    return  # Success, exit retry loop
                        
            except KeyboardInterrupt:
                logger.info("Server stopped by user interrupt")
                return
                
            except Exception as e:
                import traceback
                
                # Check if this is a wrapped validation error
                error_str = str(e)
                is_validation_error = (
                    "ValidationError" in error_str or 
                    "Field required" in error_str or 
                    "Input should be" in error_str or
                    "pydantic_core._pydantic_core.ValidationError" in error_str
                )
                
                if is_validation_error:
                    logger.warning(f"Detected validation error in session (attempt {attempt + 1}): Malformed client request", extra={
                        "structured_data": {
                            "error_type": "ValidationError", 
                            "error_message": "Client sent malformed request (likely missing clientInfo)",
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "will_retry": attempt < max_retries
                        }
                    })
                    
                    if attempt < max_retries:
                        delay = base_delay * (2 ** min(attempt, 3))  # Cap exponential growth
                        logger.info(f"Retrying server in {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.warning("Max retries exceeded for validation errors. Server is robust against malformed requests.")
                        return
                else:
                    # This is a genuine fatal error
                    logger.error(f"Fatal server error: {e}", extra={
                        "structured_data": {
                            "error_type": type(e).__name__, 
                            "error_message": str(e),
                            "traceback": traceback.format_exc()
                        }
                    })
                    raise
        
        # Clean shutdown
        await self.shutdown()
    
    async def shutdown(self) -> None:
        """Clean shutdown of server resources."""
        try:
            # Cancel any running tasks
            self.task_manager.cancel_all()
            
            # Close database connections
            await self.db_manager.close_pool()
            
            self.logger.info("Server shutdown completed successfully")
            
        except Exception as e:
            self.error_handler.log_error(e, context={"phase": "shutdown"})


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
