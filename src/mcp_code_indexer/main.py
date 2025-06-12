#!/usr/bin/env python3
"""
MCP Code Indexer Package Main Module

Entry point for the mcp-code-indexer package when installed via pip.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from . import __version__
from .logging_config import setup_logging
from .error_handler import setup_error_handling


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP Code Index Server - Track file descriptions across codebases",
        prog="mcp-code-indexer"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"mcp-code-indexer {__version__}"
    )
    
    parser.add_argument(
        "--token-limit",
        type=int,
        default=32000,
        help="Maximum tokens before recommending search instead of full overview (default: 32000)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="~/.mcp-code-index/tracker.db",
        help="Path to SQLite database (default: ~/.mcp-code-index/tracker.db)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.mcp-code-index/cache",
        help="Directory for caching token counts (default: ~/.mcp-code-index/cache)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Utility commands
    parser.add_argument(
        "--getprojects",
        action="store_true",
        help="List all projects with IDs, branches, and description counts"
    )
    
    parser.add_argument(
        "--runcommand",
        type=str,
        help="Execute a command using JSON in MCP format (single or multi-line)"
    )
    
    parser.add_argument(
        "--dumpdescriptions",
        nargs="+",
        metavar=("PROJECT_ID", "BRANCH"),
        help="Export descriptions for a project. Usage: --dumpdescriptions PROJECT_ID [BRANCH]"
    )
    
    return parser.parse_args()


async def handle_getprojects(args: argparse.Namespace) -> None:
    """Handle --getprojects command."""
    try:
        from .database.database import DatabaseManager
        
        # Initialize database
        db_path = Path(args.db_path).expanduser()
        db_manager = DatabaseManager(db_path)
        await db_manager.initialize()
        
        # Get all projects
        projects = await db_manager.get_all_projects()
        
        if not projects:
            print("No projects found.")
            return
        
        print("Projects:")
        print("-" * 80)
        
        for project in projects:
            print(f"ID: {project.id}")
            print(f"Name: {project.name}")
            print(f"Remote Origin: {project.remote_origin or 'N/A'}")
            print(f"Upstream Origin: {project.upstream_origin or 'N/A'}")
            
            # Get branch information
            try:
                branch_counts = await db_manager.get_branch_file_counts(project.id)
                if branch_counts:
                    print("Branches:")
                    for branch, count in branch_counts.items():
                        print(f"  - {branch}: {count} descriptions")
                else:
                    print("Branches: No descriptions found")
            except Exception as e:
                print(f"Branches: Error loading branch info - {e}")
            
            print("-" * 80)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def handle_runcommand(args: argparse.Namespace) -> None:
    """Handle --runcommand command."""
    from .server.mcp_server import MCPCodeIndexServer
    
    try:
        # Parse JSON (handle both single-line and multi-line)
        json_data = json.loads(args.runcommand)
    except json.JSONDecodeError as e:
        print(f"Initial JSON parse failed: {e}", file=sys.stderr)
        
        # Try to repair the JSON
        try:
            import re
            repaired = args.runcommand
            
            # Fix common issues
            # Quote unquoted URLs and paths
            url_pattern = r'("[\w]+"):\s*([a-zA-Z][a-zA-Z0-9+.-]*://[^\s,}]+|/[^\s,}]*)'
            repaired = re.sub(url_pattern, r'\1: "\2"', repaired)
            
            # Quote unquoted values
            unquoted_pattern = r'("[\w]+"):\s*([a-zA-Z0-9_-]+)(?=\s*[,}])'
            repaired = re.sub(unquoted_pattern, r'\1: "\2"', repaired)
            
            # Remove trailing commas
            repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
            
            json_data = json.loads(repaired)
            print(f"JSON repaired successfully", file=sys.stderr)
            print(f"Original: {args.runcommand}", file=sys.stderr)
            print(f"Repaired: {repaired}", file=sys.stderr)
        except json.JSONDecodeError as repair_error:
            print(f"JSON repair also failed: {repair_error}", file=sys.stderr)
            print(f"Original JSON: {args.runcommand}", file=sys.stderr)
            sys.exit(1)
    
    # Initialize server
    db_path = Path(args.db_path).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    
    server = MCPCodeIndexServer(
        token_limit=args.token_limit,
        db_path=db_path,
        cache_dir=cache_dir
    )
    await server.initialize()
    
    # Extract the tool call information from the JSON
    if "method" in json_data and json_data["method"] == "tools/call":
        tool_name = json_data["params"]["name"]
        tool_arguments = json_data["params"]["arguments"]
    elif "projectName" in json_data and "folderPath" in json_data:
        # Auto-detect: user provided just arguments, try to infer the tool
        if "filePath" in json_data and "description" in json_data:
            tool_name = "update_file_description"
            tool_arguments = json_data
            print("Auto-detected tool: update_file_description", file=sys.stderr)
        elif "branch" in json_data:
            tool_name = "check_codebase_size"
            tool_arguments = json_data
            print("Auto-detected tool: check_codebase_size", file=sys.stderr)
        else:
            print("Error: Could not auto-detect tool from arguments. Please use full MCP format:", file=sys.stderr)
            print('{"method": "tools/call", "params": {"name": "TOOL_NAME", "arguments": {...}}}', file=sys.stderr)
            sys.exit(1)
        
        try:
            # Map tool names to handler methods
            tool_handlers = {
                "get_file_description": server._handle_get_file_description,
                "update_file_description": server._handle_update_file_description,
                "check_codebase_size": server._handle_check_codebase_size,
                "find_missing_descriptions": server._handle_find_missing_descriptions,
                "search_descriptions": server._handle_search_descriptions,
                "get_codebase_overview": server._handle_get_codebase_overview,
                "merge_branch_descriptions": server._handle_merge_branch_descriptions,
            }
            
            if tool_name not in tool_handlers:
                error_result = {
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
                print(json.dumps(error_result, indent=2))
                return
            
            # Clean HTML entities from arguments before execution
            def clean_html_entities(text: str) -> str:
                if not text:
                    return text
                import html
                return html.unescape(text)
            
            def clean_arguments(arguments: dict) -> dict:
                cleaned = {}
                for key, value in arguments.items():
                    if isinstance(value, str):
                        cleaned[key] = clean_html_entities(value)
                    elif isinstance(value, list):
                        cleaned[key] = [
                            clean_html_entities(item) if isinstance(item, str) else item
                            for item in value
                        ]
                    elif isinstance(value, dict):
                        cleaned[key] = clean_arguments(value)
                    else:
                        cleaned[key] = value
                return cleaned
            
            cleaned_tool_arguments = clean_arguments(tool_arguments)
            
            # Execute the tool handler directly
            result = await tool_handlers[tool_name](cleaned_tool_arguments)
            print(json.dumps(result, indent=2, default=str))
        except Exception as e:
            error_result = {
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
            print(json.dumps(error_result, indent=2))
    else:
        print("Error: JSON must contain a valid MCP tool call", file=sys.stderr)
        sys.exit(1)


async def handle_dumpdescriptions(args: argparse.Namespace) -> None:
    """Handle --dumpdescriptions command."""
    from .database.database import DatabaseManager
    from .token_counter import TokenCounter
    
    if len(args.dumpdescriptions) < 1:
        print("Error: Project ID is required", file=sys.stderr)
        sys.exit(1)
    
    project_id = args.dumpdescriptions[0]
    branch = args.dumpdescriptions[1] if len(args.dumpdescriptions) > 1 else None
    
    # Initialize database and token counter
    db_path = Path(args.db_path).expanduser()
    db_manager = DatabaseManager(db_path)
    await db_manager.initialize()
    
    token_counter = TokenCounter(args.token_limit)
    
    # Get file descriptions
    if branch:
        file_descriptions = await db_manager.get_all_file_descriptions(
            project_id=project_id,
            branch=branch
        )
        print(f"File descriptions for project {project_id}, branch {branch}:")
    else:
        file_descriptions = await db_manager.get_all_file_descriptions(
            project_id=project_id
        )
        print(f"File descriptions for project {project_id} (all branches):")
    
    print("=" * 80)
    
    if not file_descriptions:
        print("No descriptions found.")
        total_tokens = 0
    else:
        total_tokens = 0
        for desc in file_descriptions:
            print(f"File: {desc.file_path}")
            if branch is None:
                print(f"Branch: {desc.branch}")
            print(f"Description: {desc.description}")
            print("-" * 40)
            
            # Count tokens for this description
            desc_tokens = token_counter.count_file_description_tokens(desc)
            total_tokens += desc_tokens
    
    print("=" * 80)
    print(f"Total descriptions: {len(file_descriptions)}")
    print(f"Total tokens: {total_tokens}")



async def main() -> None:
    """Main entry point for the MCP server."""
    args = parse_arguments()
    
    # Handle utility commands
    if args.getprojects:
        await handle_getprojects(args)
        return
    
    if args.runcommand:
        await handle_runcommand(args)
        return
    
    if args.dumpdescriptions:
        await handle_dumpdescriptions(args)
        return
    
    # Setup structured logging
    log_file = Path(args.cache_dir).expanduser() / "server.log" if args.cache_dir else None
    logger = setup_logging(
        log_level=args.log_level,
        log_file=log_file,
        enable_file_logging=True
    )
    
    # Setup error handling
    error_handler = setup_error_handling(logger)
    
    # Expand user paths
    db_path = Path(args.db_path).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    
    # Create directories if they don't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Log startup information to stderr (stdout reserved for MCP JSON-RPC)
    logger.info("Starting MCP Code Index Server", extra={
        "structured_data": {
            "startup": {
                "version": __version__,
                "token_limit": args.token_limit,
                "db_path": str(db_path),
                "cache_dir": str(cache_dir),
                "log_level": args.log_level
            }
        }
    })
    
    try:
        # Import and run the MCP server
        from .server.mcp_server import MCPCodeIndexServer
        
        server = MCPCodeIndexServer(
            token_limit=args.token_limit,
            db_path=db_path,
            cache_dir=cache_dir
        )
        
        await server.run()
        
    except Exception as e:
        error_handler.log_error(e, context={"phase": "startup"})
        raise


def cli_main():
    """Console script entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # For MCP servers, we should avoid stdout completely
        # The server will log shutdown through stderr
        pass
    except Exception as e:
        # Log critical errors to stderr, not stdout
        import traceback
        print(f"Server failed to start: {e}", file=sys.stderr)
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
