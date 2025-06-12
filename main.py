#!/usr/bin/env python3
"""
MCP Code Indexer Server Entry Point

This script initializes and runs the MCP code indexer server with configurable
options for token limits, database paths, and cache directories.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.mcp_code_indexer import __version__
from src.mcp_code_indexer.logging_config import setup_logging
from src.mcp_code_indexer.error_handler import setup_error_handling


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
    
    parser.add_argument(
        "--githook",
        action="store_true",
        help="Git hook mode: auto-update descriptions based on git diff using OpenRouter API"
    )
    
    return parser.parse_args()


async def handle_githook(args: argparse.Namespace) -> None:
    """Handle --githook command."""
    try:
        from src.mcp_code_indexer.database.database import DatabaseManager
        from src.mcp_code_indexer.git_hook_handler import GitHookHandler
        
        # Initialize database
        db_path = Path(args.db_path).expanduser()
        cache_dir = Path(args.cache_dir).expanduser()
        
        # Create directories if they don't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        db_manager = DatabaseManager(db_path)
        await db_manager.initialize()
        
        # Initialize git hook handler
        git_handler = GitHookHandler(db_manager, cache_dir)
        
        # Run git hook analysis
        await git_handler.run_githook_mode()
        
    except Exception as e:
        print(f"Git hook error: {e}", file=sys.stderr)
        sys.exit(1)


async def main() -> None:
    """Main entry point for the MCP server."""
    args = parse_arguments()
    
    # Handle git hook command
    if args.githook:
        await handle_githook(args)
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
    
    # Log startup information
    logger.info("Starting MCP Code Index Server", extra={
        "structured_data": {
            "startup": {
                "token_limit": args.token_limit,
                "db_path": str(db_path),
                "cache_dir": str(cache_dir),
                "log_level": args.log_level
            }
        }
    })
    
    try:
        # Import and run the MCP server
        from src.mcp_code_indexer.server.mcp_server import MCPCodeIndexServer
        
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
