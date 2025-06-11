#!/usr/bin/env python3
"""
MCP Code Indexer Package Main Module

Entry point for the mcp-code-indexer package when installed via pip.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

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
    
    return parser.parse_args()


async def main() -> None:
    """Main entry point for the MCP server."""
    args = parse_arguments()
    
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
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
