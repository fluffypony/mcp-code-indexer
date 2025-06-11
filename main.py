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

# Import will be added once server module is implemented
# from src.server import MCPCodeIndexServer


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP Code Index Server - Track file descriptions across codebases"
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
    
    # Setup logging to stderr (stdout is used for MCP communication)
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Expand user paths
    db_path = Path(args.db_path).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    
    # Create directories if they don't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Log to stderr to avoid interfering with MCP communication
    logger.info(f"Starting MCP Code Index Server")
    logger.info(f"Token limit: {args.token_limit}")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Cache directory: {cache_dir}")
    
    # Import and run the MCP server
    from src.server.mcp_server import MCPCodeIndexServer
    
    server = MCPCodeIndexServer(
        token_limit=args.token_limit,
        db_path=db_path,
        cache_dir=cache_dir
    )
    
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server failed to start: {e}")
        sys.exit(1)
