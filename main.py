#!/usr/bin/env python3
"""
MCP Code Indexer Server Entry Point

This script initializes and runs the MCP code indexer server with configurable
options for token limits, database paths, and cache directories.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from src.mcp_code_indexer import __version__
from src.mcp_code_indexer.logging_config import setup_logging
from src.mcp_code_indexer.error_handler import setup_error_handling


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP Code Index Server - Track file descriptions across codebases",
        prog="mcp-code-indexer",
    )

    parser.add_argument(
        "--version", action="version", version=f"mcp-code-indexer {__version__}"
    )

    parser.add_argument(
        "--token-limit",
        type=int,
        default=32000,
        help=(
            "Maximum tokens before recommending search instead of full overview "
            "(default: 32000)"
        ),
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="~/.mcp-code-index/tracker.db",
        help="Path to SQLite database (default: ~/.mcp-code-index/tracker.db)",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.mcp-code-index/cache",
        help="Directory for caching token counts (default: ~/.mcp-code-index/cache)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--githook",
        action="store_true",
        help=(
            "Git hook mode: auto-update descriptions based on git diff using "
            "OpenRouter API"
        ),
    )

    parser.add_argument(
        "--ask",
        type=str,
        help=(
            "Ask a question about the project (requires PROJECT_NAME as "
            "positional argument)"
        ),
    )

    parser.add_argument(
        "--deepask",
        type=str,
        help=(
            "Ask an enhanced question with file search (requires PROJECT_NAME "
            "as positional argument)"
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output response in JSON format (for --ask and --deepask commands)",
    )

    parser.add_argument(
        "--makelocal",
        type=str,
        help="Create local database in specified folder and migrate project data from global DB",
    )

    parser.add_argument(
        "project_name", nargs="?", help="Project name for --ask and --deepask commands"
    )

    # Database configuration options
    parser.add_argument(
        "--db-pool-size",
        type=int,
        default=int(os.getenv("DB_POOL_SIZE", "3")),
        help="Database connection pool size (default: 3, env: DB_POOL_SIZE)",
    )

    parser.add_argument(
        "--db-retry-count",
        type=int,
        default=int(os.getenv("DB_RETRY_COUNT", "5")),
        help=(
            "Maximum database operation retry attempts "
            "(default: 5, env: DB_RETRY_COUNT)"
        ),
    )

    parser.add_argument(
        "--db-timeout",
        type=float,
        default=float(os.getenv("DB_TIMEOUT", "10.0")),
        help=(
            "Database transaction timeout in seconds "
            "(default: 10.0, env: DB_TIMEOUT)"
        ),
    )

    parser.add_argument(
        "--enable-wal-mode",
        action="store_true",
        default=os.getenv("DB_WAL_MODE", "true").lower() == "true",
        help=(
            "Enable WAL mode for better concurrent access "
            "(default: True, env: DB_WAL_MODE)"
        ),
    )

    parser.add_argument(
        "--health-check-interval",
        type=float,
        default=float(os.getenv("DB_HEALTH_CHECK_INTERVAL", "30.0")),
        help=(
            "Database health check interval in seconds "
            "(default: 30.0, env: DB_HEALTH_CHECK_INTERVAL)"
        ),
    )

    # Retry executor configuration options
    parser.add_argument(
        "--retry-min-wait",
        type=float,
        default=float(os.getenv("DB_RETRY_MIN_WAIT", "0.1")),
        help=(
            "Minimum wait time between retries in seconds "
            "(default: 0.1, env: DB_RETRY_MIN_WAIT)"
        ),
    )

    parser.add_argument(
        "--retry-max-wait",
        type=float,
        default=float(os.getenv("DB_RETRY_MAX_WAIT", "2.0")),
        help=(
            "Maximum wait time between retries in seconds "
            "(default: 2.0, env: DB_RETRY_MAX_WAIT)"
        ),
    )

    parser.add_argument(
        "--retry-jitter",
        type=float,
        default=float(os.getenv("DB_RETRY_JITTER", "0.2")),
        help=(
            "Maximum jitter to add to retry delays in seconds "
            "(default: 0.2, env: DB_RETRY_JITTER)"
        ),
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

        db_manager = DatabaseManager(
            db_path,
            pool_size=args.db_pool_size,
            retry_count=args.db_retry_count,
            timeout=args.db_timeout,
            enable_wal_mode=args.enable_wal_mode,
            health_check_interval=args.health_check_interval,
            retry_min_wait=args.retry_min_wait,
            retry_max_wait=args.retry_max_wait,
            retry_jitter=args.retry_jitter,
        )
        await db_manager.initialize()

        # Initialize git hook handler
        git_handler = GitHookHandler(db_manager, cache_dir)

        # Run git hook analysis
        await git_handler.run_githook_mode()

    except Exception as e:
        print(f"Git hook error: {e}", file=sys.stderr)
        sys.exit(1)


async def handle_ask(args: argparse.Namespace) -> None:
    """Handle --ask command."""
    try:
        from src.mcp_code_indexer.database.database import DatabaseManager
        from src.mcp_code_indexer.ask_handler import AskHandler

        # Validate arguments
        if not args.project_name:
            print("Error: PROJECT_NAME is required for --ask command", file=sys.stderr)
            sys.exit(1)

        # Initialize database
        db_path = Path(args.db_path).expanduser()
        cache_dir = Path(args.cache_dir).expanduser()

        # Create directories if they don't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup command-specific logger with file output
        logger = logging.getLogger("ask_command")
        logger.setLevel(logging.INFO)

        # Clear any existing handlers
        logger.handlers = []

        # Add file handler
        log_file = cache_dir / "ask.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        # Only add console handler for ERROR level and above
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)

        db_manager = DatabaseManager(
            db_path,
            pool_size=args.db_pool_size,
            retry_count=args.db_retry_count,
            timeout=args.db_timeout,
            enable_wal_mode=args.enable_wal_mode,
            health_check_interval=args.health_check_interval,
            retry_min_wait=args.retry_min_wait,
            retry_max_wait=args.retry_max_wait,
            retry_jitter=args.retry_jitter,
        )
        await db_manager.initialize()

        # Initialize ask handler
        ask_handler = AskHandler(db_manager, cache_dir, logger)

        # Resolve project info - search by name only for CLI Q&A
        project_info = {
            "projectName": args.project_name,
            "remoteOrigin": None,
            "upstreamOrigin": None,
        }

        # Process the question
        result = await ask_handler.ask_question(project_info, args.ask)

        # Format and output response - clean output for CLI
        if args.json:
            import json

            print(json.dumps(result, indent=2))
        else:
            # Just print the answer, no metadata
            print(result["answer"])

        # Stop background tasks
        await db_manager._health_monitor.stop_monitoring()
        await db_manager.close_pool()

    except Exception as e:
        print(f"Ask command error: {e}", file=sys.stderr)
        sys.exit(1)


async def handle_deepask(args: argparse.Namespace) -> None:
    """Handle --deepask command."""
    try:
        from src.mcp_code_indexer.database.database import DatabaseManager
        from src.mcp_code_indexer.deepask_handler import DeepAskHandler

        # Validate arguments
        if not args.project_name:
            print(
                "Error: PROJECT_NAME is required for --deepask command", file=sys.stderr
            )
            sys.exit(1)

        # Initialize database
        db_path = Path(args.db_path).expanduser()
        cache_dir = Path(args.cache_dir).expanduser()

        # Create directories if they don't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup command-specific logger with file output
        logger = logging.getLogger("deepask_command")
        logger.setLevel(logging.INFO)

        # Clear any existing handlers
        logger.handlers = []

        # Add file handler
        log_file = cache_dir / "deepask.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        # Only add console handler for ERROR level and above
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)

        db_manager = DatabaseManager(
            db_path,
            pool_size=args.db_pool_size,
            retry_count=args.db_retry_count,
            timeout=args.db_timeout,
            enable_wal_mode=args.enable_wal_mode,
            health_check_interval=args.health_check_interval,
            retry_min_wait=args.retry_min_wait,
            retry_max_wait=args.retry_max_wait,
            retry_jitter=args.retry_jitter,
        )
        await db_manager.initialize()

        # Initialize deepask handler
        deepask_handler = DeepAskHandler(db_manager, cache_dir, logger)

        # Resolve project info - search by name only for CLI Q&A
        project_info = {
            "projectName": args.project_name,
            "remoteOrigin": None,
            "upstreamOrigin": None,
        }

        # Process the question
        result = await deepask_handler.deepask_question(project_info, args.deepask)

        # Format and output response - clean output for CLI
        if args.json:
            import json

            print(json.dumps(result, indent=2))
        else:
            # Just print the answer, no metadata
            print(result["answer"])

        # Stop background tasks
        await db_manager._health_monitor.stop_monitoring()
        await db_manager.close_pool()

    except Exception as e:
        print(f"DeepAsk command error: {e}", file=sys.stderr)
        sys.exit(1)


async def handle_makelocal(args: argparse.Namespace) -> None:
    """Handle --makelocal command."""
    try:
        from src.mcp_code_indexer.database.database_factory import DatabaseFactory
        from src.mcp_code_indexer.commands.makelocal import MakeLocalCommand

        # Initialize database factory
        db_path = Path(args.db_path).expanduser()
        cache_dir = Path(args.cache_dir).expanduser()

        # Create directories if they don't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        db_factory = DatabaseFactory(
            global_db_path=db_path,
            pool_size=args.db_pool_size,
            retry_count=args.db_retry_count,
            timeout=args.db_timeout,
            enable_wal_mode=args.enable_wal_mode,
            health_check_interval=args.health_check_interval,
            retry_min_wait=args.retry_min_wait,
            retry_max_wait=args.retry_max_wait,
            retry_jitter=args.retry_jitter,
        )

        # Initialize make local command
        makelocal_cmd = MakeLocalCommand(db_factory)

        # Execute the command
        result = await makelocal_cmd.execute(args.makelocal)

        print(f"Successfully migrated project '{result['project_name']}' to local database")
        print(f"Local database created at: {result['local_database_path']}")
        print(f"Migrated {result['migrated_files']} file descriptions")
        if result['migrated_overview']:
            print("Migrated project overview")

        # Close all database connections
        await db_factory.close_all()

    except Exception as e:
        print(f"Make local command error: {e}", file=sys.stderr)
        sys.exit(1)


async def main() -> None:
    """Main entry point for the MCP server."""
    args = parse_arguments()

    # Handle command modes
    if args.githook:
        await handle_githook(args)
        return

    if args.ask:
        await handle_ask(args)
        return

    if args.deepask:
        await handle_deepask(args)
        return

    if args.makelocal:
        await handle_makelocal(args)
        return

    # Setup structured logging
    log_file = (
        Path(args.cache_dir).expanduser() / "server.log" if args.cache_dir else None
    )
    logger = setup_logging(
        log_level=args.log_level, log_file=log_file, enable_file_logging=True
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
    logger.info(
        "Starting MCP Code Index Server",
        extra={
            "structured_data": {
                "startup": {
                    "token_limit": args.token_limit,
                    "db_path": str(db_path),
                    "cache_dir": str(cache_dir),
                    "log_level": args.log_level,
                    "database_config": {
                        "pool_size": args.db_pool_size,
                        "retry_count": args.db_retry_count,
                        "timeout": args.db_timeout,
                        "wal_mode": args.enable_wal_mode,
                        "health_check_interval": args.health_check_interval,
                        "retry_min_wait": args.retry_min_wait,
                        "retry_max_wait": args.retry_max_wait,
                        "retry_jitter": args.retry_jitter,
                    },
                }
            }
        },
    )

    try:
        # Import and run the MCP server
        from src.mcp_code_indexer.server.mcp_server import MCPCodeIndexServer

        server = MCPCodeIndexServer(
            token_limit=args.token_limit,
            db_path=db_path,
            cache_dir=cache_dir,
            db_pool_size=args.db_pool_size,
            db_retry_count=args.db_retry_count,
            db_timeout=args.db_timeout,
            enable_wal_mode=args.enable_wal_mode,
            health_check_interval=args.health_check_interval,
            retry_min_wait=args.retry_min_wait,
            retry_max_wait=args.retry_max_wait,
            retry_jitter=args.retry_jitter,
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
