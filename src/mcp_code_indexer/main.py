#!/usr/bin/env python3
"""
MCP Code Indexer Package Main Module

Entry point for the mcp-code-indexer package when installed via pip.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import __version__
from .logging_config import setup_logging
from .error_handler import setup_error_handling


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

    # Utility commands
    parser.add_argument(
        "--getprojects",
        action="store_true",
        help="List all projects with IDs, branches, and description counts",
    )

    parser.add_argument(
        "--runcommand",
        type=str,
        help="Execute a command using JSON in MCP format (single or multi-line)",
    )

    parser.add_argument(
        "--dumpdescriptions",
        nargs="+",
        metavar=("PROJECT_ID", "BRANCH"),
        help=(
            "Export descriptions for a project. Usage: "
            "--dumpdescriptions PROJECT_ID [BRANCH]"
        ),
    )

    parser.add_argument(
        "--githook",
        nargs="*",
        metavar="COMMIT_HASH",
        help=(
            "Git hook mode: auto-update descriptions based on git diff using "
            "OpenRouter API. Usage: --githook (current changes), --githook HASH "
            "(specific commit), --githook HASH1 HASH2 (commit range from "
            "HASH1 to HASH2)"
        ),
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove empty projects (no descriptions and no project overview)",
    )

    parser.add_argument(
        "--map",
        type=str,
        metavar="PROJECT_NAME_OR_ID",
        help=(
            "Generate a markdown project map for the specified project "
            "(by name or ID)"
        ),
    )

    return parser.parse_args()


async def handle_getprojects(args: argparse.Namespace) -> None:
    """Handle --getprojects command."""
    db_manager = None
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
    finally:
        # Clean up database connections
        if db_manager:
            await db_manager.close_pool()


async def handle_runcommand(args: argparse.Namespace) -> None:
    """Handle --runcommand command."""
    from .server.mcp_server import MCPCodeIndexServer
    from .logging_config import setup_command_logger

    # Set up dedicated logging for runcommand
    cache_dir = Path(args.cache_dir).expanduser()
    logger = setup_command_logger("runcommand", cache_dir)

    logger.info(
        "Starting runcommand execution",
        extra={
            "structured_data": {
                "command": args.runcommand,
                "args": {
                    "token_limit": args.token_limit,
                    "db_path": str(args.db_path),
                    "cache_dir": str(args.cache_dir),
                },
            }
        },
    )

    try:
        # Parse JSON (handle both single-line and multi-line)
        logger.debug("Parsing JSON command")
        json_data = json.loads(args.runcommand)
        logger.debug(
            "JSON parsed successfully",
            extra={"structured_data": {"parsed_json": json_data}},
        )
    except json.JSONDecodeError as e:
        logger.warning(
            "Initial JSON parse failed", extra={"structured_data": {"error": str(e)}}
        )
        print(f"Initial JSON parse failed: {e}", file=sys.stderr)

        # Try to repair the JSON
        logger.debug("Attempting JSON repair")
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
            repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

            json_data = json.loads(repaired)
            logger.info(
                "JSON repaired successfully",
                extra={
                    "structured_data": {
                        "original": args.runcommand,
                        "repaired": repaired,
                    }
                },
            )
            print("JSON repaired successfully", file=sys.stderr)
            print(f"Original: {args.runcommand}", file=sys.stderr)
            print(f"Repaired: {repaired}", file=sys.stderr)
        except json.JSONDecodeError as repair_error:
            logger.error(
                "JSON repair failed",
                extra={
                    "structured_data": {
                        "repair_error": str(repair_error),
                        "original_json": args.runcommand,
                    }
                },
            )
            print(f"JSON repair also failed: {repair_error}", file=sys.stderr)
            print(f"Original JSON: {args.runcommand}", file=sys.stderr)
            sys.exit(1)

    # Initialize server
    db_path = Path(args.db_path).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()

    logger.info(
        "Initializing MCP server",
        extra={
            "structured_data": {
                "db_path": str(db_path),
                "cache_dir": str(cache_dir),
                "token_limit": args.token_limit,
            }
        },
    )

    server = MCPCodeIndexServer(
        token_limit=args.token_limit, db_path=db_path, cache_dir=cache_dir
    )

    try:
        logger.debug("Initializing server database connection")
        await server.initialize()
        logger.debug("Server initialized successfully")

        # Extract the tool call information from the JSON
        if "method" in json_data and json_data["method"] == "tools/call":
            tool_name = json_data["params"]["name"]
            tool_arguments = json_data["params"]["arguments"]
            logger.info(
                "JSON-RPC format detected",
                extra={
                    "structured_data": {
                        "tool_name": tool_name,
                        "arguments_keys": list(tool_arguments.keys()),
                    }
                },
            )
        elif "projectName" in json_data and "folderPath" in json_data:
            # Auto-detect: user provided just arguments, try to infer the tool
            if "filePath" in json_data and "description" in json_data:
                tool_name = "update_file_description"
                tool_arguments = json_data
                logger.info("Auto-detected tool: update_file_description")
                print("Auto-detected tool: update_file_description", file=sys.stderr)
            elif "branch" in json_data:
                tool_name = "check_codebase_size"
                tool_arguments = json_data
                logger.info("Auto-detected tool: check_codebase_size")
                print("Auto-detected tool: check_codebase_size", file=sys.stderr)
            else:
                logger.error(
                    "Could not auto-detect tool from arguments",
                    extra={
                        "structured_data": {"provided_keys": list(json_data.keys())}
                    },
                )
                print(
                    "Error: Could not auto-detect tool from arguments. "
                    "Please use full MCP format:",
                    file=sys.stderr,
                )
                print(
                    '{"method": "tools/call", "params": '
                    '{"name": "TOOL_NAME", "arguments": {...}}}',
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            logger.error(
                "Invalid JSON format",
                extra={"structured_data": {"provided_keys": list(json_data.keys())}},
            )
            print("Error: JSON must contain a valid MCP tool call", file=sys.stderr)
            sys.exit(1)

        # Map tool names to handler methods - use the same mapping as MCP server
        tool_handlers = {
            "get_file_description": server._handle_get_file_description,
            "update_file_description": server._handle_update_file_description,
            "check_codebase_size": server._handle_check_codebase_size,
            "find_missing_descriptions": server._handle_find_missing_descriptions,
            "search_descriptions": server._handle_search_descriptions,
            "get_all_descriptions": server._handle_get_codebase_overview,
            "get_codebase_overview": server._handle_get_condensed_overview,
            "update_codebase_overview": server._handle_update_codebase_overview,
            "get_word_frequency": server._handle_get_word_frequency,
            "search_codebase_overview": server._handle_search_codebase_overview,
        }

        if tool_name not in tool_handlers:
            logger.error(
                "Unknown tool requested",
                extra={
                    "structured_data": {
                        "tool_name": tool_name,
                        "available_tools": list(tool_handlers.keys()),
                    }
                },
            )
            error_result = {
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
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

        logger.info(
            "Executing tool",
            extra={
                "structured_data": {
                    "tool_name": tool_name,
                    "arguments": {
                        k: v
                        for k, v in cleaned_tool_arguments.items()
                        if k not in ["description"]
                    },  # Exclude long descriptions
                }
            },
        )

        # Execute the tool handler directly
        import time

        start_time = time.time()
        result = await tool_handlers[tool_name](cleaned_tool_arguments)
        execution_time = time.time() - start_time

        logger.info(
            "Tool execution completed",
            extra={
                "structured_data": {
                    "tool_name": tool_name,
                    "execution_time_seconds": execution_time,
                    "result_type": type(result).__name__,
                    "result_size": (
                        len(json.dumps(result, default=str)) if result else 0
                    ),
                }
            },
        )

        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        logger.error(
            "Tool execution failed",
            extra={
                "structured_data": {
                    "tool_name": tool_name if "tool_name" in locals() else "unknown",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            },
        )
        error_result = {"error": {"code": -32603, "message": str(e)}}
        print(json.dumps(error_result, indent=2))
    finally:
        # Clean up database connections
        if hasattr(server, "db_manager") and server.db_manager:
            logger.debug("Closing database connections")
            await server.db_manager.close_pool()
            logger.debug("Database connections closed")
        logger.info("=== RUNCOMMAND SESSION ENDED ===")

        # Close logger handlers to flush any remaining logs
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


async def handle_dumpdescriptions(args: argparse.Namespace) -> None:
    """Handle --dumpdescriptions command."""
    from .database.database import DatabaseManager
    from .token_counter import TokenCounter

    if len(args.dumpdescriptions) < 1:
        print("Error: Project ID is required", file=sys.stderr)
        sys.exit(1)

    project_id = args.dumpdescriptions[0]
    branch = args.dumpdescriptions[1] if len(args.dumpdescriptions) > 1 else None

    db_manager = None
    try:
        # Initialize database and token counter
        db_path = Path(args.db_path).expanduser()
        db_manager = DatabaseManager(db_path)
        await db_manager.initialize()

        token_counter = TokenCounter(args.token_limit)

        # Get file descriptions
        if branch:
            file_descriptions = await db_manager.get_all_file_descriptions(
                project_id=project_id, branch=branch
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

    finally:
        # Clean up database connections
        if db_manager:
            await db_manager.close_pool()


async def handle_githook(args: argparse.Namespace) -> None:
    """Handle --githook command."""
    from .logging_config import setup_command_logger

    # Set up dedicated logging for githook
    cache_dir = Path(args.cache_dir).expanduser()
    logger = setup_command_logger("githook", cache_dir)

    try:
        from .database.database import DatabaseManager
        from .git_hook_handler import GitHookHandler

        # Process commit hash arguments
        commit_hashes = args.githook if args.githook else []

        logger.info(
            "Starting git hook execution",
            extra={
                "structured_data": {
                    "args": {
                        "db_path": str(args.db_path),
                        "cache_dir": str(args.cache_dir),
                        "token_limit": args.token_limit,
                        "commit_hashes": commit_hashes,
                    }
                }
            },
        )

        # Initialize database
        db_path = Path(args.db_path).expanduser()
        cache_dir = Path(args.cache_dir).expanduser()

        logger.info(
            "Setting up directories and database",
            extra={
                "structured_data": {
                    "db_path": str(db_path),
                    "cache_dir": str(cache_dir),
                }
            },
        )

        # Create directories if they don't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        db_manager = DatabaseManager(db_path)
        await db_manager.initialize()
        logger.debug("Database initialized successfully")

        # Initialize git hook handler
        git_handler = GitHookHandler(db_manager, cache_dir, logger)
        logger.debug("Git hook handler initialized")

        # Run git hook analysis
        logger.info("Starting git hook analysis")
        if len(commit_hashes) == 0:
            # Process current staged changes
            await git_handler.run_githook_mode()
        elif len(commit_hashes) == 1:
            # Process specific commit
            await git_handler.run_githook_mode(commit_hash=commit_hashes[0])
        elif len(commit_hashes) == 2:
            # Process commit range
            await git_handler.run_githook_mode(
                commit_range=(commit_hashes[0], commit_hashes[1])
            )
        else:
            raise ValueError("--githook accepts 0, 1, or 2 commit hashes")
        logger.info("Git hook analysis completed successfully")

    except Exception as e:
        logger.error(
            "Git hook execution failed",
            extra={
                "structured_data": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            },
        )
        print(f"Git hook error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up database connections
        if "db_manager" in locals():
            try:
                await db_manager.close_pool()
                logger.debug("Database connections closed")
            except Exception as e:
                logger.warning(f"Error closing database connections: {e}")

        logger.info("=== GITHOOK SESSION ENDED ===")

        # Close logger handlers to flush any remaining logs
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


async def handle_cleanup(args: argparse.Namespace) -> None:
    """Handle --cleanup command."""
    from .logging_config import setup_command_logger

    # Set up dedicated logging for cleanup
    cache_dir = Path(args.cache_dir).expanduser()
    logger = setup_command_logger("cleanup", cache_dir)

    db_manager = None
    try:
        from .database.database import DatabaseManager

        logger.info(
            "Starting database cleanup",
            extra={
                "structured_data": {
                    "args": {
                        "db_path": str(args.db_path),
                        "cache_dir": str(args.cache_dir),
                    }
                }
            },
        )

        # Initialize database
        db_path = Path(args.db_path).expanduser()
        db_manager = DatabaseManager(db_path)
        await db_manager.initialize()
        logger.debug("Database initialized successfully")

        # Perform cleanup
        logger.info("Removing empty projects")
        removed_count = await db_manager.cleanup_empty_projects()

        if removed_count > 0:
            print(f"Removed {removed_count} empty project(s)")
            logger.info(
                "Cleanup completed",
                extra={"structured_data": {"removed_projects": removed_count}},
            )
        else:
            print("No empty projects found")
            logger.info("No empty projects found")

    except Exception as e:
        logger.error(
            "Cleanup failed",
            extra={
                "structured_data": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            },
        )
        print(f"Cleanup error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up database connections
        if db_manager:
            logger.debug("Closing database connections")
            await db_manager.close_pool()
            logger.debug("Database connections closed")
        logger.info("=== CLEANUP SESSION ENDED ===")

        # Close logger handlers to flush any remaining logs
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


async def handle_map(args: argparse.Namespace) -> None:
    """Handle --map command."""
    from .logging_config import setup_command_logger

    # Set up dedicated logging for map
    cache_dir = Path(args.cache_dir).expanduser()
    logger = setup_command_logger("map", cache_dir)

    db_manager = None
    try:
        from .database.database import DatabaseManager

        logger.info(
            "Starting project map generation",
            extra={
                "structured_data": {
                    "project_identifier": args.map,
                    "args": {
                        "db_path": str(args.db_path),
                        "cache_dir": str(args.cache_dir),
                    },
                }
            },
        )

        # Initialize database
        db_path = Path(args.db_path).expanduser()
        db_manager = DatabaseManager(db_path)
        await db_manager.initialize()
        logger.debug("Database initialized successfully")

        # Get project data
        logger.info("Retrieving project data")
        project_data = await db_manager.get_project_map_data(args.map)

        if not project_data:
            print(f"Error: Project '{args.map}' not found", file=sys.stderr)
            logger.error(
                "Project not found", extra={"structured_data": {"identifier": args.map}}
            )
            sys.exit(1)

        project = project_data["project"]
        branch = project_data["branch"]
        overview = project_data["overview"]
        files = project_data["files"]

        logger.info(
            "Generating markdown map",
            extra={
                "structured_data": {
                    "project_name": project.name,
                    "branch": branch,
                    "file_count": len(files),
                    "has_overview": overview is not None,
                }
            },
        )

        # Generate markdown
        markdown_content = generate_project_markdown(
            project, branch, overview, files, logger
        )

        # Output the markdown
        print(markdown_content)

        logger.info("Project map generated successfully")

    except Exception as e:
        logger.error(
            "Map generation failed",
            extra={
                "structured_data": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            },
        )
        print(f"Map generation error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up database connections
        if db_manager:
            logger.debug("Closing database connections")
            await db_manager.close_pool()
            logger.debug("Database connections closed")
        logger.info("=== MAP SESSION ENDED ===")

        # Close logger handlers to flush any remaining logs
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def generate_project_markdown(project, branch, overview, files, logger):
    """Generate the markdown content for the project map."""
    import re
    from collections import defaultdict
    from pathlib import Path as PathLib

    markdown_lines = []

    # Project header with sentence case
    project_name = project.name.title() if project.name.islower() else project.name
    markdown_lines.append(f"# {project_name}")
    markdown_lines.append("")

    # Project metadata
    markdown_lines.append(f"**Branch:** {branch}")
    markdown_lines.append("")

    # Project overview (with header demotion if needed)
    if overview and overview.overview:
        markdown_lines.append("## Project Overview")
        markdown_lines.append("")

        # Check if overview contains H1 headers and demote if needed
        overview_content = overview.overview
        if re.search(r"^#\s", overview_content, re.MULTILINE):
            logger.debug("H1 headers found in overview, demoting all headers")
            # Demote all headers by one level
            overview_content = re.sub(
                r"^(#{1,6})", r"#\1", overview_content, flags=re.MULTILINE
            )

        markdown_lines.append(overview_content)
        markdown_lines.append("")

    # File structure
    if files:
        markdown_lines.append("## Codebase Structure")
        markdown_lines.append("")

        # Organize files by directory
        directories = defaultdict(list)
        for file_desc in files:
            file_path = PathLib(file_desc.file_path)
            if len(file_path.parts) == 1:
                # Root level file
                directories["(root)"].append(file_desc)
            else:
                # File in subdirectory
                directory = str(file_path.parent)
                directories[directory].append(file_desc)

        # Sort directories (root first, then alphabetically)
        sorted_dirs = sorted(
            directories.keys(), key=lambda x: ("" if x == "(root)" else x)
        )

        for directory in sorted_dirs:
            dir_files = directories[directory]

            # Directory header
            if directory == "(root)":
                markdown_lines.append("### Root Directory")
            else:
                # Create nested headers based on directory depth
                depth = len(PathLib(directory).parts)
                header_level = "#" * min(depth + 2, 6)  # Cap at H6
                markdown_lines.append(f"{header_level} {directory}/")

            markdown_lines.append("")

            # Files table
            markdown_lines.append("| File | Description |")
            markdown_lines.append("|------|-------------|")

            for file_desc in sorted(dir_files, key=lambda x: x.file_path):
                file_name = PathLib(file_desc.file_path).name
                # Escape pipe characters in descriptions for markdown table
                description = (
                    file_desc.description.replace("|", "\\|").replace("\n", " ").strip()
                )
                markdown_lines.append(f"| `{file_name}` | {description} |")

            markdown_lines.append("")

    # Footer with generation info
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_lines.append("---")
    markdown_lines.append(f"*Generated by MCP Code Indexer on {timestamp}*")

    return "\n".join(markdown_lines)


async def main() -> None:
    """Main entry point for the MCP server."""
    args = parse_arguments()

    # Handle git hook command
    if args.githook is not None:
        await handle_githook(args)
        return

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

    if args.cleanup:
        await handle_cleanup(args)
        return

    if args.map:
        await handle_map(args)
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

    # Log startup information to stderr (stdout reserved for MCP JSON-RPC)
    logger.info(
        "Starting MCP Code Index Server",
        extra={
            "structured_data": {
                "startup": {
                    "version": __version__,
                    "token_limit": args.token_limit,
                    "db_path": str(db_path),
                    "cache_dir": str(cache_dir),
                    "log_level": args.log_level,
                }
            }
        },
    )

    try:
        # Import and run the MCP server
        from .server.mcp_server import MCPCodeIndexServer

        server = MCPCodeIndexServer(
            token_limit=args.token_limit, db_path=db_path, cache_dir=cache_dir
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
