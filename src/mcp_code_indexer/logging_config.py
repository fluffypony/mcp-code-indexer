"""
Logging configuration for the MCP Code Indexer.

This module provides centralized logging setup with structured JSON output,
proper async handling, and file rotation for production use.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from .error_handler import StructuredFormatter


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_file_logging: bool = False,
    max_bytes: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 2,
) -> logging.Logger:
    """
    Set up comprehensive logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_file_logging: Whether to enable file logging
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured root logger
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler (stderr to avoid interfering with MCP stdout)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Use structured formatter for all handlers
    structured_formatter = StructuredFormatter()
    console_handler.setFormatter(structured_formatter)

    root_logger.addHandler(console_handler)

    # File handler (optional)
    if enable_file_logging and log_file:
        try:
            # Ensure log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Rotating file handler
            if max_bytes > 0:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
            else:
                # No size limit - use regular FileHandler
                file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # File gets all levels
            file_handler.setFormatter(structured_formatter)

            root_logger.addHandler(file_handler)

        except (OSError, PermissionError) as e:
            # Log to console if file logging fails
            root_logger.warning(f"Failed to set up file logging: {e}")

    # Configure specific loggers

    # Quiet down noisy libraries
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("tiktoken").setLevel(logging.WARNING)

    # MCP specific loggers
    mcp_logger = logging.getLogger("mcp")
    mcp_logger.setLevel(logging.INFO)

    # Database logger
    db_logger = logging.getLogger("src.database")
    db_logger.setLevel(logging.INFO)

    # Server logger
    server_logger = logging.getLogger("src.server")
    server_logger.setLevel(logging.INFO)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_command_logger(
    command_name: str, cache_dir: Path, log_level: str = "DEBUG"
) -> logging.Logger:
    """
    Set up a dedicated logger for specific commands (runcommand, githook).

    Args:
        command_name: Name of the command (e.g., 'runcommand', 'githook')
        cache_dir: Cache directory path
        log_level: Logging level

    Returns:
        Configured logger for the command
    """
    logger_name = f"mcp_code_indexer.{command_name}"
    logger = logging.getLogger(logger_name)

    # Don't propagate to parent loggers to avoid duplicate console output
    logger.propagate = False
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create log file path
    log_file = cache_dir / f"{command_name}.log"

    try:
        # Ensure cache directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)

        # File handler with 50MB limit
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50 * 1024 * 1024, backupCount=2, encoding="utf-8"  # 50MB
        )
        file_handler.setLevel(logging.DEBUG)

        # Use structured formatter
        structured_formatter = StructuredFormatter()
        file_handler.setFormatter(structured_formatter)

        logger.addHandler(file_handler)

        # Set up component loggers to also log to this command's log file
        _setup_component_loggers_for_command(
            command_name, file_handler, structured_formatter
        )

        logger.info(f"=== {command_name.upper()} SESSION STARTED ===")

    except (OSError, PermissionError) as e:
        # Fallback to console logging
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(StructuredFormatter())
        logger.addHandler(console_handler)
        logger.warning(f"Failed to set up {command_name} file logging: {e}")

    return logger


def _setup_component_loggers_for_command(
    command_name: str, file_handler: logging.Handler, formatter: logging.Formatter
) -> None:
    """
    Set up component loggers to also send logs to the command's log file.

    Args:
        command_name: Name of the command
        file_handler: File handler to add to component loggers
        formatter: Formatter to use for the handler
    """
    # List of component logger names that should also log to command files
    component_loggers = [
        "mcp_code_indexer.database.database",
        "mcp_code_indexer.server.mcp_server",
        "mcp_code_indexer.token_counter",
        "mcp_code_indexer.file_scanner",
        "mcp_code_indexer.error_handler",
    ]

    for component_logger_name in component_loggers:
        component_logger = logging.getLogger(component_logger_name)

        # Create a separate handler for this command to avoid interference
        command_handler = logging.handlers.RotatingFileHandler(
            file_handler.baseFilename,
            maxBytes=file_handler.maxBytes,
            backupCount=file_handler.backupCount,
            encoding="utf-8",
        )
        command_handler.setLevel(logging.DEBUG)
        command_handler.setFormatter(formatter)

        # Add a marker to identify which command this handler belongs to
        command_handler._command_name = command_name

        # Remove any existing handlers for this command (in case of multiple calls)
        existing_handlers = [
            h
            for h in component_logger.handlers
            if hasattr(h, "_command_name") and h._command_name == command_name
        ]
        for handler in existing_handlers:
            component_logger.removeHandler(handler)
            handler.close()

        # Add the new handler
        component_logger.addHandler(command_handler)
        component_logger.setLevel(
            logging.DEBUG
        )  # Ensure component loggers capture all levels


def log_performance_metrics(
    logger: logging.Logger, operation: str, duration: float, **metrics
) -> None:
    """
    Log performance metrics in structured format.

    Args:
        logger: Logger instance
        operation: Name of the operation
        duration: Duration in seconds
        **metrics: Additional metrics to log
    """
    perf_data = {
        "operation": operation,
        "duration_seconds": duration,
        "metrics": metrics,
    }

    logger.info(
        f"Performance: {operation} completed in {duration:.3f}s",
        extra={"structured_data": {"performance": perf_data}},
    )


def log_database_metrics(
    logger: logging.Logger,
    operation_name: str,
    metrics: dict,
    health_status: Optional[dict] = None,
) -> None:
    """
    Log database performance and health metrics.

    Args:
        logger: Logger instance
        operation_name: Name of the database operation
        metrics: Database performance metrics
        health_status: Current health status (optional)
    """
    log_data = {"operation": operation_name, "metrics": metrics}

    if health_status:
        log_data["health_status"] = health_status

    logger.info(
        f"Database metrics for {operation_name}",
        extra={"structured_data": {"database_metrics": log_data}},
    )


def log_tool_usage(
    logger: logging.Logger,
    tool_name: str,
    arguments: dict,
    success: bool,
    duration: Optional[float] = None,
    result_size: Optional[int] = None,
) -> None:
    """
    Log MCP tool usage for analytics.

    Args:
        logger: Logger instance
        tool_name: Name of the MCP tool
        arguments: Tool arguments (will be sanitized)
        success: Whether the operation succeeded
        duration: Operation duration in seconds
        result_size: Size of result data
    """
    # Sanitize arguments
    safe_args = {}
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > 50:
            safe_args[key] = f"{value[:50]}..."
        else:
            safe_args[key] = value

    usage_data = {"tool_name": tool_name, "arguments": safe_args, "success": success}

    if duration is not None:
        usage_data["duration_seconds"] = duration

    if result_size is not None:
        usage_data["result_size"] = result_size

    level = logging.INFO if success else logging.WARNING
    message = f"Tool {tool_name}: {'SUCCESS' if success else 'FAILED'}"

    logger.log(level, message, extra={"structured_data": {"tool_usage": usage_data}})
