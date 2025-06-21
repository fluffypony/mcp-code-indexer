"""
Comprehensive error handling for the MCP Code Indexer.

This module provides structured error handling with JSON logging,
MCP-compliant error responses, and proper async exception management.
"""

import asyncio
import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from mcp import types


class ErrorCategory(Enum):
    """Categories of errors for better handling and logging."""

    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    VALIDATION = "validation"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RESOURCE = "resource"
    INTERNAL = "internal"


class MCPError(Exception):
    """Base exception for MCP-specific errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        code: int = -32603,  # JSON-RPC internal error
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.code = code
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class DatabaseError(MCPError):
    """Database-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE,
            code=-32603,
            details=details,
        )


class ValidationError(MCPError):
    """Input validation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            code=-32602,  # Invalid params
            details=details,
        )


class FileSystemError(MCPError):
    """File system access errors."""

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if path:
            details["path"] = path

        super().__init__(
            message=message,
            category=ErrorCategory.FILE_SYSTEM,
            code=-32603,
            details=details,
        )


class ResourceError(MCPError):
    """Resource exhaustion or limit errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            code=-32603,
            details=details,
        )


class ErrorHandler:
    """
    Centralized error handling with structured logging and MCP compliance.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize error handler with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
        self._setup_structured_logging()

    def _setup_structured_logging(self) -> None:
        """Configure structured JSON logging."""
        # Create structured formatter
        formatter = StructuredFormatter()

        # Apply to logger handlers
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
    ) -> None:
        """
        Log error with structured format.

        Args:
            error: Exception to log
            context: Additional context information
            tool_name: Name of the tool where error occurred
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if tool_name:
            error_data["tool_name"] = tool_name

        if context:
            error_data["context"] = context

        if isinstance(error, MCPError):
            error_data.update(
                {
                    "category": error.category.value,
                    "code": error.code,
                    "details": error.details,
                }
            )

        # Add traceback for debugging
        error_data["traceback"] = traceback.format_exc()

        self.logger.error("MCP Error occurred", extra={"structured_data": error_data})

    def create_mcp_error_response(
        self, error: Exception, tool_name: str, arguments: Dict[str, Any]
    ) -> types.TextContent:
        """
        Create MCP-compliant error response.

        Args:
            error: Exception that occurred
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            MCP TextContent with error information
        """
        if isinstance(error, MCPError):
            error_response = {
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "category": error.category.value,
                    "details": error.details,
                },
                "tool": tool_name,
                "timestamp": error.timestamp.isoformat(),
            }
        else:
            error_response = {
                "error": {
                    "code": -32603,  # Internal error
                    "message": str(error),
                    "category": ErrorCategory.INTERNAL.value,
                    "details": {"type": type(error).__name__},
                },
                "tool": tool_name,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Add arguments for debugging (excluding sensitive data)
        safe_arguments = self._sanitize_arguments(arguments)
        if safe_arguments:
            error_response["arguments"] = safe_arguments

        import json

        return types.TextContent(
            type="text", text=json.dumps(error_response, indent=2, default=str)
        )

    def _sanitize_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from arguments."""
        sanitized = {}
        sensitive_keys = {"password", "token", "secret", "key", "auth"}

        for key, value in arguments.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 100:
                sanitized[key] = value[:100] + "..."
            else:
                sanitized[key] = value

        return sanitized

    async def handle_async_task_error(
        self,
        task: asyncio.Task,
        task_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Handle errors from async tasks.

        Args:
            task: The completed task
            task_name: Name of the task for logging
            context: Additional context
        """
        try:
            if task.done() and not task.cancelled():
                exception = task.exception()
                if exception:
                    self.log_error(
                        exception,
                        context={**(context or {}), "task_name": task_name},
                        tool_name="async_task",
                    )
        except Exception as e:
            self.logger.error(f"Error handling task error for {task_name}: {e}")


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        import json
        from . import __version__

        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "version": __version__,
        }

        # Add structured data if present
        if hasattr(record, "structured_data"):
            log_data.update(record.structured_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


def setup_error_handling(logger: logging.Logger) -> ErrorHandler:
    """
    Set up comprehensive error handling for the application.

    Args:
        logger: Logger instance to configure

    Returns:
        Configured ErrorHandler instance
    """
    error_handler = ErrorHandler(logger)

    # Set up asyncio exception handler
    def asyncio_exception_handler(loop, context):
        exception = context.get("exception")
        if exception:
            error_handler.log_error(
                exception, context={"asyncio_context": context, "loop": str(loop)}
            )
        else:
            logger.error(f"Asyncio error: {context}")

    # Apply to current event loop if available
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(asyncio_exception_handler)
    except RuntimeError:
        # No running loop, will be set when loop starts
        pass

    return error_handler


# Decorators for common error handling patterns


def handle_database_errors(func):
    """Decorator to handle database errors."""

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if "database" in str(e).lower() or "sqlite" in str(e).lower():
                raise DatabaseError(f"Database operation failed: {e}") from e
            raise

    return wrapper


def handle_file_errors(func):
    """Decorator to handle file system errors."""

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (FileNotFoundError, PermissionError, OSError) as e:
            raise FileSystemError(f"File system error: {e}") from e
        except Exception:
            raise

    return wrapper


def validate_arguments(required_fields: list, optional_fields: list = None):
    """Decorator to validate tool arguments."""

    def decorator(func):
        async def wrapper(self, arguments: Dict[str, Any], *args, **kwargs):
            # Check required fields
            missing_fields = [
                field for field in required_fields if field not in arguments
            ]
            if missing_fields:
                raise ValidationError(
                    f"Missing required fields: {', '.join(missing_fields)}",
                    details={"missing_fields": missing_fields},
                )

            # Check for unexpected fields
            all_fields = set(required_fields + (optional_fields or []))
            unexpected_fields = [
                field for field in arguments.keys() if field not in all_fields
            ]
            if unexpected_fields:
                raise ValidationError(
                    f"Unexpected fields: {', '.join(unexpected_fields)}",
                    details={"unexpected_fields": unexpected_fields},
                )

            return await func(self, arguments, *args, **kwargs)

        return wrapper

    return decorator
