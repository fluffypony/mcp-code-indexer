"""
Error handling middleware for MCP tools.

This module provides decorators and middleware functions to standardize
error handling across all MCP tool implementations.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, List

import aiosqlite
from mcp import types

from mcp_code_indexer.error_handler import ErrorHandler
from mcp_code_indexer.logging_config import (
    get_logger,
    log_tool_usage,
    log_performance_metrics,
)

logger = get_logger(__name__)


class ToolMiddleware:
    """Middleware for MCP tool error handling and logging."""

    def __init__(self, error_handler: ErrorHandler):
        """Initialize middleware with error handler."""
        self.error_handler = error_handler

    def wrap_tool_handler(self, tool_name: str):
        """
        Decorator to wrap tool handlers with error handling and logging.

        Args:
            tool_name: Name of the MCP tool

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(arguments: Dict[str, Any]) -> List[types.TextContent]:
                start_time = time.time()
                success = False
                result_size = 0

                try:
                    # Log tool invocation
                    logger.info(
                        f"Tool {tool_name} called",
                        extra={
                            "structured_data": {
                                "tool_invocation": {
                                    "tool_name": tool_name,
                                    "arguments_count": len(arguments),
                                }
                            }
                        },
                    )

                    # Call the actual tool handler
                    result = await func(arguments)

                    # Calculate result size
                    if isinstance(result, list):
                        result_size = sum(
                            len(item.text) if hasattr(item, "text") else 0
                            for item in result
                        )

                    success = True
                    duration = time.time() - start_time

                    # Log performance metrics
                    log_performance_metrics(
                        logger,
                        f"tool_{tool_name}",
                        duration,
                        result_size=result_size,
                        arguments_count=len(arguments),
                    )

                    return result

                except Exception as e:
                    duration = time.time() - start_time

                    # Enhanced SQLite error handling
                    if self._is_database_locking_error(e):
                        logger.warning(
                            f"Database locking error in tool {tool_name}: {e}",
                            extra={
                                "structured_data": {
                                    "database_locking_error": {
                                        "tool_name": tool_name,
                                        "error_type": type(e).__name__,
                                        "error_message": str(e),
                                        "duration": duration,
                                    }
                                }
                            },
                        )

                    # Log the error
                    self.error_handler.log_error(
                        e,
                        context={"arguments_count": len(arguments)},
                        tool_name=tool_name,
                    )

                    # Create error response
                    error_response = self.error_handler.create_mcp_error_response(
                        e, tool_name, arguments
                    )

                    return [error_response]

                finally:
                    # Always log tool usage
                    log_tool_usage(
                        logger,
                        tool_name,
                        arguments,
                        success,
                        time.time() - start_time,
                        result_size if success else None,
                    )

            return wrapper

        return decorator

    def validate_tool_arguments(
        self, required_fields: List[str], optional_fields: List[str] = None
    ):
        """
        Decorator to validate tool arguments.

        Args:
            required_fields: List of required argument names
            optional_fields: List of optional argument names

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(arguments: Dict[str, Any]) -> Any:
                from ..error_handler import ValidationError

                # Check required fields
                missing_fields = [
                    field for field in required_fields if field not in arguments
                ]
                if missing_fields:
                    raise ValidationError(
                        f"Missing required fields: {', '.join(missing_fields)}",
                        details={
                            "missing_fields": missing_fields,
                            "provided_fields": list(arguments.keys()),
                        },
                    )

                # Check for unexpected fields if optional_fields is provided
                if optional_fields is not None:
                    all_fields = set(required_fields + optional_fields)
                    unexpected_fields = [
                        field for field in arguments.keys() if field not in all_fields
                    ]
                    if unexpected_fields:
                        raise ValidationError(
                            f"Unexpected fields: {', '.join(unexpected_fields)}",
                            details={
                                "unexpected_fields": unexpected_fields,
                                "allowed_fields": list(all_fields),
                            },
                        )

                return await func(arguments)

            return wrapper

        return decorator

    def _is_database_locking_error(self, error: Exception) -> bool:
        """
        Check if an error is related to database locking.

        Args:
            error: Exception to check

        Returns:
            True if this is a database locking error
        """
        # Check for SQLite locking errors
        if isinstance(error, aiosqlite.OperationalError):
            error_message = str(error).lower()
            locking_keywords = [
                "database is locked",
                "database is busy",
                "sqlite_busy",
                "sqlite_locked",
                "cannot start a transaction within a transaction",
            ]
            return any(keyword in error_message for keyword in locking_keywords)

        return False


class AsyncTaskManager:
    """Manages async tasks with proper error handling."""

    def __init__(self, error_handler: ErrorHandler):
        """Initialize task manager."""
        self.error_handler = error_handler
        self._tasks: List[asyncio.Task] = []

    def create_task(self, coro, name: str = None) -> asyncio.Task:
        """
        Create a managed async task.

        Args:
            coro: Coroutine to run
            name: Optional task name for logging

        Returns:
            Created task
        """
        task = asyncio.create_task(coro, name=name)
        self._tasks.append(task)

        # Add done callback for error handling
        task.add_done_callback(
            lambda t: asyncio.create_task(
                self._handle_task_completion(t, name or "unnamed_task")
            )
        )

        return task

    async def _handle_task_completion(self, task: asyncio.Task, task_name: str) -> None:
        """Handle task completion and errors."""
        try:
            if task.done() and not task.cancelled():
                exception = task.exception()
                if exception:
                    await self.error_handler.handle_async_task_error(task, task_name)
        except Exception as e:
            logger.error(f"Error handling task completion for {task_name}: {e}")
        finally:
            # Remove completed task from tracking
            if task in self._tasks:
                self._tasks.remove(task)

    async def wait_for_all(self, timeout: float = None) -> None:
        """
        Wait for all managed tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds
        """
        if not self._tasks:
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for {len(self._tasks)} tasks")
            # Cancel remaining tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
        except Exception as e:
            logger.error(f"Error waiting for tasks: {e}")

    def cancel_all(self) -> None:
        """Cancel all managed tasks."""
        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()

    @property
    def active_task_count(self) -> int:
        """Get count of active tasks."""
        return len([task for task in self._tasks if not task.done()])


def create_tool_middleware(error_handler: ErrorHandler) -> ToolMiddleware:
    """
    Create tool middleware instance.

    Args:
        error_handler: Error handler instance

    Returns:
        Configured ToolMiddleware
    """
    return ToolMiddleware(error_handler)


# Convenience decorators for common patterns


def require_fields(*required_fields):
    """Decorator that requires specific fields in arguments."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, arguments: Dict[str, Any]):
            from ..error_handler import ValidationError

            missing = [field for field in required_fields if field not in arguments]
            if missing:
                raise ValidationError(f"Missing required fields: {', '.join(missing)}")

            return await func(self, arguments)

        return wrapper

    return decorator


def handle_file_operations(func):
    """Decorator for file operation error handling."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (FileNotFoundError, PermissionError, OSError) as e:
            from ..error_handler import FileSystemError

            raise FileSystemError(f"File operation failed: {e}") from e

    return wrapper


def handle_database_operations(func):
    """Decorator for database operation error handling."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if any(
                keyword in str(e).lower() for keyword in ["database", "sqlite", "sql"]
            ):
                from ..error_handler import DatabaseError

                raise DatabaseError(f"Database operation failed: {e}") from e
            raise

    return wrapper
