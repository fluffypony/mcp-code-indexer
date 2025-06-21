"""
Custom exception hierarchy for SQLite errors with retry classification.

This module provides structured error handling for database operations,
with specific exceptions for different types of SQLite errors and
comprehensive error context for monitoring and debugging.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional


class DatabaseError(Exception):
    """Base exception for all database-related errors."""

    def __init__(
        self,
        message: str,
        operation_name: str = "",
        error_context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.operation_name = operation_name
        self.error_context = error_context or {}
        self.timestamp = datetime.now(timezone.utc)
        super().__init__(f"{operation_name}: {message}" if operation_name else message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "operation_name": self.operation_name,
            "timestamp": self.timestamp.isoformat(),
            "error_context": self.error_context,
        }


class DatabaseLockError(DatabaseError):
    """Exception for SQLite database locking issues that are retryable."""

    def __init__(
        self,
        message: str,
        retry_count: int = 0,
        operation_name: str = "",
        last_attempt: Optional[datetime] = None,
        lock_type: str = "unknown",
    ):
        self.retry_count = retry_count
        self.last_attempt = last_attempt or datetime.now(timezone.utc)
        self.lock_type = lock_type  # 'read', 'write', 'exclusive'

        error_context = {
            "retry_count": retry_count,
            "last_attempt": self.last_attempt.isoformat(),
            "lock_type": lock_type,
            "retryable": True,
        }

        super().__init__(message, operation_name, error_context)


class DatabaseBusyError(DatabaseError):
    """Exception for SQLite database busy errors that are retryable."""

    def __init__(
        self,
        message: str,
        operation_name: str = "",
        busy_timeout: float = 0.0,
        resource_type: str = "connection",
    ):
        self.busy_timeout = busy_timeout
        self.resource_type = resource_type  # 'connection', 'transaction', 'table'

        error_context = {
            "busy_timeout": busy_timeout,
            "resource_type": resource_type,
            "retryable": True,
        }

        super().__init__(message, operation_name, error_context)


class DatabaseConnectionError(DatabaseError):
    """Exception for database connection issues."""

    def __init__(
        self,
        message: str,
        operation_name: str = "",
        connection_info: Optional[Dict[str, Any]] = None,
    ):
        self.connection_info = connection_info or {}

        error_context = {
            "connection_info": self.connection_info,
            "retryable": False,  # Connection errors usually indicate config issues
        }

        super().__init__(message, operation_name, error_context)


class DatabaseSchemaError(DatabaseError):
    """Exception for database schema-related errors."""

    def __init__(
        self,
        message: str,
        operation_name: str = "",
        schema_version: Optional[str] = None,
        migration_info: Optional[Dict] = None,
    ):
        self.schema_version = schema_version
        self.migration_info = migration_info or {}

        error_context = {
            "schema_version": schema_version,
            "migration_info": self.migration_info,
            "retryable": False,  # Schema errors require manual intervention
        }

        super().__init__(message, operation_name, error_context)


class DatabaseIntegrityError(DatabaseError):
    """Exception for database integrity constraint violations."""

    def __init__(
        self,
        message: str,
        operation_name: str = "",
        constraint_type: str = "unknown",
        affected_table: str = "",
    ):
        self.constraint_type = (
            constraint_type  # 'primary_key', 'foreign_key', 'unique', 'check'
        )
        self.affected_table = affected_table

        error_context = {
            "constraint_type": constraint_type,
            "affected_table": affected_table,
            "retryable": False,  # Integrity errors indicate data issues
        }

        super().__init__(message, operation_name, error_context)


class DatabaseTimeoutError(DatabaseError):
    """Exception for database operation timeouts."""

    def __init__(
        self,
        message: str,
        operation_name: str = "",
        timeout_seconds: float = 0.0,
        operation_type: str = "unknown",
    ):
        self.timeout_seconds = timeout_seconds
        self.operation_type = operation_type  # 'read', 'write', 'transaction'

        error_context = {
            "timeout_seconds": timeout_seconds,
            "operation_type": operation_type,
            "retryable": True,  # Timeouts might be transient
        }

        super().__init__(message, operation_name, error_context)


def classify_sqlite_error(error: Exception, operation_name: str = "") -> DatabaseError:
    """
    Classify a raw SQLite error into our structured exception hierarchy.

    Args:
        error: Original exception from SQLite
        operation_name: Name of the operation that failed

    Returns:
        Appropriate DatabaseError subclass with context
    """
    error_message = str(error).lower()
    original_message = str(error)

    # Database locking errors
    if any(
        msg in error_message
        for msg in [
            "database is locked",
            "sqlite_locked",
            "attempt to write a readonly database",
        ]
    ):
        lock_type = (
            "write"
            if "write" in error_message or "readonly" in error_message
            else "read"
        )
        return DatabaseLockError(
            original_message, operation_name=operation_name, lock_type=lock_type
        )

    # Database busy errors
    if any(
        msg in error_message
        for msg in [
            "database is busy",
            "sqlite_busy",
            "cannot start a transaction within a transaction",
        ]
    ):
        resource_type = (
            "transaction" if "transaction" in error_message else "connection"
        )
        return DatabaseBusyError(
            original_message, operation_name=operation_name, resource_type=resource_type
        )

    # Connection errors
    if any(
        msg in error_message
        for msg in [
            "unable to open database",
            "disk i/o error",
            "database disk image is malformed",
            "no such database",
        ]
    ):
        return DatabaseConnectionError(original_message, operation_name=operation_name)

    # Schema errors
    if any(
        msg in error_message
        for msg in [
            "no such table",
            "no such column",
            "table already exists",
            "syntax error",
        ]
    ):
        return DatabaseSchemaError(original_message, operation_name=operation_name)

    # Integrity constraint errors
    if any(
        msg in error_message
        for msg in [
            "unique constraint failed",
            "foreign key constraint failed",
            "primary key constraint failed",
            "check constraint failed",
        ]
    ):
        constraint_type = "unknown"
        if "unique" in error_message:
            constraint_type = "unique"
        elif "foreign key" in error_message:
            constraint_type = "foreign_key"
        elif "primary key" in error_message:
            constraint_type = "primary_key"
        elif "check" in error_message:
            constraint_type = "check"

        return DatabaseIntegrityError(
            original_message,
            operation_name=operation_name,
            constraint_type=constraint_type,
        )

    # Default to generic database error
    return DatabaseError(
        original_message,
        operation_name=operation_name,
        error_context={"original_error_type": type(error).__name__},
    )


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable based on our classification.

    Args:
        error: Exception to check

    Returns:
        True if the error should trigger a retry
    """
    if isinstance(error, DatabaseError):
        return error.error_context.get("retryable", False)

    # For raw exceptions, use simple classification
    error_message = str(error).lower()
    retryable_patterns = [
        "database is locked",
        "database is busy",
        "sqlite_busy",
        "sqlite_locked",
        "cannot start a transaction within a transaction",
    ]

    return any(pattern in error_message for pattern in retryable_patterns)


def get_error_classification_stats(errors: list) -> Dict[str, Any]:
    """
    Analyze a list of errors and provide classification statistics.

    Args:
        errors: List of Exception objects to analyze

    Returns:
        Dictionary with error classification statistics
    """
    stats = {
        "total_errors": len(errors),
        "error_types": {},
        "retryable_count": 0,
        "non_retryable_count": 0,
        "most_common_errors": {},
    }

    error_messages = {}

    for error in errors:
        # Classify error
        classified = (
            classify_sqlite_error(error)
            if not isinstance(error, DatabaseError)
            else error
        )
        error_type = type(classified).__name__

        # Count by type
        stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1

        # Count retryable vs non-retryable
        if is_retryable_error(classified):
            stats["retryable_count"] += 1
        else:
            stats["non_retryable_count"] += 1

        # Track common error messages
        message = str(error)
        error_messages[message] = error_messages.get(message, 0) + 1

    # Find most common error messages
    stats["most_common_errors"] = sorted(
        error_messages.items(), key=lambda x: x[1], reverse=True
    )[:5]

    return stats
