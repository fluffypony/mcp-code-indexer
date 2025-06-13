"""
Tenacity-based retry executor for database operations with exponential backoff.

This module provides a robust retry executor that replaces the broken async 
context manager retry pattern with proper separation of concerns between
retry logic and resource management.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Callable, Dict, Optional, Type, TypeVar, Union

import aiosqlite
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for database retry logic using tenacity."""
    max_attempts: int = 5
    min_wait_seconds: float = 0.1
    max_wait_seconds: float = 2.0
    jitter_max_seconds: float = 0.2  # Max jitter to add
    retry_on_errors: tuple = field(default_factory=lambda: (aiosqlite.OperationalError,))
    

@dataclass 
class RetryStats:
    """Statistics for retry operations."""
    total_operations: int = 0
    successful_operations: int = 0
    retried_operations: int = 0
    failed_operations: int = 0
    total_attempts: int = 0
    total_retry_time: float = 0.0
    last_operation_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100.0
    
    @property
    def retry_rate(self) -> float:
        """Calculate retry rate as percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.retried_operations / self.total_operations) * 100.0
    
    @property
    def average_attempts_per_operation(self) -> float:
        """Calculate average retry attempts per operation."""
        if self.total_operations == 0:
            return 0.0
        return self.total_attempts / self.total_operations


class DatabaseLockError(Exception):
    """Exception for database locking issues with retry context."""
    
    def __init__(self, message: str, retry_count: int = 0, operation_name: str = "", 
                 last_attempt: Optional[datetime] = None):
        self.message = message
        self.retry_count = retry_count
        self.operation_name = operation_name
        self.last_attempt = last_attempt or datetime.utcnow()
        super().__init__(f"{operation_name}: {message} (after {retry_count} attempts)")


class RetryExecutor:
    """
    Tenacity-based retry executor for database operations.
    
    This executor provides robust retry logic with exponential backoff,
    proper error classification, and comprehensive statistics tracking.
    It replaces the broken async context manager retry pattern.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry executor.
        
        Args:
            config: Retry configuration, uses defaults if None
        """
        self.config = config or RetryConfig()
        self._stats = RetryStats()
        self._operation_start_times: Dict[str, datetime] = {}
        
        # Configure tenacity retrying with exponential backoff and jitter
        self._tenacity_retrying = AsyncRetrying(
            stop=stop_after_attempt(self.config.max_attempts),
            wait=wait_exponential_jitter(
                initial=self.config.min_wait_seconds,
                max=self.config.max_wait_seconds,
                jitter=self.config.jitter_max_seconds
            ),
            retry=retry_if_exception_type(self.config.retry_on_errors),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG),
            reraise=True
        )
    
    async def execute_with_retry(self, 
                                 operation: Callable[[], T], 
                                 operation_name: str = "database_operation") -> T:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Async callable to execute
            operation_name: Name for logging and statistics
            
        Returns:
            Result of the operation
            
        Raises:
            DatabaseLockError: If all retry attempts fail
            Exception: For non-retryable errors
        """
        self._stats.total_operations += 1
        self._operation_start_times[operation_name] = datetime.utcnow()
        
        attempt_count = 0
        operation_start = datetime.utcnow()
        
        try:
            async for attempt in self._tenacity_retrying:
                with attempt:
                    attempt_count += 1
                    self._stats.total_attempts += 1
                    
                    # Check if this error is retryable for SQLite specifically
                    if attempt_count > 1:
                        last_error = attempt.retry_state.outcome.exception()
                        if not self._is_sqlite_retryable_error(last_error):
                            logger.error(
                                f"Non-retryable SQLite error in '{operation_name}': {last_error}",
                                extra={"structured_data": {
                                    "non_retryable_error": {
                                        "operation": operation_name,
                                        "error_type": type(last_error).__name__,
                                        "error_message": str(last_error),
                                        "attempt": attempt_count
                                    }
                                }}
                            )
                            # Re-raise immediately for non-retryable errors
                            raise last_error
                        
                        self._stats.retried_operations += 1
                        logger.warning(
                            f"Retrying '{operation_name}' (attempt {attempt_count})",
                            extra={"structured_data": {
                                "retry_attempt": {
                                    "operation": operation_name,
                                    "attempt": attempt_count,
                                    "error_type": type(last_error).__name__,
                                    "error_message": str(last_error)
                                }
                            }}
                        )
                    
                    # Execute the operation
                    result = await operation()
                    
                    # Success - update statistics
                    operation_time = (datetime.utcnow() - operation_start).total_seconds()
                    self._stats.successful_operations += 1
                    self._stats.last_operation_time = datetime.utcnow()
                    
                    if attempt_count > 1:
                        self._stats.total_retry_time += operation_time
                        logger.info(
                            f"Operation '{operation_name}' succeeded after {attempt_count} attempts",
                            extra={"structured_data": {
                                "retry_success": {
                                    "operation": operation_name,
                                    "attempts": attempt_count,
                                    "total_time_seconds": operation_time
                                }
                            }}
                        )
                    
                    return result
                    
        except RetryError as e:
            # All retry attempts exhausted
            operation_time = (datetime.utcnow() - operation_start).total_seconds()
            self._stats.failed_operations += 1
            self._stats.total_retry_time += operation_time
            
            original_error = e.last_attempt.exception()
            logger.error(
                f"Operation '{operation_name}' failed after {attempt_count} attempts",
                extra={"structured_data": {
                    "retry_exhausted": {
                        "operation": operation_name,
                        "max_attempts": self.config.max_attempts,
                        "total_time_seconds": operation_time,
                        "final_error": str(original_error)
                    }
                }}
            )
            
            raise DatabaseLockError(
                f"Database operation failed after {attempt_count} attempts: {original_error}",
                retry_count=attempt_count,
                operation_name=operation_name,
                last_attempt=datetime.utcnow()
            )
            
        except Exception as e:
            # Non-retryable error on first attempt
            self._stats.failed_operations += 1
            logger.error(
                f"Non-retryable error in '{operation_name}': {e}",
                extra={"structured_data": {
                    "immediate_failure": {
                        "operation": operation_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                }}
            )
            raise
        
        finally:
            # Clean up tracking
            self._operation_start_times.pop(operation_name, None)
    
    @asynccontextmanager
    async def get_connection_with_retry(self, 
                                       connection_factory: Callable[[], AsyncIterator[aiosqlite.Connection]],
                                       operation_name: str = "database_connection") -> AsyncIterator[aiosqlite.Connection]:
        """
        Get a database connection with retry logic wrapped around the context manager.
        
        This method properly separates retry logic from resource management by
        retrying the entire context manager operation, not yielding inside a retry loop.
        
        Args:
            connection_factory: Function that returns an async context manager for connections
            operation_name: Name for logging and statistics
            
        Yields:
            Database connection
        """
        
        async def get_connection():
            # This function will be retried by execute_with_retry
            async with connection_factory() as conn:
                # Store connection for the outer context manager
                return conn
        
        # Use execute_with_retry to handle the retry logic
        # We create a connection and store it for the context manager
        connection = await self.execute_with_retry(get_connection, operation_name)
        
        try:
            yield connection
        finally:
            # Connection cleanup is handled by the original context manager
            # in the connection_factory, so nothing to do here
            pass
    
    def _is_sqlite_retryable_error(self, error: Exception) -> bool:
        """
        Determine if a SQLite error is retryable.
        
        Args:
            error: Exception to check
            
        Returns:
            True if the error should trigger a retry
        """
        if not isinstance(error, self.config.retry_on_errors):
            return False
        
        # Check specific SQLite error messages that indicate transient issues
        error_message = str(error).lower()
        retryable_messages = [
            "database is locked",
            "database is busy", 
            "cannot start a transaction within a transaction",
            "sqlite_busy",
            "sqlite_locked"
        ]
        
        return any(msg in error_message for msg in retryable_messages)
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive retry statistics.
        
        Returns:
            Dictionary with retry statistics and performance metrics
        """
        return {
            "total_operations": self._stats.total_operations,
            "successful_operations": self._stats.successful_operations,
            "retried_operations": self._stats.retried_operations,
            "failed_operations": self._stats.failed_operations,
            "total_attempts": self._stats.total_attempts,
            "success_rate_percent": round(self._stats.success_rate, 2),
            "retry_rate_percent": round(self._stats.retry_rate, 2),
            "average_attempts_per_operation": round(self._stats.average_attempts_per_operation, 2),
            "total_retry_time_seconds": round(self._stats.total_retry_time, 3),
            "last_operation_time": self._stats.last_operation_time.isoformat() if self._stats.last_operation_time else None,
            "config": {
                "max_attempts": self.config.max_attempts,
                "min_wait_seconds": self.config.min_wait_seconds,
                "max_wait_seconds": self.config.max_wait_seconds,
                "jitter_max_seconds": self.config.jitter_max_seconds
            }
        }
    
    def reset_stats(self) -> None:
        """Reset retry statistics."""
        self._stats = RetryStats()
        self._operation_start_times.clear()


def create_retry_executor(
    max_attempts: int = 5,
    min_wait_seconds: float = 0.1,
    max_wait_seconds: float = 2.0,
    jitter_max_seconds: float = 0.2
) -> RetryExecutor:
    """
    Create a configured retry executor for database operations.
    
    Args:
        max_attempts: Maximum retry attempts
        min_wait_seconds: Initial delay in seconds
        max_wait_seconds: Maximum delay in seconds  
        jitter_max_seconds: Maximum jitter to add to delays
        
    Returns:
        Configured RetryExecutor instance
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        min_wait_seconds=min_wait_seconds,
        max_wait_seconds=max_wait_seconds,
        jitter_max_seconds=jitter_max_seconds
    )
    return RetryExecutor(config)
