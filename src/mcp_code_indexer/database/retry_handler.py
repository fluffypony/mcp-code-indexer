"""
Database retry handling for SQLite locking scenarios.

This module provides specialized retry logic for database operations that may
encounter locking issues in high-concurrency environments.
"""

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional, Type, Union
from datetime import datetime

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for database retry logic."""
    max_attempts: int = 5
    initial_delay: float = 0.1  # seconds
    max_delay: float = 2.0      # seconds
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on_error_types: tuple = (aiosqlite.OperationalError,)


class DatabaseLockError(Exception):
    """Exception for database locking issues."""
    
    def __init__(self, message: str, retry_count: int = 0, last_attempt: Optional[datetime] = None):
        self.message = message
        self.retry_count = retry_count
        self.last_attempt = last_attempt or datetime.utcnow()
        super().__init__(message)


class RetryHandler:
    """Handles database operation retries with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.
        
        Args:
            config: Retry configuration, uses defaults if None
        """
        self.config = config or RetryConfig()
        self._retry_stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_operations": 0,
            "avg_retry_delay": 0.0
        }
    
    @asynccontextmanager
    async def with_retry(self, operation_name: str = "database_operation") -> AsyncIterator[None]:
        """
        Context manager that provides retry logic for database operations.
        
        Args:
            operation_name: Name of the operation for logging
            
        Usage:
            async with retry_handler.with_retry("create_project"):
                # Your database operation here
                await db.execute(...)
        """
        last_error = None
        total_delay = 0.0
        
        for attempt in range(1, self.config.max_attempts + 1):
            self._retry_stats["total_attempts"] += 1
            
            try:
                yield
                
                # Success - log if this was a retry
                if attempt > 1:
                    self._retry_stats["successful_retries"] += 1
                    logger.info(
                        f"Database operation '{operation_name}' succeeded on attempt {attempt}",
                        extra={
                            "structured_data": {
                                "retry_success": {
                                    "operation": operation_name,
                                    "attempt": attempt,
                                    "total_delay": total_delay
                                }
                            }
                        }
                    )
                return
                
            except GeneratorExit:
                # Handle cleanup properly when context manager is being torn down
                logger.debug(f"Retry context manager for '{operation_name}' being cleaned up")
                raise
                
            except Exception as e:
                last_error = e
                
                # Check if this is a retryable error
                if not self._is_retryable_error(e):
                    logger.error(
                        f"Non-retryable error in '{operation_name}': {e}",
                        extra={
                            "structured_data": {
                                "non_retryable_error": {
                                    "operation": operation_name,
                                    "error_type": type(e).__name__,
                                    "error_message": str(e)
                                }
                            }
                        }
                    )
                    raise
                
                # If this is the last attempt, give up
                if attempt >= self.config.max_attempts:
                    self._retry_stats["failed_operations"] += 1
                    logger.error(
                        f"Database operation '{operation_name}' failed after {attempt} attempts",
                        extra={
                            "structured_data": {
                                "retry_exhausted": {
                                    "operation": operation_name,
                                    "max_attempts": self.config.max_attempts,
                                    "total_delay": total_delay,
                                    "final_error": str(e)
                                }
                            }
                        }
                    )
                    raise DatabaseLockError(
                        f"Database operation failed after {attempt} attempts: {e}",
                        retry_count=attempt,
                        last_attempt=datetime.utcnow()
                    )
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                total_delay += delay
                
                logger.warning(
                    f"Database operation '{operation_name}' failed on attempt {attempt}, retrying in {delay:.2f}s",
                    extra={
                        "structured_data": {
                            "retry_attempt": {
                                "operation": operation_name,
                                "attempt": attempt,
                                "delay_seconds": delay,
                                "error_type": type(e).__name__,
                                "error_message": str(e)
                            }
                        }
                    }
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Args:
            error: Exception to check
            
        Returns:
            True if the error should trigger a retry
        """
        # Check error type
        if not isinstance(error, self.config.retry_on_error_types):
            return False
        
        # Check specific SQLite error messages
        error_message = str(error).lower()
        retryable_messages = [
            "database is locked",
            "database is busy",
            "cannot start a transaction within a transaction",
            "sqlite_busy",
            "sqlite_locked"
        ]
        
        return any(msg in error_message for msg in retryable_messages)
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt with exponential backoff and jitter.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: initial_delay * (multiplier ^ (attempt - 1))
        delay = self.config.initial_delay * (self.config.backoff_multiplier ** (attempt - 1))
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        # Ensure delay is positive
        return max(0.0, delay)
    
    def get_retry_stats(self) -> dict:
        """
        Get retry statistics.
        
        Returns:
            Dictionary with retry statistics
        """
        if self._retry_stats["successful_retries"] > 0:
            self._retry_stats["avg_retry_delay"] = (
                self._retry_stats["total_attempts"] / self._retry_stats["successful_retries"]
            )
        
        return self._retry_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset retry statistics."""
        self._retry_stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_operations": 0,
            "avg_retry_delay": 0.0
        }


class ConnectionRecoveryManager:
    """Manages database connection recovery for persistent failures."""
    
    def __init__(self, database_manager):
        """
        Initialize connection recovery manager.
        
        Args:
            database_manager: DatabaseManager instance to manage
        """
        self.database_manager = database_manager
        self._recovery_stats = {
            "pool_refreshes": 0,
            "last_refresh": None,
            "consecutive_failures": 0
        }
        self._failure_threshold = 3  # Refresh pool after 3 consecutive failures
    
    async def handle_persistent_failure(self, operation_name: str, error: Exception) -> bool:
        """
        Handle persistent database failures by attempting pool refresh.
        
        Args:
            operation_name: Name of the failing operation
            error: The persistent error
            
        Returns:
            True if pool refresh was attempted, False otherwise
        """
        self._recovery_stats["consecutive_failures"] += 1
        
        # Only refresh if we've hit the threshold
        if self._recovery_stats["consecutive_failures"] >= self._failure_threshold:
            logger.warning(
                f"Attempting connection pool refresh after {self._recovery_stats['consecutive_failures']} failures",
                extra={
                    "structured_data": {
                        "pool_recovery": {
                            "operation": operation_name,
                            "consecutive_failures": self._recovery_stats["consecutive_failures"],
                            "trigger_error": str(error)
                        }
                    }
                }
            )
            
            await self._refresh_connection_pool()
            return True
        
        return False
    
    def reset_failure_count(self) -> None:
        """Reset consecutive failure count after successful operation."""
        self._recovery_stats["consecutive_failures"] = 0
    
    async def _refresh_connection_pool(self) -> None:
        """
        Refresh the database connection pool by closing all connections.
        
        This forces creation of new connections on next access.
        """
        try:
            # Close existing pool
            await self.database_manager.close_pool()
            
            # Update stats
            self._recovery_stats["pool_refreshes"] += 1
            self._recovery_stats["last_refresh"] = datetime.utcnow()
            self._recovery_stats["consecutive_failures"] = 0
            
            logger.info("Database connection pool refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh connection pool: {e}")
            raise
    
    def get_recovery_stats(self) -> dict:
        """
        Get connection recovery statistics.
        
        Returns:
            Dictionary with recovery statistics
        """
        return self._recovery_stats.copy()


def create_retry_handler(
    max_attempts: int = 5,
    initial_delay: float = 0.1,
    max_delay: float = 2.0
) -> RetryHandler:
    """
    Create a configured retry handler for database operations.
    
    Args:
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Configured RetryHandler instance
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay
    )
    return RetryHandler(config)
