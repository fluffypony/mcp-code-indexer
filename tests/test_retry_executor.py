"""
Tests for the retry executor with tenacity-based retry logic.

This module tests the RetryExecutor class that replaces the broken async
context manager retry pattern with proper separation of concerns.
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import aiosqlite
import pytest

from src.mcp_code_indexer.database.retry_executor import (
    DatabaseLockError,
    RetryConfig,
    RetryExecutor,
    RetryStats,
    create_retry_executor,
)


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_config(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 5
        assert config.min_wait_seconds == 0.1
        assert config.max_wait_seconds == 2.0
        assert config.jitter_max_seconds == 0.2
        assert config.retry_on_errors == (aiosqlite.OperationalError,)

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=3,
            min_wait_seconds=0.05,
            max_wait_seconds=1.0,
            jitter_max_seconds=0.1,
        )
        assert config.max_attempts == 3
        assert config.min_wait_seconds == 0.05
        assert config.max_wait_seconds == 1.0
        assert config.jitter_max_seconds == 0.1


class TestRetryStats:
    """Test retry statistics tracking."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = RetryStats()
        assert stats.total_operations == 0
        assert stats.successful_operations == 0
        assert stats.retried_operations == 0
        assert stats.failed_operations == 0
        assert stats.total_attempts == 0
        assert stats.total_retry_time == 0.0
        assert stats.last_operation_time is None

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = RetryStats(total_operations=10, successful_operations=8)
        assert stats.success_rate == 80.0

        # Test with zero operations
        empty_stats = RetryStats()
        assert empty_stats.success_rate == 0.0

    def test_retry_rate_calculation(self):
        """Test retry rate calculation."""
        stats = RetryStats(total_operations=10, retried_operations=3)
        assert stats.retry_rate == 30.0

    def test_average_attempts_calculation(self):
        """Test average attempts per operation calculation."""
        stats = RetryStats(total_operations=5, total_attempts=12)
        assert stats.average_attempts_per_operation == 2.4


class TestDatabaseLockError:
    """Test custom database lock error."""

    def test_error_creation(self):
        """Test error creation with context."""
        error = DatabaseLockError(
            "Database locked", retry_count=3, operation_name="test_operation"
        )

        assert "test_operation" in str(error)
        assert "Database locked" in str(error)
        assert "3 attempts" in str(error)
        assert error.retry_count == 3
        assert error.operation_name == "test_operation"
        assert isinstance(error.last_attempt, datetime)


@pytest.mark.asyncio
class TestRetryExecutor:
    """Test the retry executor functionality."""

    @pytest.fixture
    def retry_executor(self):
        """Create a retry executor for testing."""
        config = RetryConfig(
            max_attempts=3,
            min_wait_seconds=0.01,  # Very short delays for testing
            max_wait_seconds=0.05,
            jitter_max_seconds=0.001,
        )
        return RetryExecutor(config)

    async def test_successful_operation(self, retry_executor):
        """Test successful operation without retries."""
        call_count = 0

        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_executor.execute_with_retry(
            successful_operation, "test_operation"
        )

        assert result == "success"
        assert call_count == 1

        stats = retry_executor.get_retry_stats()
        assert stats["total_operations"] == 1
        assert stats["successful_operations"] == 1
        assert stats["retried_operations"] == 0
        assert stats["failed_operations"] == 0
        assert stats["total_attempts"] == 1

    async def test_retry_until_success(self, retry_executor):
        """Test operation that fails then succeeds."""
        call_count = 0

        async def retry_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise aiosqlite.OperationalError("database is locked")
            return "success"

        result = await retry_executor.execute_with_retry(
            retry_then_succeed, "test_retry_operation"
        )

        assert result == "success"
        assert call_count == 3

        stats = retry_executor.get_retry_stats()
        assert stats["total_operations"] == 1
        assert stats["successful_operations"] == 1
        assert stats["retried_operations"] == 1
        assert stats["total_attempts"] == 3

    async def test_max_retries_exceeded(self, retry_executor):
        """Test operation that exceeds max retry attempts."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise aiosqlite.OperationalError("database is locked")

        with pytest.raises(DatabaseLockError) as exc_info:
            await retry_executor.execute_with_retry(always_fail, "test_fail_operation")

        assert "test_fail_operation" in str(exc_info.value)
        assert call_count == 3  # max_attempts

        stats = retry_executor.get_retry_stats()
        assert stats["total_operations"] == 1
        assert stats["successful_operations"] == 0
        assert stats["failed_operations"] == 1
        assert stats["total_attempts"] == 3

    async def test_non_retryable_error(self, retry_executor):
        """Test that non-retryable errors are not retried."""
        call_count = 0

        async def non_retryable_error():
            nonlocal call_count
            call_count += 1
            raise aiosqlite.OperationalError("syntax error")

        with pytest.raises(aiosqlite.OperationalError):
            await retry_executor.execute_with_retry(
                non_retryable_error, "test_non_retryable"
            )

        assert call_count == 1  # Should not retry

        stats = retry_executor.get_retry_stats()
        assert stats["failed_operations"] == 1
        assert stats["total_attempts"] == 1

    async def test_concurrent_operations(self, retry_executor):
        """Test concurrent operations with retry logic."""
        operation_count = 0

        async def concurrent_operation(operation_id):
            nonlocal operation_count
            operation_count += 1

            # Simulate some operations failing randomly
            if operation_id % 3 == 0:
                raise aiosqlite.OperationalError("database is locked")

            await asyncio.sleep(0.001)  # Simulate work
            return f"result_{operation_id}"

        # Run multiple operations concurrently
        tasks = [
            retry_executor.execute_with_retry(
                lambda oid=i: concurrent_operation(oid), f"concurrent_op_{i}"
            )
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that most operations succeeded
        successful_results = [r for r in results if isinstance(r, str)]
        _ = [
            r for r in results if isinstance(r, Exception)
        ]  # failed_results for potential future use

        assert len(successful_results) >= 6  # Most should succeed
        assert all("result_" in result for result in successful_results)

        stats = retry_executor.get_retry_stats()
        assert stats["total_operations"] == 10

    async def test_stats_reset(self, retry_executor):
        """Test statistics reset functionality."""

        # Run some operations to build up stats
        async def simple_op():
            return "result"

        await retry_executor.execute_with_retry(simple_op, "op1")
        await retry_executor.execute_with_retry(simple_op, "op2")

        stats_before = retry_executor.get_retry_stats()
        assert stats_before["total_operations"] == 2

        # Reset stats
        retry_executor.reset_stats()

        stats_after = retry_executor.get_retry_stats()
        assert stats_after["total_operations"] == 0
        assert stats_after["successful_operations"] == 0
        assert stats_after["total_attempts"] == 0


@pytest.mark.asyncio
class TestRetryExecutorWithRealDatabase:
    """Test retry executor with real database operations."""



    async def test_database_operations_with_retries(self, temp_db_with_test_table):
        """Test actual database operations with retry logic."""
        retry_executor = create_retry_executor(max_attempts=5)

        async def insert_data(data_value):
            async with aiosqlite.connect(temp_db_with_test_table) as db:
                await db.execute(
                    "INSERT INTO test_table (data) VALUES (?)", (data_value,)
                )
                await db.commit()
                return f"inserted_{data_value}"

        # Test successful database operation
        result = await retry_executor.execute_with_retry(
            lambda: insert_data("test_data"), "database_insert"
        )

        assert result == "inserted_test_data"

        # Verify data was inserted
        async with aiosqlite.connect(temp_db_with_test_table) as db:
            cursor = await db.execute(
                "SELECT data FROM test_table WHERE data = ?", ("test_data",)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "test_data"

    async def test_simulated_lock_contention(self, temp_db_with_test_table):
        """Test retry logic under simulated lock contention."""
        retry_executor = create_retry_executor(max_attempts=3, min_wait_seconds=0.001)

        # Create a long-running transaction to cause lock contention
        blocking_conn = await aiosqlite.connect(temp_db_with_test_table)
        await blocking_conn.execute("BEGIN EXCLUSIVE")

        attempt_count = 0

        async def contended_operation():
            nonlocal attempt_count
            attempt_count += 1

            # This should fail while the exclusive transaction is active
            async with aiosqlite.connect(temp_db_with_test_table) as db:
                await db.execute(
                    "INSERT INTO test_table (data) VALUES (?)", ("contended",)
                )
                await db.commit()

        # Start the contended operation (should fail initially)
        contention_task = asyncio.create_task(
            retry_executor.execute_with_retry(contended_operation, "contended_insert")
        )

        # Let it fail a couple times
        await asyncio.sleep(0.01)

        # Release the lock
        await blocking_conn.rollback()
        await blocking_conn.close()

        # Now the operation should succeed
        try:
            await asyncio.wait_for(contention_task, timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail("Operation should have succeeded after lock release")

        # Verify operation completed (retries may or may not have occurred due to timing)
        stats = retry_executor.get_retry_stats()
        # We can't guarantee retries occurred (timing dependent), but can verify stats exist
        assert "retried_operations" in stats
        assert "total_attempts" in stats


class TestCreateRetryExecutor:
    """Test the retry executor factory function."""

    def test_create_with_defaults(self):
        """Test creating retry executor with default values."""
        executor = create_retry_executor()

        assert executor.config.max_attempts == 5
        assert executor.config.min_wait_seconds == 0.1
        assert executor.config.max_wait_seconds == 2.0
        assert executor.config.jitter_max_seconds == 0.2

    def test_create_with_custom_values(self):
        """Test creating retry executor with custom values."""
        executor = create_retry_executor(
            max_attempts=10,
            min_wait_seconds=0.05,
            max_wait_seconds=5.0,
            jitter_max_seconds=0.5,
        )

        assert executor.config.max_attempts == 10
        assert executor.config.min_wait_seconds == 0.05
        assert executor.config.max_wait_seconds == 5.0
        assert executor.config.jitter_max_seconds == 0.5
