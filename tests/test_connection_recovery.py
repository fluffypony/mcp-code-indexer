"""
Tests for database connection recovery and resilience scenarios.

This module tests connection recovery mechanisms, pool refresh functionality,
and edge cases in database connection management.
"""

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path

import aiosqlite
import pytest

from src.mcp_code_indexer.database.connection_health import (
    DatabaseMetricsCollector,
)
from src.mcp_code_indexer.database.database import DatabaseManager

# ConnectionRecoveryManager removed - recovery is now handled by RetryExecutor


class TestConnectionRecovery:
    """Test connection recovery and resilience mechanisms."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [3], indirect=True)
    async def test_connection_pool_refresh(self, temp_db_manager_pool: DatabaseManager):
        """Test connection pool refresh functionality."""
        # Get initial pool state
        initial_stats = temp_db_manager_pool.get_database_stats()
        # Note: initial_pool_size available if needed for future checks
        _ = initial_stats["connection_pool"]["current_size"]

        # Manually trigger pool refresh
        await temp_db_manager_pool.close_pool()

        # Pool should be empty after close
        post_close_stats = temp_db_manager_pool.get_database_stats()
        assert post_close_stats["connection_pool"]["current_size"] == 0

        # Creating a new connection should work after refresh
        async with temp_db_manager_pool.get_connection() as conn:
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result[0] == 1

    @pytest.mark.asyncio
    async def test_metrics_collector_operation_recording(self):
        """Test database metrics collection for operations."""
        collector = DatabaseMetricsCollector()

        # Record some operations
        collector.record_operation("create_project", 100.5, True, 3)
        collector.record_operation("create_project", 150.2, True, 3)
        collector.record_operation("create_project", 200.0, False, 3)
        collector.record_operation("batch_insert", 500.0, True, 3)

        # Get metrics
        metrics = collector.get_operation_metrics()

        # Verify create_project metrics
        create_metrics = metrics["create_project"]
        assert create_metrics["total_operations"] == 3
        assert create_metrics["successful_operations"] == 2
        assert create_metrics["failed_operations"] == 1
        assert create_metrics["min_duration_ms"] == 100.5
        assert create_metrics["max_duration_ms"] == 200.0
        assert abs(create_metrics["avg_duration_ms"] - 150.23) < 0.1

        # Verify batch_insert metrics
        batch_metrics = metrics["batch_insert"]
        assert batch_metrics["total_operations"] == 1
        assert batch_metrics["successful_operations"] == 1
        assert batch_metrics["avg_duration_ms"] == 500.0

    @pytest.mark.asyncio
    async def test_metrics_collector_locking_events(self):
        """Test locking event recording and frequency analysis."""
        collector = DatabaseMetricsCollector()

        # Record some locking events
        collector.record_locking_event("create_project", "database is locked")
        collector.record_locking_event("create_project", "database is locked")
        collector.record_locking_event("batch_insert", "database is busy")

        # Get locking frequency
        locking_stats = collector.get_locking_frequency()

        assert locking_stats["total_events"] == 3
        assert len(locking_stats["most_frequent_operations"]) >= 1

        # Check most frequent operation
        most_frequent = locking_stats["most_frequent_operations"][0]
        assert most_frequent["operation"] == "create_project"
        assert most_frequent["count"] == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [3], indirect=True)
    async def test_connection_exhaustion_recovery(
        self, temp_db_manager_pool: DatabaseManager
    ):
        """Test recovery from connection pool exhaustion."""
        # Create more connections than pool size to test exhaustion
        connections = []

        try:
            # Try to get more connections than pool allows
            for i in range(temp_db_manager_pool.pool_size + 2):
                conn_ctx = temp_db_manager_pool.get_connection()
                conn = await conn_ctx.__aenter__()
                connections.append((conn_ctx, conn))

            # Pool should handle this gracefully by creating new connections
            assert len(connections) == temp_db_manager_pool.pool_size + 2

            # All connections should work
            for conn_ctx, conn in connections:
                cursor = await conn.execute("SELECT 1")
                result = await cursor.fetchone()
                assert result[0] == 1

        finally:
            # Clean up all connections
            for conn_ctx, conn in connections:
                await conn_ctx.__aexit__(None, None, None)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_database_corruption_handling(
        self, temp_db_manager_pool: DatabaseManager, temp_db
    ):
        """Test handling of database corruption scenarios."""
        # Close the manager first to corrupt the database file
        await temp_db_manager_pool.close_pool()

        # Corrupt the database file and remove WAL/SHM files
        # (SQLite WAL mode stores data in -wal file that can recover the DB)
        with open(temp_db, "wb") as f:
            f.write(b"corrupted data")

        # Remove WAL and SHM files to ensure corruption is complete
        wal_path = Path(str(temp_db) + "-wal")
        shm_path = Path(str(temp_db) + "-shm")
        if wal_path.exists():
            wal_path.unlink()
        if shm_path.exists():
            shm_path.unlink()

        # Try to reinitialize
        new_db_manager = DatabaseManager(temp_db, pool_size=2)

        # This should handle corruption gracefully
        with pytest.raises((aiosqlite.DatabaseError, sqlite3.DatabaseError)):
            await new_db_manager.initialize()

        await new_db_manager.close_pool()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [3], indirect=True)
    async def test_concurrent_pool_refresh(self, temp_db_manager_pool: DatabaseManager):
        """Test concurrent pool refresh operations."""
        # Start multiple pool refresh operations concurrently
        refresh_tasks = [
            temp_db_manager_pool.close_pool(),
            temp_db_manager_pool.close_pool(),
            temp_db_manager_pool.close_pool(),
        ]

        # This should not cause errors
        await asyncio.gather(*refresh_tasks, return_exceptions=True)

        # Database should still be functional
        async with temp_db_manager_pool.get_connection() as conn:
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result[0] == 1


if __name__ == "__main__":
    # Run tests with asyncio
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
