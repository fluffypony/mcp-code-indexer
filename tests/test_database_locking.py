"""
Tests for database locking resilience and concurrent access scenarios.

This module tests the enhanced database locking mechanisms including WAL mode,
write serialization, retry logic, and connection health monitoring.
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import aiosqlite
import pytest

from src.mcp_code_indexer.database.database import DatabaseManager
from src.mcp_code_indexer.database.models import FileDescription, Project


class TestDatabaseLocking:
    """Test database locking resilience and concurrent access."""



    @pytest.fixture
    def sample_project(self):
        """Create a sample project for testing."""
        return Project(
            id="test-project-123",
            name="test-project",
            aliases=["/test/path"],
            created=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
        )

    @pytest.fixture
    def sample_file_description(self):
        """Create a sample file description for testing."""
        return FileDescription(
            project_id="test-project-123",
            file_path="src/test.py",
            description="Test file description",
            file_hash="abc123",
            last_modified=datetime.utcnow(),
            version=1,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_wal_mode_enabled(self, temp_db_manager_pool):
        """Test that WAL mode is enabled correctly."""
        async with temp_db_manager_pool.get_connection() as conn:
            cursor = await conn.execute("PRAGMA journal_mode")
            result = await cursor.fetchone()
            assert result[0].upper() == "WAL"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_concurrent_writes_succeed(self, temp_db_manager_pool, sample_project):
        """Test that concurrent write operations complete successfully with WAL mode."""
        completed_operations = []

        async def tracked_write_operation(project_suffix: str):
            """Write operation that tracks completion."""
            # Create project with slight delay to test concurrency
            project = Project(
                id=f"test-project-{project_suffix}",
                name=f"test-project-{project_suffix}",
                aliases=[f"/test/path/{project_suffix}"],
                created=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
            )

            await asyncio.sleep(0.01)  # Simulate work
            await temp_db_manager_pool.create_project(project)
            completed_operations.append(project_suffix)

        # Run multiple write operations concurrently
        tasks = [
            tracked_write_operation("1"),
            tracked_write_operation("2"),
            tracked_write_operation("3"),
        ]

        await asyncio.gather(*tasks)

        # Verify all operations completed successfully (no locking errors)
        assert len(completed_operations) == 3
        assert set(completed_operations) == {"1", "2", "3"}

        # Verify all projects were created
        all_projects = await temp_db_manager_pool.get_all_projects()
        project_ids = [p.id for p in all_projects]
        assert "test-project-1" in project_ids
        assert "test-project-2" in project_ids
        assert "test-project-3" in project_ids

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_immediate_transaction_basic(self, temp_db_manager_pool):
        """Test basic immediate transaction works correctly."""
        # Test that get_immediate_transaction context manager works
        async with temp_db_manager_pool.get_immediate_transaction(
            "test_basic", timeout_seconds=5.0
        ) as conn:
            # Perform a simple operation
            await conn.execute("SELECT 1")
        # If we get here without exception, the test passed

    # Retry handler tests removed - comprehensive retry testing is now in
    # test_retry_executor.py
    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_concurrent_reads_during_write(self, temp_db_manager_pool, sample_project):
        """Test that reads can proceed during writes (WAL mode benefit)."""
        # First, create a project to read
        await temp_db_manager_pool.create_project(sample_project)

        read_results = []
        write_started = asyncio.Event()
        write_can_continue = asyncio.Event()

        async def slow_write_operation():
            """Write operation that signals when it starts and waits."""
            write_started.set()
            await write_can_continue.wait()

            # Create another project
            new_project = Project(
                id="test-project-456",
                name="test-project-2",
                aliases=["/test/path2"],
                created=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
            )
            await temp_db_manager_pool.create_project(new_project)

        async def read_operation():
            """Read operation that should not be blocked by write."""
            await write_started.wait()  # Wait for write to start

            # This read should not be blocked by the ongoing write
            project = await temp_db_manager_pool.get_project(sample_project.id)
            read_results.append(project.name if project else None)

        # Start write operation
        write_task = asyncio.create_task(slow_write_operation())

        # Start read operation after write begins
        read_task = asyncio.create_task(read_operation())

        # Let the read complete
        await asyncio.sleep(0.1)

        # Allow write to continue
        write_can_continue.set()

        # Wait for both operations to complete
        await asyncio.gather(write_task, read_task)

        # Verify read was successful
        assert len(read_results) == 1
        assert read_results[0] == sample_project.name

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_batch_operations_performance(self, temp_db_manager_pool, sample_project):
        """Test that batch operations are more efficient than individual operations."""
        await temp_db_manager_pool.create_project(sample_project)

        # Create file descriptions for batch testing (no branch field - branch-independent)
        file_descriptions = [
            FileDescription(
                project_id=sample_project.id,
                file_path=f"src/test_{i}.py",
                description=f"Test file {i} description",
                file_hash=f"hash_{i}",
                last_modified=datetime.utcnow(),
                version=1,
            )
            for i in range(100)
        ]

        # Test batch creation
        start_time = asyncio.get_event_loop().time()
        await temp_db_manager_pool.batch_create_file_descriptions(file_descriptions)
        batch_time = asyncio.get_event_loop().time() - start_time

        # Verify all files were created
        all_descriptions = await temp_db_manager_pool.get_all_file_descriptions(
            sample_project.id
        )
        assert len(all_descriptions) == 100

        # Batch operation should be reasonably fast (less than 5 seconds)
        assert batch_time < 5.0, (
            f"Batch operation took {batch_time:.2f}s, expected < 5.0s"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_connection_health_monitoring(self, temp_db_manager_pool):
        """Test connection health monitoring functionality."""
        # Get health status
        health_status = await temp_db_manager_pool.check_health()

        # Verify health check structure
        assert "health_check" in health_status
        assert "overall_status" in health_status
        assert "recent_history" in health_status

        # Verify health check result
        health_check = health_status["health_check"]
        assert "is_healthy" in health_check
        assert "response_time_ms" in health_check
        assert "timestamp" in health_check

        # Health should be True for a functioning database
        assert health_check["is_healthy"] is True
        assert health_check["response_time_ms"] > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_database_stats_collection(self, temp_db_manager_pool):
        """Test database statistics collection."""
        stats = temp_db_manager_pool.get_database_stats()

        # Verify stats structure
        assert "connection_pool" in stats
        assert "retry_executor" in stats
        assert "health_status" in stats

        # Verify connection pool info
        pool_info = stats["connection_pool"]
        assert "configured_size" in pool_info
        assert "current_size" in pool_info
        assert pool_info["configured_size"] == 2  # Test pool size

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_retry_executor_exists(self, temp_db_manager_pool):
        """Test that retry executor is properly initialized."""
        # Verify retry executor is available
        assert temp_db_manager_pool._retry_executor is not None
        
        # Verify we can get retry stats
        stats = temp_db_manager_pool._retry_executor.get_retry_stats()
        assert "total_attempts" in stats
        assert "failed_operations" in stats
        assert "retried_operations" in stats


class TestConcurrentAccess:
    """Test concurrent access scenarios."""



    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [5], indirect=True)
    async def test_high_concurrency_writes(self, temp_db_manager_pool):
        """Test database resilience under high concurrent write load."""

        async def create_project_batch(batch_id: int, batch_size: int = 10):
            """Create a batch of projects concurrently."""
            projects = [
                Project(
                    id=f"project-{batch_id}-{i}",
                    name=f"project-{batch_id}-{i}",
                    aliases=[f"/test/path/{batch_id}/{i}"],
                    created=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                )
                for i in range(batch_size)
            ]

            # Create projects with retry mechanism
            for project in projects:
                await temp_db_manager_pool.create_project(project)

            return len(projects)

        # Run multiple batches concurrently
        batch_tasks = [create_project_batch(batch_id, 5) for batch_id in range(10)]

        results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Verify all batches completed successfully
        assert all(isinstance(result, int) and result == 5 for result in results)

        # Verify total project count
        all_projects = await temp_db_manager_pool.get_all_projects()
        assert len(all_projects) == 50  # 10 batches * 5 projects each

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [5], indirect=True)
    async def test_mixed_read_write_operations(self, temp_db_manager_pool):
        """Test mixed read and write operations under concurrent load."""
        # Create initial projects
        initial_projects = [
            Project(
                id=f"initial-project-{i}",
                name=f"initial-project-{i}",
                aliases=[f"/initial/path/{i}"],
                created=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
            )
            for i in range(5)
        ]

        for project in initial_projects:
            await temp_db_manager_pool.create_project(project)

        read_results = []
        write_results = []

        async def read_operation(op_id: int):
            """Perform read operations."""
            for _ in range(10):
                projects = await temp_db_manager_pool.get_all_projects()
                read_results.append(len(projects))
                await asyncio.sleep(0.01)  # Small delay

        async def write_operation(op_id: int):
            """Perform write operations."""
            for i in range(5):
                project = Project(
                    id=f"concurrent-project-{op_id}-{i}",
                    name=f"concurrent-project-{op_id}-{i}",
                    aliases=[f"/concurrent/path/{op_id}/{i}"],
                    created=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                )
                await temp_db_manager_pool.create_project(project)
                write_results.append(f"written_{op_id}_{i}")
                await asyncio.sleep(0.02)  # Small delay

        # Run mixed operations concurrently
        tasks = [
            read_operation(1),
            read_operation(2),
            write_operation(1),
            write_operation(2),
            read_operation(3),
        ]

        await asyncio.gather(*tasks)

        # Verify operations completed
        assert len(read_results) >= 30  # 3 read operations * 10 reads each
        assert len(write_results) == 10  # 2 write operations * 5 writes each

        # Verify final state
        final_projects = await temp_db_manager_pool.get_all_projects()
        assert len(final_projects) == 15  # 5 initial + 10 concurrent


@pytest.mark.asyncio
async def test_database_locking_error_detection():
    """Test that database locking errors are properly detected."""
    from src.mcp_code_indexer.error_handler import ErrorHandler
    from src.mcp_code_indexer.middleware.error_middleware import ToolMiddleware

    error_handler = ErrorHandler(logging.getLogger(__name__))
    middleware = ToolMiddleware(error_handler)

    # Test various SQLite locking errors
    test_errors = [
        aiosqlite.OperationalError("database is locked"),
        aiosqlite.OperationalError("database is busy"),
        aiosqlite.OperationalError("SQLITE_BUSY: database is locked"),
        aiosqlite.OperationalError("cannot start a transaction within a transaction"),
        Exception("Some other error"),  # Non-locking error
    ]

    expected_results = [True, True, True, True, False]

    for error, expected in zip(test_errors, expected_results):
        result = middleware._is_database_locking_error(error)
        assert result == expected, f"Error detection failed for: {error}"


if __name__ == "__main__":
    # Run tests with asyncio
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
