"""
Unit tests for DatabaseManager IndexMeta operations.

This module tests the create_index_meta, update_index_meta, and get_index_meta methods
including error handling, data persistence, and edge cases.
"""

from datetime import datetime, timezone
from typing import Any, Dict

import pytest
import pytest_asyncio

from mcp_code_indexer.database.database import DatabaseManager
from mcp_code_indexer.database.models import IndexMeta, Project, SyncStatus


class TestDatabaseManagerIndexMeta:
    """Test DatabaseManager IndexMeta operations."""

    @pytest_asyncio.fixture
    async def sample_index_meta(self, sample_project: Project) -> IndexMeta:
        """Create a sample IndexMeta for testing."""
        return IndexMeta(
            project_id=sample_project.id,
            total_chunks=100,
            indexed_chunks=75,
            total_files=25,
            indexed_files=20,
            last_sync=datetime.now(timezone.utc),
            sync_status=SyncStatus.IN_PROGRESS,
            error_message=None,
            queue_depth=5,
            processing_rate=2.5,
            estimated_completion=None,
            metadata={"version": "1.0", "last_indexer": "test-indexer"},
        )

    async def test_create_index_meta(
        self, db_manager: DatabaseManager, sample_index_meta: IndexMeta
    ) -> None:
        """Test creating new index metadata."""
        await db_manager.create_index_meta(sample_index_meta)

        # Verify the metadata was created
        retrieved = await db_manager.get_index_meta(sample_index_meta.project_id)
        assert retrieved is not None
        assert retrieved.project_id == sample_index_meta.project_id
        assert retrieved.total_chunks == sample_index_meta.total_chunks
        assert retrieved.indexed_chunks == sample_index_meta.indexed_chunks
        assert retrieved.total_files == sample_index_meta.total_files
        assert retrieved.indexed_files == sample_index_meta.indexed_files
        assert retrieved.sync_status == sample_index_meta.sync_status
        assert retrieved.error_message == sample_index_meta.error_message
        assert retrieved.queue_depth == sample_index_meta.queue_depth
        assert retrieved.processing_rate == sample_index_meta.processing_rate
        assert retrieved.metadata == sample_index_meta.metadata

    async def test_create_index_meta_upsert_behavior(
        self, db_manager: DatabaseManager, sample_index_meta: IndexMeta
    ) -> None:
        """Test that create_index_meta acts as upsert (INSERT OR REPLACE)."""
        # First create
        await db_manager.create_index_meta(sample_index_meta)
        
        # Modify some values and create again
        sample_index_meta.total_chunks = 150
        sample_index_meta.indexed_chunks = 100
        sample_index_meta.sync_status = SyncStatus.COMPLETED
        sample_index_meta.metadata = {"version": "2.0", "updated": True}
        
        await db_manager.create_index_meta(sample_index_meta)

        # Verify the record was updated, not duplicated
        retrieved = await db_manager.get_index_meta(sample_index_meta.project_id)
        assert retrieved is not None
        assert retrieved.total_chunks == 150
        assert retrieved.indexed_chunks == 100
        assert retrieved.sync_status == SyncStatus.COMPLETED
        assert retrieved.metadata == {"version": "2.0", "updated": True}

    async def test_update_index_meta(
        self, db_manager: DatabaseManager, sample_index_meta: IndexMeta
    ) -> None:
        """Test updating existing index metadata."""
        # First create the metadata
        await db_manager.create_index_meta(sample_index_meta)

        # Update values
        sample_index_meta.total_chunks = 200
        sample_index_meta.indexed_chunks = 150
        sample_index_meta.sync_status = SyncStatus.COMPLETED
        sample_index_meta.error_message = "Test error message"
        sample_index_meta.metadata = {"updated": True, "version": "1.1"}

        await db_manager.update_index_meta(sample_index_meta)

        # Verify the update
        retrieved = await db_manager.get_index_meta(sample_index_meta.project_id)
        assert retrieved is not None
        assert retrieved.total_chunks == 200
        assert retrieved.indexed_chunks == 150
        assert retrieved.sync_status == SyncStatus.COMPLETED
        assert retrieved.error_message == "Test error message"
        assert retrieved.metadata == {"updated": True, "version": "1.1"}

    async def test_update_index_meta_nonexistent_project(
        self, db_manager: DatabaseManager
    ) -> None:
        """Test updating metadata for non-existent project raises DatabaseError."""
        from mcp_code_indexer.database.exceptions import DatabaseError
        
        nonexistent_meta = IndexMeta(
            project_id="nonexistent-project-id",
            total_chunks=10,
            indexed_chunks=5,
            total_files=2,
            indexed_files=1,
            sync_status=SyncStatus.PENDING,
        )

        with pytest.raises(DatabaseError, match="Index metadata not found for project"):
            await db_manager.update_index_meta(nonexistent_meta)

    async def test_get_index_meta_existing(
        self, db_manager: DatabaseManager, sample_index_meta: IndexMeta
    ) -> None:
        """Test retrieving existing index metadata."""
        await db_manager.create_index_meta(sample_index_meta)

        retrieved = await db_manager.get_index_meta(sample_index_meta.project_id)
        assert retrieved is not None
        assert retrieved.project_id == sample_index_meta.project_id
        assert retrieved.total_chunks == sample_index_meta.total_chunks
        assert retrieved.sync_status == sample_index_meta.sync_status

    async def test_get_index_meta_nonexistent(
        self, db_manager: DatabaseManager
    ) -> None:
        """Test retrieving metadata for non-existent project returns None."""
        retrieved = await db_manager.get_index_meta("nonexistent-project-id")
        assert retrieved is None

    async def test_index_meta_with_empty_metadata_dict(
        self, db_manager: DatabaseManager, sample_project: Project
    ) -> None:
        """Test IndexMeta with empty metadata dictionary."""
        index_meta = IndexMeta(
            project_id=sample_project.id,
            total_chunks=50,
            indexed_chunks=25,
            total_files=10,
            indexed_files=5,
            sync_status=SyncStatus.PENDING,
            metadata={},  # Empty dict
        )

        await db_manager.create_index_meta(index_meta)
        retrieved = await db_manager.get_index_meta(sample_project.id)
        
        assert retrieved is not None
        assert retrieved.metadata == {}

    async def test_index_meta_with_complex_metadata(
        self, db_manager: DatabaseManager, sample_project: Project
    ) -> None:
        """Test IndexMeta with complex metadata structure."""
        complex_metadata = {
            "version": "2.0",
            "config": {
                "batch_size": 100,
                "timeout": 30.0,
                "retry_count": 3,
                "features": ["vector_search", "semantic_analysis"]
            },
            "stats": {
                "avg_chunk_size": 256,
                "max_chunk_size": 1024,
                "file_types": ["py", "js", "ts", "md"]
            }
        }

        index_meta = IndexMeta(
            project_id=sample_project.id,
            total_chunks=1000,
            indexed_chunks=750,
            total_files=100,
            indexed_files=85,
            sync_status=SyncStatus.IN_PROGRESS,
            metadata=complex_metadata,
        )

        await db_manager.create_index_meta(index_meta)
        retrieved = await db_manager.get_index_meta(sample_project.id)
        
        assert retrieved is not None
        assert retrieved.metadata == complex_metadata

    async def test_index_meta_datetime_handling(
        self, db_manager: DatabaseManager, sample_project: Project
    ) -> None:
        """Test proper datetime handling in IndexMeta operations."""
        now = datetime.now(timezone.utc)
        future_time = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59)

        index_meta = IndexMeta(
            project_id=sample_project.id,
            total_chunks=100,
            indexed_chunks=50,
            total_files=10,
            indexed_files=5,
            last_sync=now,
            sync_status=SyncStatus.IN_PROGRESS,
            estimated_completion=future_time,
        )

        await db_manager.create_index_meta(index_meta)
        retrieved = await db_manager.get_index_meta(sample_project.id)
        
        assert retrieved is not None
        # Note: SQLite doesn't preserve timezone info, so we compare with some tolerance
        assert abs((retrieved.last_sync - now).total_seconds()) < 1
        assert abs((retrieved.estimated_completion - future_time).total_seconds()) < 1
        assert retrieved.created is not None
        assert retrieved.last_modified is not None

    async def test_index_meta_nullable_fields(
        self, db_manager: DatabaseManager, sample_project: Project
    ) -> None:
        """Test IndexMeta with nullable fields set to None."""
        index_meta = IndexMeta(
            project_id=sample_project.id,
            total_chunks=100,
            indexed_chunks=50,
            total_files=10,
            indexed_files=5,
            last_sync=None,  # Nullable
            sync_status=SyncStatus.PENDING,
            error_message=None,  # Nullable
            estimated_completion=None,  # Nullable
        )

        await db_manager.create_index_meta(index_meta)
        retrieved = await db_manager.get_index_meta(sample_project.id)
        
        assert retrieved is not None
        assert retrieved.last_sync is None
        assert retrieved.error_message is None
        assert retrieved.estimated_completion is None

    async def test_multiple_projects_index_meta(
        self, db_manager: DatabaseManager
    ) -> None:
        """Test creating index metadata for multiple projects."""
        # Create multiple projects
        project1 = Project(id="project_1", name="Project One")
        project2 = Project(id="project_2", name="Project Two")
        
        await db_manager.create_project(project1)
        await db_manager.create_project(project2)

        # Create index metadata for each
        index_meta1 = IndexMeta(
            project_id=project1.id,
            total_chunks=100,
            indexed_chunks=50,
            total_files=10,
            indexed_files=5,
            sync_status=SyncStatus.IN_PROGRESS,
        )

        index_meta2 = IndexMeta(
            project_id=project2.id,
            total_chunks=200,
            indexed_chunks=150,
            total_files=20,
            indexed_files=18,
            sync_status=SyncStatus.COMPLETED,
        )

        await db_manager.create_index_meta(index_meta1)
        await db_manager.create_index_meta(index_meta2)

        # Verify both exist independently
        retrieved1 = await db_manager.get_index_meta(project1.id)
        retrieved2 = await db_manager.get_index_meta(project2.id)

        assert retrieved1 is not None
        assert retrieved2 is not None
        assert retrieved1.total_chunks == 100
        assert retrieved2.total_chunks == 200
        assert retrieved1.sync_status == SyncStatus.IN_PROGRESS
        assert retrieved2.sync_status == SyncStatus.COMPLETED

    async def test_get_or_create_index_meta_creates_new(
        self, db_manager: DatabaseManager, sample_project: Project
    ) -> None:
        """Test get_or_create_index_meta creates new metadata when none exists."""
        # Should not exist initially
        initial = await db_manager.get_index_meta(sample_project.id)
        assert initial is None

        # Get or create should create new metadata with defaults
        result = await db_manager.get_or_create_index_meta(sample_project.id)
        
        assert result is not None
        assert result.project_id == sample_project.id
        assert result.total_chunks == 0
        assert result.indexed_chunks == 0
        assert result.total_files == 0
        assert result.indexed_files == 0
        assert result.sync_status == SyncStatus.PENDING
        assert result.error_message is None
        assert result.queue_depth == 0
        assert result.processing_rate == 0.0
        assert result.metadata == {}

    async def test_get_or_create_index_meta_returns_existing(
        self, db_manager: DatabaseManager, sample_index_meta: IndexMeta
    ) -> None:
        """Test get_or_create_index_meta returns existing metadata when it exists."""
        # Create metadata first
        await db_manager.create_index_meta(sample_index_meta)

        # Get or create should return the existing metadata
        result = await db_manager.get_or_create_index_meta(sample_index_meta.project_id)
        
        assert result is not None
        assert result.project_id == sample_index_meta.project_id
        assert result.total_chunks == sample_index_meta.total_chunks
        assert result.indexed_chunks == sample_index_meta.indexed_chunks
        assert result.sync_status == sample_index_meta.sync_status
        assert result.metadata == sample_index_meta.metadata

    async def test_get_or_create_index_meta_with_custom_defaults(
        self, db_manager: DatabaseManager, sample_project: Project
    ) -> None:
        """Test get_or_create_index_meta with custom default values."""
        custom_metadata = {"custom": "value", "version": "2.0"}
        
        result = await db_manager.get_or_create_index_meta(
            sample_project.id,
            total_chunks=50,
            indexed_chunks=25,
            sync_status=SyncStatus.IN_PROGRESS,
            metadata=custom_metadata,
        )
        
        assert result is not None
        assert result.project_id == sample_project.id
        assert result.total_chunks == 50
        assert result.indexed_chunks == 25
        assert result.sync_status == SyncStatus.IN_PROGRESS
        assert result.metadata == custom_metadata
        # Other fields should still have defaults
        assert result.total_files == 0
        assert result.indexed_files == 0
        assert result.queue_depth == 0

    async def test_get_or_create_index_meta_idempotent(
        self, db_manager: DatabaseManager, sample_project: Project
    ) -> None:
        """Test get_or_create_index_meta is idempotent - multiple calls return same data."""
        # First call creates the metadata
        result1 = await db_manager.get_or_create_index_meta(
            sample_project.id,
            total_chunks=100,
            sync_status=SyncStatus.IN_PROGRESS,
        )
        
        # Second call should return the same data (existing record)
        result2 = await db_manager.get_or_create_index_meta(
            sample_project.id,
            total_chunks=200,  # This should be ignored since record exists
            sync_status=SyncStatus.COMPLETED,  # This should be ignored too
        )
        
        assert result1.id == result2.id
        assert result1.project_id == result2.project_id
        assert result1.total_chunks == result2.total_chunks == 100
        assert result1.sync_status == result2.sync_status == SyncStatus.IN_PROGRESS
