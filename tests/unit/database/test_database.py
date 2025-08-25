"""
Unit tests for DatabaseManager vector mode operations.

This module tests the DatabaseManager.set_project_vector_mode method including:
- Enabling and disabling vector mode
- Error handling for nonexistent projects
- Toggling vector mode multiple times
- Edge cases like setting same value twice and concurrent operations
"""

from typing import Any

import pytest
import pytest_asyncio

from mcp_code_indexer.database.database import DatabaseManager
from mcp_code_indexer.database.models import Project


class TestDatabaseManagerVectorMode:
    """Test DatabaseManager vector mode operations."""

    async def test_set_project_vector_mode_enable(self, db_manager: DatabaseManager, sample_project: Project) -> None:
        """Test enabling vector mode for a project."""
        # Enable vector mode
        await db_manager.set_project_vector_mode(sample_project.id, True)
        
        # Verify the change
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project is not None
        assert project.vector_mode is True

    async def test_set_project_vector_mode_disable(self, db_manager: DatabaseManager, sample_project: Project) -> None:
        """Test disabling vector mode for a project."""
        # First enable it
        await db_manager.set_project_vector_mode(sample_project.id, True)
        
        # Then disable it
        await db_manager.set_project_vector_mode(sample_project.id, False)
        
        # Verify the change
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project is not None
        assert project.vector_mode is False

    async def test_set_project_vector_mode_nonexistent_project(self, db_manager: DatabaseManager) -> None:
        """Test error handling for nonexistent project."""
        from mcp_code_indexer.database.exceptions import DatabaseError
        with pytest.raises(DatabaseError, match="Project not found"):
            await db_manager.set_project_vector_mode("nonexistent_id", True)

    async def test_set_project_vector_mode_toggle(self, db_manager: DatabaseManager, sample_project: Project) -> None:
        """Test toggling vector mode multiple times."""
        # Default should be False
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is False
        
        # Enable
        await db_manager.set_project_vector_mode(sample_project.id, True)
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is True
        
        # Disable
        await db_manager.set_project_vector_mode(sample_project.id, False)
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is False
        
        # Enable again
        await db_manager.set_project_vector_mode(sample_project.id, True)
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is True


class TestVectorModeEdgeCases:
    """Test edge cases for vector mode operations."""

    async def test_set_same_value_twice(self, db_manager: DatabaseManager, sample_project: Project) -> None:
        """Test setting the same vector mode value twice."""
        # Enable twice
        await db_manager.set_project_vector_mode(sample_project.id, True)
        await db_manager.set_project_vector_mode(sample_project.id, True)
        
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is True
        
        # Disable twice
        await db_manager.set_project_vector_mode(sample_project.id, False)
        await db_manager.set_project_vector_mode(sample_project.id, False)
        
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project.vector_mode is False

    async def test_concurrent_vector_mode_operations(self, db_manager: DatabaseManager, sample_project: Project) -> None:
        """Test concurrent vector mode operations."""
        import asyncio
        
        # Run multiple operations concurrently
        tasks = [
            db_manager.set_project_vector_mode(sample_project.id, True),
            db_manager.set_project_vector_mode(sample_project.id, False),
            db_manager.set_project_vector_mode(sample_project.id, True),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Final state should be deterministic (last operation wins)
        projects = await db_manager.get_all_projects()
        project = next((p for p in projects if p.id == sample_project.id), None)
        assert project is not None
        # The final state will be True since that's the last operation
        assert project.vector_mode is True
