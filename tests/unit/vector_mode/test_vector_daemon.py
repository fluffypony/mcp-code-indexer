"""
Unit tests for VectorDaemon monitored_projects management.

This module tests the VectorDaemon._get_project_monitoring_status method including:
- Monitoring status when no projects exist
- Monitoring new vector-enabled projects
- Unmonitoring projects when vector mode is disabled
- Handling projects without aliases
"""

from typing import Any

import pytest
import pytest_asyncio

from mcp_code_indexer.database.database import DatabaseManager
from mcp_code_indexer.database.models import Project
from mcp_code_indexer.vector_mode.daemon import VectorDaemon


class TestVectorDaemonMonitoringStatus:
    """Test VectorDaemon monitored_projects management."""
    
    @pytest.fixture
    async def vector_daemon(self, db_manager: DatabaseManager) -> VectorDaemon:
        """Create a VectorDaemon for testing."""
        from mcp_code_indexer.vector_mode.config import VectorConfig
        from pathlib import Path
        
        config = VectorConfig()
        cache_dir = Path("/tmp/test_cache")
        daemon = VectorDaemon(config, db_manager, cache_dir)
        return daemon

    async def test_get_project_monitoring_status_empty(self, db_manager: DatabaseManager, vector_daemon: VectorDaemon) -> None:
        """Test monitoring status when no projects exist."""
        status = await vector_daemon._get_project_monitoring_status()
        
        assert status["monitored"] == []
        assert status["unmonitored"] == []
        assert len(vector_daemon.monitored_projects) == 0

    async def test_get_project_monitoring_status_new_project(self, db_manager: DatabaseManager, sample_project: Project, vector_daemon: VectorDaemon) -> None:
        """Test monitoring status with a new vector-enabled project."""
        # Enable vector mode for the project
        await db_manager.set_project_vector_mode(sample_project.id, True)
        
        # Project already has aliases from fixture, so it should be monitorable
        status = await vector_daemon._get_project_monitoring_status()
        
        # Should be marked for monitoring
        assert len(status["monitored"]) == 1
        assert status["monitored"][0].name == sample_project.name
        assert len(status["unmonitored"]) == 0

    async def test_get_project_monitoring_status_unmonitor_project(self, db_manager: DatabaseManager, sample_project: Project, vector_daemon: VectorDaemon) -> None:
        """Test monitoring status when project should be unmonitored."""
        # Setup: enable vector mode (project already has aliases from fixture)
        await db_manager.set_project_vector_mode(sample_project.id, True)
        
        # Simulate project being monitored
        vector_daemon.monitored_projects.add(sample_project.name)
        
        # Now disable vector mode
        await db_manager.set_project_vector_mode(sample_project.id, False)
        
        status = await vector_daemon._get_project_monitoring_status()
        
        # Should be marked for unmonitoring
        assert len(status["monitored"]) == 0
        assert len(status["unmonitored"]) == 1
        assert status["unmonitored"][0].name == sample_project.name

    async def test_get_project_monitoring_status_no_aliases(self, db_manager: DatabaseManager, vector_daemon: VectorDaemon) -> None:
        """Test that projects without aliases are not monitored."""
        # Create a project without aliases
        from mcp_code_indexer.database.models import Project
        from datetime import datetime
        
        project_no_aliases = Project(
            id="no_aliases_project",
            name="no aliases project", 
            aliases=[],  # No aliases
            created=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            vector_mode=False
        )
        await db_manager.create_project(project_no_aliases)
        
        # Enable vector mode
        await db_manager.set_project_vector_mode(project_no_aliases.id, True)
        
        status = await vector_daemon._get_project_monitoring_status()
        
        # Should not be monitored due to missing aliases
        assert len(status["monitored"]) == 0
        assert len(status["unmonitored"]) == 0
