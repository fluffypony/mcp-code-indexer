"""
Pytest configuration and fixtures for the MCP Code Indexer tests.

This module provides shared fixtures and configuration for async testing,
database setup, and mock data generation.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
import pytest
import pytest_asyncio

from mcp_code_indexer.database.database import DatabaseManager
from mcp_code_indexer.database.models import Project, FileDescription
from mcp_code_indexer.token_counter import TokenCounter
from mcp_code_indexer.merge_handler import MergeHandler
from mcp_code_indexer.error_handler import setup_error_handling
from mcp_code_indexer.logging_config import setup_logging


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def temp_db() -> AsyncGenerator[Path, None]:
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        db_path = Path(temp_file.name)
    
    yield db_path
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest_asyncio.fixture
async def db_manager(temp_db: Path) -> AsyncGenerator[DatabaseManager, None]:
    """Create and initialize a database manager for testing."""
    manager = DatabaseManager(temp_db)
    await manager.initialize()
    
    yield manager
    
    # Cleanup
    await manager.close_pool()


@pytest_asyncio.fixture
async def sample_project(db_manager: DatabaseManager) -> Project:
    """Create a sample project for testing."""
    project = Project(
        id="test_project_123",
        name="Test Project",
        remote_origin="https://github.com/test/repo.git",
        upstream_origin="https://github.com/upstream/repo.git",
        aliases=["test-project", "/path/to/project"]
    )
    
    await db_manager.create_project(project)
    return project


@pytest_asyncio.fixture
async def sample_file_descriptions(
    db_manager: DatabaseManager, 
    sample_project: Project
) -> list[FileDescription]:
    """Create sample file descriptions for testing."""
    descriptions = [
        FileDescription(
            project_id=sample_project.id,
            branch="main",
            file_path="src/main.py",
            description="Main entry point for the application with CLI argument parsing and server initialization.",
            file_hash="abc123",
            version=1
        ),
        FileDescription(
            project_id=sample_project.id,
            branch="main",
            file_path="src/database/models.py",
            description="Pydantic data models for projects, file descriptions, and search results.",
            file_hash="def456",
            version=1
        ),
        FileDescription(
            project_id=sample_project.id,
            branch="main",
            file_path="tests/test_main.py",
            description="Unit tests for the main module functionality.",
            file_hash="ghi789",
            version=1
        ),
        FileDescription(
            project_id=sample_project.id,
            branch="feature/new-ui",
            file_path="src/main.py",
            description="Main entry point with enhanced CLI interface and new UI components.",
            file_hash="abc124",
            version=2
        ),
        FileDescription(
            project_id=sample_project.id,
            branch="feature/new-ui",
            file_path="src/ui/components.py",
            description="React-like UI components for the new interface.",
            file_hash="jkl012",
            version=1
        )
    ]
    
    await db_manager.batch_create_file_descriptions(descriptions)
    return descriptions


@pytest.fixture
def token_counter() -> TokenCounter:
    """Create a token counter for testing."""
    return TokenCounter(token_limit=1000)  # Lower limit for testing


@pytest_asyncio.fixture
async def merge_handler(db_manager: DatabaseManager) -> MergeHandler:
    """Create a merge handler for testing."""
    return MergeHandler(db_manager)


@pytest.fixture
def mock_file_system(tmp_path: Path) -> Path:
    """Create a mock file system structure for testing."""
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    
    # Create some test files
    (project_root / "src").mkdir()
    (project_root / "src" / "main.py").write_text("# Main file")
    (project_root / "src" / "utils.py").write_text("# Utilities")
    (project_root / "tests").mkdir()
    (project_root / "tests" / "test_main.py").write_text("# Tests")
    (project_root / "README.md").write_text("# Test Project")
    (project_root / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    
    # Create some ignored files
    (project_root / "__pycache__").mkdir()
    (project_root / "__pycache__" / "main.cpython-39.pyc").write_text("binary")
    (project_root / "node_modules").mkdir()
    (project_root / "node_modules" / "package").mkdir()
    
    return project_root


@pytest.fixture
def setup_test_logging():
    """Set up logging for tests."""
    logger = setup_logging(log_level="DEBUG", enable_file_logging=False)
    error_handler = setup_error_handling(logger)
    return logger, error_handler


class MockGitRepository:
    """Mock git repository for testing."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.remotes = {
            "origin": "https://github.com/test/repo.git",
            "upstream": "https://github.com/upstream/repo.git"
        }
        self.current_branch = "main"
        self.branches = ["main", "develop", "feature/new-ui"]
    
    def get_remote_url(self, remote_name: str = "origin") -> str:
        """Get remote URL for a given remote name."""
        return self.remotes.get(remote_name, "")
    
    def get_current_branch(self) -> str:
        """Get the current branch name."""
        return self.current_branch
    
    def list_branches(self) -> list[str]:
        """List all branches."""
        return self.branches.copy()


@pytest.fixture
def mock_git_repo(mock_file_system: Path) -> MockGitRepository:
    """Create a mock git repository."""
    return MockGitRepository(mock_file_system)


# Performance testing fixtures

@pytest.fixture
def large_file_descriptions(sample_project: Project) -> list[FileDescription]:
    """Generate a large number of file descriptions for performance testing."""
    descriptions = []
    
    for i in range(1000):
        descriptions.append(FileDescription(
            project_id=sample_project.id,
            branch="main",
            file_path=f"src/module_{i:03d}.py",
            description=f"Module {i} containing utility functions and classes for feature set {i // 100}.",
            file_hash=f"hash_{i:03d}",
            version=1
        ))
    
    return descriptions


@pytest.fixture
def performance_markers():
    """Markers for performance testing."""
    return {
        "slow": pytest.mark.slow,
        "performance": pytest.mark.performance,
        "integration": pytest.mark.integration
    }


# Helper functions for tests

def assert_file_description_equal(actual: FileDescription, expected: FileDescription) -> None:
    """Assert that two file descriptions are equal (ignoring timestamps)."""
    assert actual.project_id == expected.project_id
    assert actual.branch == expected.branch
    assert actual.file_path == expected.file_path
    assert actual.description == expected.description
    assert actual.file_hash == expected.file_hash
    assert actual.version == expected.version
    assert actual.source_project_id == expected.source_project_id


def create_test_file_description(
    project_id: str = "test_project",
    branch: str = "main",
    file_path: str = "test.py",
    description: str = "Test file",
    file_hash: str = "test_hash"
) -> FileDescription:
    """Create a test file description with default values."""
    return FileDescription(
        project_id=project_id,
        branch=branch,
        file_path=file_path,
        description=description,
        file_hash=file_hash,
        version=1
    )


# Async context managers for testing

class AsyncTestContext:
    """Context manager for async test setup and teardown."""
    
    def __init__(self):
        self.resources = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources in reverse order
        for resource in reversed(self.resources):
            if hasattr(resource, 'close'):
                await resource.close()
            elif hasattr(resource, 'cleanup'):
                await resource.cleanup()
    
    def add_resource(self, resource):
        """Add a resource to be cleaned up."""
        self.resources.append(resource)
        return resource


@pytest_asyncio.fixture
async def async_test_context():
    """Provide an async test context for resource management."""
    async with AsyncTestContext() as context:
        yield context
