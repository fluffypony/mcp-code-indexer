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
from _pytest.fixtures import FixtureRequest

from mcp_code_indexer.database.database import DatabaseManager
from mcp_code_indexer.database.models import FileDescription, Project
from mcp_code_indexer.error_handler import setup_error_handling
from mcp_code_indexer.logging_config import setup_logging
from mcp_code_indexer.token_counter import TokenCounter


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
async def temp_db_with_test_table(temp_db: Path) -> AsyncGenerator[Path, None]:
    """Create a temporary database with a test table for retry testing."""
    import aiosqlite

    # Create database with tables
    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            """
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                data TEXT NOT NULL
            )
        """
        )
        await db.commit()

    yield temp_db


@pytest_asyncio.fixture
async def db_manager(temp_db: Path) -> AsyncGenerator[DatabaseManager, None]:
    """Create and initialize a database manager for testing."""
    manager = DatabaseManager(temp_db)
    await manager.initialize()

    yield manager

    # Cleanup
    await manager.close_pool()


@pytest_asyncio.fixture
async def temp_db_manager_pool(
    temp_db: Path, request: FixtureRequest
) -> AsyncGenerator[DatabaseManager, None]:
    """Create and initialize a database manager with specified pool size for testing.

    Use with @pytest.mark.parametrize("temp_db_manager_pool", [pool_size], indirect=True)
    """
    pool_size = request.param
    manager = DatabaseManager(temp_db, pool_size=pool_size)
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
        aliases=["test-project", "/path/to/project"],
    )

    await db_manager.create_project(project)
    return project


@pytest_asyncio.fixture
async def sample_file_descriptions(
    db_manager: DatabaseManager, sample_project: Project
) -> list[FileDescription]:
    """Create sample file descriptions for testing."""
    descriptions = [
        FileDescription(
            project_id=sample_project.id,
            file_path="src/main.py",
            description=(
                "Main entry point for the application with CLI argument "
                "parsing and server initialization."
            ),
            file_hash="abc123",
            version=1,
        ),
        FileDescription(
            project_id=sample_project.id,
            file_path="src/database/models.py",
            description=(
                "Pydantic data models for projects, file descriptions, "
                "and search results."
            ),
            file_hash="def456",
            version=1,
        ),
        FileDescription(
            project_id=sample_project.id,
            file_path="tests/test_main.py",
            description="Unit tests for the main module functionality.",
            file_hash="ghi789",
            version=1,
        ),
        FileDescription(
            project_id=sample_project.id,
            file_path="src/main_enhanced.py",
            description=(
                "Enhanced main entry point with improved CLI interface "
                "and extended functionality."
            ),
            file_hash="abc124",
            version=2,
        ),
        FileDescription(
            project_id=sample_project.id,
            file_path="src/ui/components.py",
            description="UI components for the enhanced interface.",
            file_hash="jkl012",
            version=1,
        ),
    ]

    await db_manager.batch_create_file_descriptions(descriptions)
    return descriptions


@pytest.fixture
def token_counter() -> TokenCounter:
    """Create a token counter for testing."""
    return TokenCounter(token_limit=1000)  # Lower limit for testing


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
            "upstream": "https://github.com/upstream/repo.git",
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
def large_file_descriptions(
    sample_project: Project,
) -> list[FileDescription]:
    """Generate a large number of file descriptions for performance testing."""
    descriptions = []

    for i in range(1000):
        descriptions.append(
            FileDescription(
                project_id=sample_project.id,
                file_path=f"src/module_{i:03d}.py",
                description=(
                    f"Module {i} containing utility functions and classes "
                    f"for feature set {i // 100}."
                ),
                file_hash=f"hash_{i:03d}",
                version=1,
            )
        )

    return descriptions


@pytest.fixture
def performance_markers():
    """Markers for performance testing."""
    return {
        "slow": pytest.mark.slow,
        "performance": pytest.mark.performance,
        "integration": pytest.mark.integration,
    }


# Helper functions for tests


def assert_file_description_equal(
    actual: FileDescription, expected: FileDescription
) -> None:
    """Assert that two file descriptions are equal (ignoring timestamps)."""
    assert actual.project_id == expected.project_id
    assert actual.file_path == expected.file_path
    assert actual.description == expected.description
    assert actual.file_hash == expected.file_hash
    assert actual.version == expected.version
    assert actual.source_project_id == expected.source_project_id


def create_test_file_description(
    project_id: str = "test_project",
    file_path: str = "test.py",
    description: str = "Test file",
    file_hash: str = "test_hash",
) -> FileDescription:
    """Create a test file description with default values."""
    return FileDescription(
        project_id=project_id,
        file_path=file_path,
        description=description,
        file_hash=file_hash,
        version=1,
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
            if hasattr(resource, "close"):
                await resource.close()
            elif hasattr(resource, "cleanup"):
                await resource.cleanup()

    def add_resource(self, resource):
        """Add a resource to be cleaned up."""
        self.resources.append(resource)
        return resource


@pytest_asyncio.fixture
async def mcp_server(tmp_path: Path):
    """Create an MCP server for testing."""
    db_path = tmp_path / "test.db"
    cache_dir = tmp_path / "cache"

    # Import here to avoid circular imports
    from mcp_code_indexer.server.mcp_server import MCPCodeIndexServer

    server = MCPCodeIndexServer(token_limit=1000, db_path=db_path, cache_dir=cache_dir)
    await server.initialize()

    yield server

    # Cleanup
    if hasattr(server, "shutdown"):
        await server.shutdown()
    elif hasattr(server, "db_manager") and server.db_manager:
        await server.db_manager.close_pool()


@pytest_asyncio.fixture
async def async_test_context():
    """Provide an async test context for resource management."""
    async with AsyncTestContext() as context:
        yield context


# Secret generation helpers for testing

def generate_fake_aws_access_key() -> str:
    """Generate a fake AWS access key for testing."""
    import random
    import string
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
    return f"AKIA{suffix}"


def generate_fake_github_token() -> str:
    """Generate a fake GitHub token for testing."""
    import random
    import string
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=36))
    return f"ghp_{suffix}"


def generate_fake_google_api_key() -> str:
    """Generate a fake Google API key for testing."""
    import random
    import string
    suffix = ''.join(random.choices(string.ascii_letters + string.digits + '-_', k=35))
    return f"AIza{suffix}"


def generate_fake_openai_key() -> str:
    """Generate a fake OpenAI API key for testing."""
    import random
    import string
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=48))
    return f"sk-{suffix}"


def generate_fake_anthropic_key() -> str:
    """Generate a fake Anthropic API key for testing."""
    import random
    import string
    suffix = ''.join(random.choices(string.ascii_letters + string.digits + '-_', k=95))
    return f"sk-ant-api03-{suffix}"


def generate_fake_jwt_token() -> str:
    """Generate a fake JWT token for testing."""
    import random
    import string
    import base64
    
    def random_base64(length):
        chars = string.ascii_letters + string.digits
        return base64.b64encode(''.join(random.choices(chars, k=length)).encode()).decode().rstrip('=')
    
    header = random_base64(20)
    payload = random_base64(30)
    signature = random_base64(25)
    return f"eyJ{header}.eyJ{payload}.{signature}"


def generate_fake_connection_strings() -> dict:
    """Generate fake database connection strings for testing."""
    import random
    import string
    
    def random_string(length):
        return ''.join(random.choices(string.ascii_lowercase, k=length))
    
    user = f"test_{random_string(6)}"
    password = f"pass_{random_string(8)}"
    host = f"host_{random_string(5)}"
    db = f"db_{random_string(6)}"
    
    return {
        'postgres_url': f"postgresql://{user}:{password}@{host}:5432/{db}",
        'redis_url': f"redis://{user}:{password}@{host}:6379/0",
        'mongodb_url': f"mongodb://{user}:{password}@{host}:27017/{db}"
    }


@pytest.fixture
def fake_secrets():
    """Fixture providing dynamically generated fake secrets for testing."""
    connection_strings = generate_fake_connection_strings()
    return {
        'aws_access_key': generate_fake_aws_access_key(),
        'github_token': generate_fake_github_token(),
        'google_api_key': generate_fake_google_api_key(),
        'openai_key': generate_fake_openai_key(),
        'anthropic_key': generate_fake_anthropic_key(),
        'jwt_token': generate_fake_jwt_token(),
        **connection_strings,
    }
