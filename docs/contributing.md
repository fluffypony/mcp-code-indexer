# Contributing Guide 🤝

Welcome to the MCP Code Indexer project! We're thrilled you want to help make this AI-powered codebase navigation tool even better. Whether you're fixing a bug, adding a feature, or improving documentation, your contribution makes a difference.

**🎯 New contributor?** Start with [Getting Started](#getting-started) for a quick setup guide.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Performance Considerations](#performance-considerations)
- [Security Guidelines](#security-guidelines)

## Getting Started

### Prerequisites

**Required**:
- **Python 3.8+** with asyncio support
- **Git** for version control  
- **SQLite 3.35+** (included with Python)

**Recommended**:
- **VS Code** with Python extension for the best development experience
- **Basic familiarity** with async Python and SQLite

### Development Environment

```bash
# Clone the repository
git clone https://github.com/your-username/mcp-code-indexer.git
cd mcp-code-indexer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package in editable mode (REQUIRED)
pip install -e .

# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Verify setup
python main.py --help
python -m pytest tests/ -v
```

> **⚠️ Important:** The editable install (`pip install -e .`) is **required** for development. The project uses proper PyPI package structure with absolute imports like `from mcp_code_indexer.database.database import DatabaseManager`. Without editable installation, you'll get `ModuleNotFoundError` exceptions.

### Project Structure

```
mcp-code-indexer/
├── src/                    # Source code
│   ├── database/          # Data models and operations
│   ├── server/            # MCP server implementation
│   ├── middleware/        # Error handling and middleware
│   ├── tiktoken_cache/    # Token counting cache
│   └── *.py              # Core modules
├── tests/                 # Test suite
│   ├── integration/       # Integration tests
│   ├── conftest.py       # Test configuration
│   └── test_*.py         # Unit tests
├── docs/                  # Documentation
├── migrations/            # Database migrations
├── examples/              # Usage examples
└── main.py               # Entry point
```

## Development Setup

### 🔧 IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

### 🐍 Python Environment

**Requirements** (`requirements-dev.txt`):

```text
# Testing
pytest>=8.0.0
pytest-asyncio>=0.21.0
pytest-mock>=3.11.0
pytest-cov>=4.0.0

# Code quality
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0

# Development tools
pre-commit>=3.0.0
```

### 🔍 Pre-commit Configuration

**Pre-commit hooks** (`.pre-commit-config.yaml`):

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## Code Standards

### 🎨 Code Style

We follow **Black** formatting with these standards:

```python
# Good: Clear, descriptive names
async def get_file_description(
    project_id: str, 
    branch: str, 
    file_path: str
) -> Optional[FileDescription]:
    """
    Retrieve file description from database.
    
    Args:
        project_id: Unique project identifier
        branch: Git branch name
        file_path: Relative path from project root
        
    Returns:
        FileDescription if found, None otherwise
        
    Raises:
        DatabaseError: If database operation fails
    """
    async with self.db_manager.get_connection() as conn:
        # Implementation...

# Good: Type hints and docstrings
class MergeConflict:
    """Represents a merge conflict between file descriptions."""
    
    def __init__(
        self,
        file_path: str,
        source_description: str,
        target_description: str
    ) -> None:
        self.file_path = file_path
        self.source_description = source_description
        self.target_description = target_description
```

### 📝 Documentation Standards

**Docstring Format** (Google style):

```python
async def search_file_descriptions(
    self,
    project_id: str,
    branch: str, 
    query: str,
    max_results: int = 20
) -> List[SearchResult]:
    """
    Search file descriptions using full-text search.
    
    Performs FTS5 search across file descriptions with relevance ranking.
    Results are ordered by relevance score (highest first).
    
    Args:
        project_id: Project to search within
        branch: Branch to search
        query: Search query string (supports FTS5 syntax)
        max_results: Maximum number of results to return
        
    Returns:
        List of SearchResult objects ordered by relevance
        
    Raises:
        ValidationError: If query is invalid or too long
        DatabaseError: If search operation fails
        
    Example:
        >>> results = await db.search_file_descriptions(
        ...     "proj_123", "main", "authentication middleware", 10
        ... )
        >>> print(f"Found {len(results)} matches")
    """
```

### 🔒 Type Safety

**Comprehensive type hints**:

```python
from typing import List, Optional, Dict, Any, Union, Protocol
from pathlib import Path

# Use type aliases for clarity
ProjectId = str
FilePath = str
TokenCount = int

class DatabaseProtocol(Protocol):
    """Protocol for database operations."""
    
    async def get_file_description(
        self, 
        project_id: ProjectId, 
        branch: str, 
        file_path: FilePath
    ) -> Optional[FileDescription]: ...

# Generic types for reusability
from typing import TypeVar, Generic

T = TypeVar('T')

class Repository(Generic[T]):
    """Generic repository pattern."""
    
    async def get_by_id(self, id: str) -> Optional[T]: ...
    async def save(self, entity: T) -> None: ...
```

## Testing Guidelines

### 🧪 Test Structure

**Comprehensive test coverage** with async support:

```python
# tests/test_merge_handler.py
import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock

from src.merge_handler import MergeHandler, MergeConflict
from tests.conftest import sample_project, sample_file_descriptions

class TestMergeHandler:
    """Test merge functionality with various scenarios."""
    
    @pytest_asyncio.fixture
    async def merge_handler(self, db_manager):
        """Create merge handler for testing."""
        return MergeHandler(db_manager)
    
    async def test_merge_no_conflicts(self, merge_handler, sample_project):
        """Test merge when no conflicts exist."""
        # Setup test data
        # Execute operation
        # Assert expected results
        
    async def test_merge_with_conflicts(self, merge_handler, sample_project):
        """Test merge conflict detection and resolution."""
        # Test conflict detection
        # Test resolution validation
        # Test merge completion
        
    @pytest.mark.parametrize("source_branch,target_branch,expected_conflicts", [
        ("feature/auth", "main", 1),
        ("feature/ui", "develop", 0),
        ("hotfix/security", "main", 2),
    ])
    async def test_merge_scenarios(
        self, 
        merge_handler, 
        source_branch, 
        target_branch, 
        expected_conflicts
    ):
        """Test various merge scenarios."""
        # Parameterized testing for different scenarios
```

### 🏃 Performance Testing

**Benchmark critical operations**:

```python
import time
import pytest

@pytest.mark.performance
class TestPerformance:
    """Performance benchmarks for critical operations."""
    
    async def test_large_search_performance(self, db_manager_with_data):
        """Search should complete quickly even with large datasets."""
        start_time = time.time()
        
        results = await db_manager_with_data.search_file_descriptions(
            "large_project", "main", "authentication", 50
        )
        
        duration = time.time() - start_time
        
        assert len(results) > 0
        assert duration < 0.5  # 500ms limit
        
    async def test_token_counting_performance(self, token_counter):
        """Token counting should be efficient for large codebases."""
        # Create large dataset
        descriptions = [create_test_description() for _ in range(1000)]
        
        start_time = time.time()
        total_tokens = token_counter.calculate_codebase_tokens(descriptions)
        duration = time.time() - start_time
        
        assert total_tokens > 0
        assert duration < 2.0  # 2 second limit
```

### 🔧 Test Utilities

**Helper functions and fixtures**:

```python
# tests/conftest.py
import pytest_asyncio
from pathlib import Path
import tempfile

@pytest_asyncio.fixture
async def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    
    yield db_path
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()

def create_test_file_description(**overrides):
    """Factory function for test file descriptions."""
    defaults = {
        "project_id": "test_project",
        "branch": "main",
        "file_path": "test.py",
        "description": "Test file description",
        "file_hash": "abc123"
    }
    defaults.update(overrides)
    return FileDescription(**defaults)
```

### 🎯 Test Categories

Run specific test categories:

```bash
# Unit tests only
python -m pytest tests/ -v -k "not integration and not performance"

# Integration tests
python -m pytest tests/integration/ -v

# Performance tests
python -m pytest tests/ -v -m performance

# Coverage report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
```

## Pull Request Process

### 🚀 PR Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   git checkout -b fix/issue-description
   git checkout -b docs/update-api-reference
   ```

2. **Development Process**
   ```bash
   # Make your changes
   # Write/update tests
   # Update documentation if needed
   
   # Run tests locally
   python -m pytest tests/ -v
   
   # Check code quality
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

3. **Commit Guidelines**
   ```bash
   # Use conventional commits
   git commit -m "feat: add merge conflict resolution"
   git commit -m "fix: handle edge case in token counting"
   git commit -m "docs: update API reference examples"
   git commit -m "test: add integration tests for search"
   git commit -m "perf: optimize database connection pooling"
   ```

### 📋 PR Checklist

**Before submitting your PR**:

- [ ] **Tests**: All new code has comprehensive tests
- [ ] **Documentation**: Updated relevant documentation
- [ ] **Type Hints**: All functions have proper type annotations
- [ ] **Performance**: No significant performance regressions
- [ ] **Compatibility**: Changes maintain backward compatibility
- [ ] **Error Handling**: Proper error handling and logging
- [ ] **Code Quality**: Passes all linting and formatting checks

### 📝 PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Performance Impact
- [ ] No performance impact
- [ ] Performance improvement
- [ ] Potential performance regression (explain mitigation)

## Documentation
- [ ] API documentation updated
- [ ] README updated if needed
- [ ] Configuration guide updated if needed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests pass locally
- [ ] Documentation is clear and accurate
```

## Issue Guidelines

### 🐛 Bug Reports

```markdown
**Bug Description**
Clear description of the bug.

**To Reproduce**
Steps to reproduce:
1. Configure server with...
2. Call MCP tool...
3. Observe error...

**Expected Behavior**
What should happen instead.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.2]
- MCP Code Indexer version: [e.g., 1.2.0]

**Additional Context**
- Error logs
- Configuration details
- Related issues
```

### 💡 Feature Requests

```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
- Related features
- Implementation ideas
- API design considerations
```

## Performance Considerations

### 🚄 Optimization Guidelines

**Database Operations**:
- Use connection pooling for all database access
- Prefer batch operations over individual queries
- Add indexes for new query patterns
- Use EXPLAIN QUERY PLAN to verify performance

**Memory Management**:
- Process large datasets in batches
- Use async generators for streaming operations
- Clean up resources in finally blocks
- Monitor memory usage in tests

**I/O Operations**:
- Make all file operations async
- Respect .gitignore and file filters
- Use appropriate timeouts
- Cache results when beneficial

### 📊 Performance Testing

**Benchmark new features**:

```python
@pytest.mark.performance
async def test_new_feature_performance():
    """New feature should meet performance requirements."""
    # Setup large dataset
    # Measure operation time
    # Assert performance constraints
    # Record baseline metrics
```

## Security Guidelines

### 🔒 Security Best Practices

**Input Validation**:
```python
def validate_file_path(file_path: str) -> str:
    """Validate and sanitize file paths."""
    # Check for path traversal
    if ".." in file_path or file_path.startswith("/"):
        raise ValidationError("Invalid file path")
    
    # Normalize path
    return Path(file_path).as_posix()

def validate_project_name(name: str) -> str:
    """Validate project names."""
    if not name or len(name) > 100:
        raise ValidationError("Invalid project name")
    
    # Allow alphanumeric, hyphens, underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValidationError("Project name contains invalid characters")
    
    return name
```

**SQL Safety**:
```python
# Good: Parameterized queries
async def get_descriptions_safe(self, project_id: str, branch: str):
    """Safe database query with parameters."""
    cursor = await conn.execute(
        "SELECT * FROM file_descriptions WHERE project_id = ? AND branch = ?",
        (project_id, branch)
    )
    return await cursor.fetchall()

# Bad: String concatenation (NEVER DO THIS)
# query = f"SELECT * FROM files WHERE id = '{user_input}'"
```

**Error Information**:
```python
def sanitize_error_for_client(error: Exception) -> str:
    """Remove sensitive information from error messages."""
    message = str(error)
    
    # Remove file paths
    message = re.sub(r'/[^\s]+', '[PATH]', message)
    
    # Remove stack traces in production
    if not DEBUG_MODE:
        message = message.split('\n')[0]
    
    return message
```

### 🛡️ Security Checklist

- [ ] **Input Validation**: All user inputs are validated
- [ ] **Path Safety**: File paths are checked for traversal attacks
- [ ] **SQL Safety**: All queries use parameterization
- [ ] **Error Sanitization**: Sensitive data not exposed in errors
- [ ] **Resource Limits**: Memory and file size limits enforced
- [ ] **Access Control**: Proper permission checks implemented

---

## 🎉 Recognition

**Contributors are recognized through**:
- GitHub contributors list
- Release notes acknowledgments
- Project documentation mentions

**Questions?** Open an issue on GitHub for development discussions and support!

Thank you for contributing to the MCP Code Indexer! Your help makes this tool better for the entire AI development community. 🚀
