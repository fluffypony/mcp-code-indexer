"""
Unit tests for ASTChunker class.

Tests verify redaction behavior, chunk generation, and statistics tracking
for the AST-based code chunker in vector mode.
"""

from pathlib import Path
from typing import List, Generator

import pytest

from mcp_code_indexer.vector_mode.chunking.ast_chunker import ASTChunker, CodeChunk
from mcp_code_indexer.vector_mode.security.patterns import SecurityPatterns


@pytest.fixture
def temp_file_factory() -> Generator[callable, None, None]:
    """Factory fixture to create temporary files with automatic cleanup."""
    temp_files = []
    
    def create_temp_file(filename: str, content: str) -> Path:
        """Create a temporary file with given name and content."""
        temp_path = Path("/tmp") / filename
        temp_path.write_text(content)
        temp_files.append(temp_path)
        return temp_path
    
    yield create_temp_file
    
    # Cleanup
    for temp_path in temp_files:
        if temp_path.exists():
            temp_path.unlink()


@pytest.fixture
def security_patterns() -> SecurityPatterns:
    """Create SecurityPatterns instance for testing."""
    return SecurityPatterns()


@pytest.fixture
def test_code_with_secrets(fake_secrets) -> str:
    """Generate Python code containing various types of secrets."""
    code_lines = [
        "# Test Python file with embedded secrets",
        "import os",
        "import requests",
        "",
        "class SecretManager:",
        "    def __init__(self):",
    ]

    # Add secrets as variable assignments using dynamically generated ones
    for secret_name, secret_value in fake_secrets.items():
        code_lines.extend([
            f'        self.{secret_name} = "{secret_value}"',
        ])

    # Add connection strings from fake_secrets
    code_lines.extend([
        f'        self.postgres_url = "{fake_secrets["postgres_url"]}"',
        f'        self.redis_url = "{fake_secrets["redis_url"]}"',
    ])

    code_lines.extend(
        [
            "",
            "    def get_secret(self, name: str) -> str:",
            "        return getattr(self, name, None)",
            "",
            "def main():",
            "    manager = SecretManager()",
            "    return manager",
            "",
            'if __name__ == "__main__":',
            "    main()",
        ]
    )

    return "\n".join(code_lines)


@pytest.fixture
def test_code_without_secrets() -> str:
    """Generate Python code without any secrets."""
    return """# Test Python file without secrets
import os
import json
from typing import List, Dict, Any

class DataProcessor:
    def __init__(self):
        self.data = []
        self.config = {"max_items": 100}
    
    def process_data(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = []
        for item in items:
            if self.validate_item(item):
                processed.append(self.transform_item(item))
        return processed
    
    def validate_item(self, item: Dict[str, Any]) -> bool:
        return "id" in item and "name" in item
    
    def transform_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": item["id"],
            "name": item["name"].upper(),
            "processed": True
        }

def main():
    processor = DataProcessor()
    test_data = [
        {"id": 1, "name": "test"},
        {"id": 2, "name": "example"}
    ]
    result = processor.process_data(test_data)
    return result

if __name__ == "__main__":
    main()
"""


def test_ast_chunker_with_redaction_enabled(temp_file_factory, fake_secrets):
    """Test that ASTChunker with enable_redaction=True redacts secrets."""
    # Create ASTChunker with redaction enabled
    chunker = ASTChunker(
        enable_redaction=True,
        redaction_confidence=0.3,  # Lower confidence to catch more patterns
        max_chunk_size=1500,
        min_chunk_size=50,
    )

    # Create simple test content with clear secrets using dynamic generation
    content_lines = [
        "# Simple config with secret",
        f'API_KEY = "{fake_secrets["google_api_key"]}"',
        f'AWS_ACCESS_KEY = "{fake_secrets["aws_access_key"]}"',
        f'GITHUB_TOKEN = "{fake_secrets["github_token"]}"',
        "",
        "def get_config():",
        "    return {",
        '        "api_key": API_KEY,',
        '        "aws_key": AWS_ACCESS_KEY,',
        '        "github": GITHUB_TOKEN',
        "    }",
    ]

    simple_secret_content = "\n".join(content_lines)

    # Create temporary file with secrets using fixture
    temp_path = temp_file_factory("simple_secrets.py", simple_secret_content)

    # Chunk the file
    chunks = chunker.chunk_file(str(temp_path))

    # Verify chunks were created
    assert len(chunks) > 0, "Should create at least one chunk"
    
    # Check statistics
    stats = chunker.get_stats()
    assert stats.files_processed == 1
    assert stats.total_chunks == len(chunks)
    assert stats.redacted_chunks > 0, "Should have redacted at least one chunk"

    # Verify that at least some chunks were redacted
    redacted_chunks = [chunk for chunk in chunks if chunk.redacted]
    assert len(redacted_chunks) > 0, "Should have at least one redacted chunk"

    # Check redaction metadata
    for chunk in redacted_chunks:
        assert "redaction_count" in chunk.metadata
        assert "redacted_patterns" in chunk.metadata
        assert chunk.metadata["redaction_count"] > 0
        assert len(chunk.metadata["redacted_patterns"]) > 0

    # Verify redaction markers are present in at least some content
    all_content = " ".join([chunk.content for chunk in chunks])
    assert "[REDACTED:" in all_content, "Should contain redaction markers"


def test_ast_chunker_with_redaction_disabled(test_code_with_secrets: str, temp_file_factory, fake_secrets):
    """Test that ASTChunker with enable_redaction=False doesn't redact secrets."""
    # Create ASTChunker with redaction disabled
    chunker = ASTChunker(enable_redaction=False, max_chunk_size=1500, min_chunk_size=50)

    # Create temporary file with secrets using fixture
    temp_path = temp_file_factory("secrets_module_no_redaction.py", test_code_with_secrets)

    # Chunk the file
    chunks = chunker.chunk_file(str(temp_path))

    # Verify chunks were created
    assert len(chunks) > 0, "Should create at least one chunk"

    # Check statistics
    stats = chunker.get_stats()
    assert stats.files_processed == 1
    assert stats.total_chunks == len(chunks)
    assert stats.redacted_chunks == 0, "Should have no redacted chunks when disabled"

    # Verify that no chunks were redacted
    redacted_chunks = [chunk for chunk in chunks if chunk.redacted]
    assert len(redacted_chunks) == 0, "Should have no redacted chunks"

    # Check that redaction metadata is not present
    for chunk in chunks:
        assert "redaction_count" not in chunk.metadata
        assert "redacted_patterns" not in chunk.metadata

    # Verify that original secrets are preserved in content
    all_content = " ".join([chunk.content for chunk in chunks])

    # Check that secrets are still present (not redacted) - use dynamic values
    original_secrets = [
        fake_secrets["aws_access_key"],
        fake_secrets["github_token"], 
        fake_secrets["google_api_key"],
    ]

    secrets_found = 0
    for secret in original_secrets:
        if secret in all_content:
            secrets_found += 1

    assert secrets_found > 0, "Should preserve original secrets when redaction is disabled"

    # Verify no redaction markers are present
    assert "[REDACTED:" not in all_content, "Should not contain redaction markers"


def test_ast_chunker_with_clean_code(test_code_without_secrets: str, temp_file_factory):
    """Test ASTChunker behavior with code that has no secrets."""
    # Test with redaction enabled
    chunker_with_redaction = ASTChunker(
        enable_redaction=True,
        redaction_confidence=0.5,
        max_chunk_size=1500,
        min_chunk_size=50,
    )

    # Create temporary file with clean code using fixture
    temp_path = temp_file_factory("clean_code.py", test_code_without_secrets)

    # Chunk the file
    chunks = chunker_with_redaction.chunk_file(str(temp_path))

    # Verify chunks were created
    assert len(chunks) > 0, "Should create at least one chunk"

    # Check statistics
    stats = chunker_with_redaction.get_stats()
    assert stats.files_processed == 1
    assert stats.total_chunks == len(chunks)
    assert stats.redacted_chunks == 0, "Should have no redacted chunks for clean code"

    # Verify that no chunks were redacted
    redacted_chunks = [chunk for chunk in chunks if chunk.redacted]
    assert len(redacted_chunks) == 0, "Should have no redacted chunks"

    # Verify no redaction metadata
    for chunk in chunks:
        assert "redaction_count" not in chunk.metadata
        assert "redacted_patterns" not in chunk.metadata

    # Verify content integrity
    all_content = " ".join([chunk.content for chunk in chunks])
    assert "DataProcessor" in all_content, "Should preserve original class name"
    assert "process_data" in all_content, "Should preserve method names"
    assert "[REDACTED:" not in all_content, "Should not contain redaction markers"


def test_security_patterns_coverage(security_patterns: SecurityPatterns):
    """Test that SecurityPatterns covers expected pattern types."""
    # Get pattern summary
    pattern_summary = security_patterns.get_pattern_summary()

    # Verify expected pattern types are present
    expected_types = [
        "api_key",
        "token",
        "connection_string",
        "private_key",
        "password",
        "secret",
    ]

    for pattern_type in expected_types:
        assert pattern_type in pattern_summary, f"Missing pattern type: {pattern_type}"
        assert (
            pattern_summary[pattern_type] > 0
        ), f"No patterns for type: {pattern_type}"

    # Verify we have high-confidence patterns
    high_confidence = security_patterns.get_high_confidence_patterns(min_confidence=0.8)
    assert len(high_confidence) > 0, "Should have high-confidence patterns"

    # Verify context-sensitive patterns
    context_sensitive = security_patterns.get_context_sensitive_patterns()
    assert len(context_sensitive) > 0, "Should have context-sensitive patterns"


def test_ast_chunker_stats_tracking(temp_file_factory, fake_secrets):
    """Test that ASTChunker properly tracks statistics."""
    chunker = ASTChunker(
        enable_redaction=True,
        max_chunk_size=500,  # Smaller chunks for testing
        min_chunk_size=50,
    )

    # Test with multiple files using dynamic secrets
    test_files = {
        "file1.py": "def hello():\n    return 'world'",
        "file2.py": f"class Test:\n    def __init__(self):\n        self.api_key = '{fake_secrets['google_api_key']}'",
    }

    # Create temp files and chunk them using fixture
    for filename, content in test_files.items():
        temp_path = temp_file_factory(filename, content)
        chunks = chunker.chunk_file(str(temp_path))
        assert len(chunks) > 0

    # Check final statistics
    stats = chunker.get_stats()
    assert stats.files_processed == 2
    assert stats.total_chunks > 0
    assert stats.processing_time > 0

    # Should have some chunks by type
    assert len(stats.chunks_by_type) > 0

    # Should have python chunks
    assert "python" in stats.chunks_by_language
    assert stats.chunks_by_language["python"] > 0

    # Reset stats and verify
    chunker.reset_stats()
    new_stats = chunker.get_stats()
    assert new_stats.files_processed == 0
    assert new_stats.total_chunks == 0
    assert new_stats.redacted_chunks == 0


def test_dynamic_secret_detection(security_patterns: SecurityPatterns, fake_secrets):
    """Test dynamic secret detection using all SecurityPatterns."""
    # Get all high-confidence patterns
    high_conf_patterns = security_patterns.get_high_confidence_patterns(
        min_confidence=0.8
    )
    assert len(high_conf_patterns) > 5, "Should have multiple high-confidence patterns"

    # Test sample text with various secrets using dynamic generation
    test_text = f"""
    API_KEY = "{fake_secrets['google_api_key']}"
    AWS_ACCESS = "{fake_secrets['aws_access_key']}"
    GITHUB_TOKEN = "{fake_secrets['github_token']}"
    JWT = "{fake_secrets['jwt_token']}"
    DB_URL = "{fake_secrets['postgres_url']}"
    """

    # Find matches
    matches = security_patterns.find_matches(test_text, min_confidence=0.8)

    # Should find multiple matches
    assert len(matches) > 0, "Should detect secrets in test text"

    # Verify match details
    for match in matches:
        assert match.confidence >= 0.8
        assert match.start_pos >= 0
        assert match.end_pos > match.start_pos
        assert len(match.matched_text) > 0
        assert match.pattern_name in [p.name for p in high_conf_patterns]
