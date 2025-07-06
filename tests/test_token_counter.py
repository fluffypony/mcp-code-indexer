"""
Tests for token counter functionality.

Verifies that tiktoken works correctly with the bundled cache file
for offline operation and token counting accuracy.
"""

import pytest
from pathlib import Path

from src.mcp_code_indexer.token_counter import TokenCounter, verify_tiktoken_setup
from src.mcp_code_indexer.database.models import FileDescription


class TestTokenCounter:
    """Test cases for TokenCounter class."""

    def test_initialization(self):
        """Test that TokenCounter initializes correctly."""
        counter = TokenCounter(token_limit=50000)
        assert counter.token_limit == 50000
        assert counter.encoder is not None

    def test_basic_token_counting(self):
        """Test basic token counting functionality."""
        counter = TokenCounter()

        # Test empty string
        assert counter.count_tokens("") == 0

        # Test known string - "Hello, world!" should be 4 tokens in cl100k_base
        hello_tokens = counter.count_tokens("Hello, world!")
        assert hello_tokens == 4

        # Test longer text
        longer_text = "This is a longer piece of text that should have more tokens."
        longer_tokens = counter.count_tokens(longer_text)
        assert longer_tokens > hello_tokens

    def test_file_description_tokens(self):
        """Test token counting for file descriptions."""
        counter = TokenCounter()

        file_desc = FileDescription(
            project_id="test-project",
            branch="main",
            file_path="src/main.py",
            description=(
                "Main entry point for the application with argument parsing and setup."
            ),
        )

        tokens = counter.count_file_description_tokens(file_desc)
        assert tokens > 0

        # Token count should include both path and description
        path_tokens = counter.count_tokens(file_desc.file_path)
        desc_tokens = counter.count_tokens(file_desc.description)

        # Should be approximately path + description + formatting
        assert tokens >= path_tokens + desc_tokens

    def test_codebase_token_calculation(self):
        """Test calculating tokens for multiple file descriptions."""
        counter = TokenCounter()

        file_descriptions = [
            FileDescription(
                project_id="test",
                branch="main",
                file_path="src/main.py",
                description="Main entry point",
            ),
            FileDescription(
                project_id="test",
                branch="main",
                file_path="src/utils.py",
                description="Utility functions",
            ),
            FileDescription(
                project_id="test",
                branch="main",
                file_path="tests/test_main.py",
                description="Tests for main module",
            ),
        ]

        total_tokens = counter.calculate_codebase_tokens(file_descriptions)
        assert total_tokens > 0

        # Should be sum of individual file tokens
        individual_sum = sum(
            counter.count_file_description_tokens(fd) for fd in file_descriptions
        )
        assert total_tokens == individual_sum

    def test_large_codebase_detection(self):
        """Test detection of large codebases."""
        # Small limit for testing
        counter = TokenCounter(token_limit=100)

        # Small codebase
        assert not counter.is_large_codebase(50)
        assert counter.get_recommendation(50) == "use_overview"

        # Large codebase
        assert counter.is_large_codebase(200)
        assert counter.get_recommendation(200) == "use_search"

        # Exactly at limit
        assert not counter.is_large_codebase(100)
        assert counter.is_large_codebase(101)

    def test_cache_key_generation(self):
        """Test cache key generation."""
        counter = TokenCounter()

        key1 = counter.generate_cache_key("project1", "main", "hash1")
        key2 = counter.generate_cache_key("project1", "main", "hash1")
        key3 = counter.generate_cache_key("project1", "main", "hash2")
        key4 = counter.generate_cache_key("project2", "main", "hash1")

        # Same inputs should produce same key
        assert key1 == key2

        # Different inputs should produce different keys
        assert key1 != key3
        assert key1 != key4

        # Keys should be reasonably short
        assert len(key1) == 16

    def test_error_handling(self):
        """Test error handling for token counting."""
        counter = TokenCounter()

        # Test with None - should not crash
        result = counter.count_tokens(None)
        assert result == 0


class TestTiktokenSetup:
    """Test tiktoken offline setup verification."""

    def test_tiktoken_verification(self):
        """Test that tiktoken setup verification works."""
        result = verify_tiktoken_setup()
        assert result is True

    def test_cache_file_exists(self):
        """Test that the required cache file exists."""
        cache_dir = (
            Path(__file__).parent.parent / "src" / "mcp_code_indexer" / "tiktoken_cache"
        )
        cache_file = cache_dir / "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"

        assert cache_dir.exists(), f"Cache directory not found: {cache_dir}"
        assert cache_file.exists(), f"Cache file not found: {cache_file}"

        # File should not be empty
        assert cache_file.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__])
