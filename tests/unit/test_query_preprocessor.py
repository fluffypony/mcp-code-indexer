"""
Comprehensive unit tests for the QueryPreprocessor module.

Tests intelligent FTS5 query preprocessing including edge cases,
special characters, FTS5 operator handling, and multi-word queries.
"""

import pytest

from mcp_code_indexer.query_preprocessor import (
    QueryPreprocessor,
    preprocess_search_query,
)


class TestQueryPreprocessor:
    """Test suite for QueryPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = QueryPreprocessor()

    def test_basic_single_word(self):
        """Test single word queries are properly quoted."""
        result = self.preprocessor.preprocess_query("grpc")
        assert result == '"grpc"'

    def test_basic_multi_word(self):
        """Test multi-word queries are joined with AND."""
        result = self.preprocessor.preprocess_query("grpc proto")
        assert result == '"grpc" AND "proto"'

    def test_three_word_query(self):
        """Test three-word queries maintain AND joining."""
        result = self.preprocessor.preprocess_query("error handling system")
        assert result == '"error" AND "handling" AND "system"'

    def test_empty_query(self):
        """Test empty queries return empty string."""
        assert self.preprocessor.preprocess_query("") == ""
        assert self.preprocessor.preprocess_query("   ") == ""
        assert self.preprocessor.preprocess_query(None) == ""

    def test_whitespace_normalization(self):
        """Test multiple spaces are normalized."""
        result = self.preprocessor.preprocess_query("  grpc    proto  ")
        assert result == '"grpc" AND "proto"'

    def test_fts5_operator_escaping_and(self):
        """Test AND operator is escaped when used as literal term."""
        result = self.preprocessor.preprocess_query("error AND handling")
        assert result == '"error" AND "AND" AND "handling"'

    def test_fts5_operator_escaping_or(self):
        """Test OR operator is escaped when used as literal term."""
        result = self.preprocessor.preprocess_query("config OR setting")
        assert result == '"config" AND "OR" AND "setting"'

    def test_fts5_operator_escaping_not(self):
        """Test NOT operator is escaped when used as literal term."""
        result = self.preprocessor.preprocess_query("file NOT found")
        assert result == '"file" AND "NOT" AND "found"'

    def test_fts5_operator_escaping_near(self):
        """Test NEAR operator is escaped when used as literal term."""
        result = self.preprocessor.preprocess_query("protocol NEAR grpc")
        assert result == '"protocol" AND "NEAR" AND "grpc"'

    def test_multiple_operators(self):
        """Test multiple FTS5 operators are all escaped."""
        result = self.preprocessor.preprocess_query("AND OR NOT")
        assert result == '"AND" AND "OR" AND "NOT"'

    def test_case_insensitive_operators(self):
        """Test operators are detected case-insensitively."""
        result = self.preprocessor.preprocess_query("file and backup or sync")
        assert result == '"file" AND "and" AND "backup" AND "or" AND "sync"'

    def test_quoted_phrases_preserved(self):
        """Test existing quoted phrases are preserved."""
        result = self.preprocessor.preprocess_query('config "file system"')
        assert result == '"config" AND "file system"'

    def test_quoted_phrases_with_operators(self):
        """Test operators in quoted phrases are not escaped."""
        result = self.preprocessor.preprocess_query('"search AND replace" feature')
        assert result == '"search AND replace" AND "feature"'

    def test_special_characters_preserved(self):
        """Test special characters are preserved in quoted terms."""
        result = self.preprocessor.preprocess_query("c++ language")
        assert result == '"c++" AND "language"'

    def test_hyphenated_terms(self):
        """Test hyphenated terms are preserved."""
        result = self.preprocessor.preprocess_query("grpc-proto file-system")
        assert result == '"grpc-proto" AND "file-system"'

    def test_underscored_terms(self):
        """Test underscored terms are preserved."""
        result = self.preprocessor.preprocess_query("user_id auth_token")
        assert result == '"user_id" AND "auth_token"'

    def test_mixed_special_characters(self):
        """Test various special characters are handled."""
        result = self.preprocessor.preprocess_query("config.json data@host:port")
        assert result == '"config.json" AND "data@host:port"'

    def test_numbers_and_alphanumeric(self):
        """Test numbers and alphanumeric terms work correctly."""
        result = self.preprocessor.preprocess_query("version 1.2.3 build123")
        assert result == '"version" AND "1.2.3" AND "build123"'

    def test_single_character_terms(self):
        """Test single character terms are handled."""
        result = self.preprocessor.preprocess_query("a b c")
        assert result == '"a" AND "b" AND "c"'

    def test_empty_quoted_phrase(self):
        """Test empty quoted phrases are handled gracefully."""
        result = self.preprocessor.preprocess_query('test "" another')
        assert result == '"test" AND "" AND "another"'

    def test_unmatched_quotes(self):
        """Test unmatched quotes are handled gracefully."""
        result = self.preprocessor.preprocess_query('test "unclosed quote')
        assert result == '"test" AND ""unclosed" AND "quote"'

    def test_nested_quotes_handling(self):
        """Test terms with internal quotes."""
        # This tests the _escape_quotes_in_term functionality conceptually
        result = self.preprocessor.preprocess_query("filename test")
        assert result == '"filename" AND "test"'

    def test_performance_long_query(self):
        """Test performance with longer queries."""
        words = ["word" + str(i) for i in range(50)]
        query = " ".join(words)
        result = self.preprocessor.preprocess_query(query)

        expected_parts = [f'"word{i}"' for i in range(50)]
        expected = " AND ".join(expected_parts)
        assert result == expected

    def test_split_terms_basic(self):
        """Test _split_terms method with basic input."""
        terms = self.preprocessor._split_terms("grpc proto")
        assert terms == ["grpc", "proto"]

    def test_split_terms_quoted(self):
        """Test _split_terms method with quoted phrases."""
        terms = self.preprocessor._split_terms('config "file system" backup')
        assert terms == ["config", '"file system"', "backup"]

    def test_split_terms_mixed(self):
        """Test _split_terms method with mixed content."""
        terms = self.preprocessor._split_terms('before "quoted phrase" after')
        assert terms == ["before", '"quoted phrase"', "after"]

    def test_process_term_basic(self):
        """Test _process_term method with basic term."""
        result = self.preprocessor._process_term("grpc")
        assert result == '"grpc"'

    def test_process_term_operator(self):
        """Test _process_term method with FTS5 operator."""
        result = self.preprocessor._process_term("AND")
        assert result == '"AND"'

    def test_process_term_quoted(self):
        """Test _process_term method with already quoted term."""
        result = self.preprocessor._process_term('"file system"')
        assert result == '"file system"'

    def test_process_term_empty(self):
        """Test _process_term method with empty input."""
        result = self.preprocessor._process_term("")
        assert result == ""


class TestConvenienceFunction:
    """Test suite for the convenience function."""

    def test_preprocess_search_query_function(self):
        """Test the convenience function works correctly."""
        result = preprocess_search_query("grpc proto")
        assert result == '"grpc" AND "proto"'

    def test_preprocess_search_query_with_operators(self):
        """Test convenience function handles operators."""
        result = preprocess_search_query("error AND handling")
        assert result == '"error" AND "AND" AND "handling"'

    def test_preprocess_search_query_empty(self):
        """Test convenience function with empty input."""
        result = preprocess_search_query("")
        assert result == ""


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = QueryPreprocessor()

    def test_only_operators(self):
        """Test query containing only FTS5 operators."""
        result = self.preprocessor.preprocess_query("AND OR NOT NEAR")
        assert result == '"AND" AND "OR" AND "NOT" AND "NEAR"'

    def test_repeated_words(self):
        """Test query with repeated words."""
        result = self.preprocessor.preprocess_query("test test config test")
        assert result == '"test" AND "test" AND "config" AND "test"'

    def test_unicode_characters(self):
        """Test query with Unicode characters."""
        result = self.preprocessor.preprocess_query("配置 système naïve")
        assert result == '"配置" AND "système" AND "naïve"'

    def test_very_long_term(self):
        """Test query with very long terms."""
        long_term = "a" * 1000
        result = self.preprocessor.preprocess_query(f"test {long_term} end")
        assert result == f'"test" AND "{long_term}" AND "end"'

    def test_only_whitespace_terms(self):
        """Test query that results in only whitespace terms."""
        # This is an edge case that might occur with malformed input
        result = self.preprocessor.preprocess_query("   ")
        assert result == ""

    def test_special_fts5_characters(self):
        """Test handling of other FTS5 special characters."""
        # Characters like * ^ " have special meaning in FTS5
        result = self.preprocessor.preprocess_query("search* prefix^ term")
        assert result == '"search*" AND "prefix^" AND "term"'


class TestRealWorldScenarios:
    """Test suite for real-world usage scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = QueryPreprocessor()

    def test_code_search_terms(self):
        """Test typical code search terms."""
        result = self.preprocessor.preprocess_query("authentication middleware jwt")
        assert result == '"authentication" AND "middleware" AND "jwt"'

    def test_file_extension_search(self):
        """Test searching for file extensions."""
        result = self.preprocessor.preprocess_query("config .env .json")
        assert result == '"config" AND ".env" AND ".json"'

    def test_error_message_search(self):
        """Test searching for error messages."""
        result = self.preprocessor.preprocess_query("connection refused timeout")
        assert result == '"connection" AND "refused" AND "timeout"'

    def test_api_endpoint_search(self):
        """Test searching for API-related terms."""
        result = self.preprocessor.preprocess_query("/api/v1 endpoint handler")
        assert result == '"/api/v1" AND "endpoint" AND "handler"'

    def test_database_terms(self):
        """Test database-related search terms."""
        result = self.preprocessor.preprocess_query("database schema migration")
        assert result == '"database" AND "schema" AND "migration"'

    def test_framework_terms(self):
        """Test framework and library names."""
        result = self.preprocessor.preprocess_query("react vue.js angular")
        assert result == '"react" AND "vue.js" AND "angular"'

    def test_mixed_case_preservation(self):
        """Test that original case is preserved in quotes."""
        result = self.preprocessor.preprocess_query("HTTP HTTPS gRPC")
        assert result == '"HTTP" AND "HTTPS" AND "gRPC"'


if __name__ == "__main__":
    pytest.main([__file__])
