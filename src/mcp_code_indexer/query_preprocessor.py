"""
Query preprocessing module for intelligent FTS5 search.

This module provides intelligent query preprocessing for SQLite FTS5 full-text search
to enable multi-word search with case insensitive matching, whole word enforcement,
and proper handling of FTS5 operators as literal search terms.

Key features:
- Multi-word queries: "grpc proto" becomes "grpc" AND "proto" for
  order-agnostic matching
- FTS5 operator escaping: "AND OR" becomes '"AND" AND "OR"' to treat
  operators as literals
- Whole word matching: prevents partial matches by relying on proper tokenization
- Case insensitive: leverages FTS5 default behavior
- Special character handling: preserves special characters in quoted terms
"""

import re
import logging
from typing import List, Set

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """
    Preprocesses user queries for optimal FTS5 search performance.

    Handles multi-word queries, operator escaping, and special character preservation
    while maintaining BM25 ranking performance.
    """

    # FTS5 operators that need to be escaped when used as literal search terms
    FTS5_OPERATORS: Set[str] = {"AND", "OR", "NOT", "NEAR"}

    def __init__(self):
        """Initialize the query preprocessor."""
        pass

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess a user query for FTS5 search.

        Args:
            query: Raw user query string

        Returns:
            Preprocessed query string optimized for FTS5

        Examples:
            >>> preprocessor = QueryPreprocessor()
            >>> preprocessor.preprocess_query("grpc proto")
            '"grpc" AND "proto"'
            >>> preprocessor.preprocess_query("error AND handling")
            '"error" AND "AND" AND "handling"'
            >>> preprocessor.preprocess_query('config "file system"')
            '"config" AND "file system"'
        """
        if not query or not query.strip():
            return ""

        # Normalize whitespace
        query = query.strip()

        # Split into terms while preserving quoted phrases
        terms = self._split_terms(query)

        if not terms:
            return ""

        # Process each term: escape operators and add quotes
        processed_terms = []
        for term in terms:
            processed_term = self._process_term(term)
            if processed_term:  # Skip empty terms
                processed_terms.append(processed_term)

        if not processed_terms:
            return ""

        # Join with AND for multi-word matching
        result = " AND ".join(processed_terms)

        logger.debug(f"Preprocessed query: '{query}' -> '{result}'")
        return result

    def _split_terms(self, query: str) -> List[str]:
        """
        Split query into terms while preserving quoted phrases.

        Args:
            query: Input query string

        Returns:
            List of terms and quoted phrases

        Examples:
            'grpc proto' -> ['grpc', 'proto']
            'config "file system"' -> ['config', '"file system"']
            'error AND handling' -> ['error', 'AND', 'handling']
        """
        terms = []

        # Regex to match quoted phrases or individual words
        # This pattern captures:
        # 1. Double-quoted strings (including the quotes)
        # 2. Single words (sequences of non-whitespace characters)
        pattern = r'"[^"]*"|\S+'

        matches = re.findall(pattern, query)

        for match in matches:
            # Skip empty matches
            if match.strip():
                terms.append(match)

        return terms

    def _process_term(self, term: str) -> str:
        """
        Process a single term: escape operators and ensure proper quoting.

        Args:
            term: Single term or quoted phrase

        Returns:
            Processed term ready for FTS5

        Examples:
            'grpc' -> '"grpc"'
            'AND' -> '"AND"'
            '"file system"' -> '"file system"'
            'c++' -> '"c++"'
        """
        if not term:
            return ""

        # If already quoted, return as-is (user intentional phrase)
        if term.startswith('"') and term.endswith('"') and len(term) >= 2:
            return term

        # Check if term is an FTS5 operator (case-insensitive)
        if term.upper() in self.FTS5_OPERATORS:
            # Escape operator by quoting
            escaped_term = f'"{term}"'
            logger.debug(f"Escaped FTS5 operator: '{term}' -> '{escaped_term}'")
            return escaped_term

        # Quote all terms to ensure whole-word matching and handle special characters
        return f'"{term}"'

    def _escape_quotes_in_term(self, term: str) -> str:
        """
        Escape internal quotes in a term for FTS5 compatibility.

        Args:
            term: Term that may contain quotes

        Returns:
            Term with escaped quotes

        Examples:
            'say "hello"' -> 'say ""hello""'
            "test's file" -> "test's file"
        """
        # In FTS5, quotes inside quoted strings are escaped by doubling them
        return term.replace('"', '""')


def preprocess_search_query(query: str) -> str:
    """
    Convenience function for preprocessing search queries.

    Args:
        query: Raw user query

    Returns:
        Preprocessed query ready for FTS5
    """
    preprocessor = QueryPreprocessor()
    return preprocessor.preprocess_query(query)
