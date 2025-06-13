"""
Tests for database error classification and structured exception handling.

This module tests the custom exception hierarchy and error classification
logic for SQLite database operations.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

import aiosqlite

from src.mcp_code_indexer.database.exceptions import (
    DatabaseError, DatabaseLockError, DatabaseBusyError, DatabaseConnectionError,
    DatabaseSchemaError, DatabaseIntegrityError, DatabaseTimeoutError,
    classify_sqlite_error, is_retryable_error, get_error_classification_stats
)


class TestDatabaseError:
    """Test the base DatabaseError class."""
    
    def test_basic_creation(self):
        """Test basic error creation."""
        error = DatabaseError("Test error message")
        assert error.message == "Test error message"
        assert error.operation_name == ""
        assert error.error_context == {}
        assert isinstance(error.timestamp, datetime)
        assert str(error) == "Test error message"
    
    def test_creation_with_operation(self):
        """Test error creation with operation name."""
        error = DatabaseError("Test error", operation_name="test_op")
        assert str(error) == "test_op: Test error"
    
    def test_creation_with_context(self):
        """Test error creation with context."""
        context = {"key": "value", "number": 42}
        error = DatabaseError("Test error", error_context=context)
        assert error.error_context == context
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = DatabaseError(
            "Test message",
            operation_name="test_operation",
            error_context={"test": "data"}
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "DatabaseError"
        assert error_dict["message"] == "Test message"
        assert error_dict["operation_name"] == "test_operation"
        assert error_dict["error_context"] == {"test": "data"}
        assert "timestamp" in error_dict


class TestDatabaseLockError:
    """Test the DatabaseLockError class."""
    
    def test_lock_error_creation(self):
        """Test lock error creation with retry context."""
        error = DatabaseLockError(
            "Database is locked",
            retry_count=3,
            operation_name="write_operation",
            lock_type="write"
        )
        
        assert error.retry_count == 3
        assert error.lock_type == "write"
        assert isinstance(error.last_attempt, datetime)
        assert error.error_context["retryable"] is True
        assert error.error_context["lock_type"] == "write"
    
    def test_lock_error_defaults(self):
        """Test lock error with default values."""
        error = DatabaseLockError("Lock error")
        
        assert error.retry_count == 0
        assert error.lock_type == "unknown"
        assert error.error_context["retryable"] is True


class TestDatabaseBusyError:
    """Test the DatabaseBusyError class."""
    
    def test_busy_error_creation(self):
        """Test busy error creation."""
        error = DatabaseBusyError(
            "Database is busy",
            operation_name="read_operation",
            busy_timeout=5.0,
            resource_type="connection"
        )
        
        assert error.busy_timeout == 5.0
        assert error.resource_type == "connection"
        assert error.error_context["retryable"] is True
        assert error.error_context["busy_timeout"] == 5.0


class TestDatabaseConnectionError:
    """Test the DatabaseConnectionError class."""
    
    def test_connection_error_creation(self):
        """Test connection error creation."""
        connection_info = {"host": "localhost", "port": 5432}
        error = DatabaseConnectionError(
            "Connection failed",
            operation_name="connect",
            connection_info=connection_info
        )
        
        assert error.connection_info == connection_info
        assert error.error_context["retryable"] is False
        assert error.error_context["connection_info"] == connection_info


class TestDatabaseSchemaError:
    """Test the DatabaseSchemaError class."""
    
    def test_schema_error_creation(self):
        """Test schema error creation."""
        migration_info = {"version": "1.2.3", "migration": "add_index"}
        error = DatabaseSchemaError(
            "Schema mismatch",
            operation_name="migrate",
            schema_version="1.2.0",
            migration_info=migration_info
        )
        
        assert error.schema_version == "1.2.0"
        assert error.migration_info == migration_info
        assert error.error_context["retryable"] is False


class TestDatabaseIntegrityError:
    """Test the DatabaseIntegrityError class."""
    
    def test_integrity_error_creation(self):
        """Test integrity error creation."""
        error = DatabaseIntegrityError(
            "Unique constraint failed",
            operation_name="insert",
            constraint_type="unique",
            affected_table="users"
        )
        
        assert error.constraint_type == "unique"
        assert error.affected_table == "users"
        assert error.error_context["retryable"] is False
        assert error.error_context["constraint_type"] == "unique"


class TestDatabaseTimeoutError:
    """Test the DatabaseTimeoutError class."""
    
    def test_timeout_error_creation(self):
        """Test timeout error creation."""
        error = DatabaseTimeoutError(
            "Operation timed out",
            operation_name="long_query",
            timeout_seconds=30.0,
            operation_type="read"
        )
        
        assert error.timeout_seconds == 30.0
        assert error.operation_type == "read"
        assert error.error_context["retryable"] is True


class TestSQLiteErrorClassification:
    """Test the SQLite error classification logic."""
    
    def test_classify_lock_errors(self):
        """Test classification of lock errors."""
        # Database locked error
        raw_error = aiosqlite.OperationalError("database is locked")
        classified = classify_sqlite_error(raw_error, "test_operation")
        
        assert isinstance(classified, DatabaseLockError)
        assert classified.operation_name == "test_operation"
        assert classified.lock_type == "read"
        
        # Write lock error
        readonly_error = aiosqlite.OperationalError("attempt to write a readonly database")
        classified = classify_sqlite_error(readonly_error, "write_op")
        
        assert isinstance(classified, DatabaseLockError)
        assert classified.lock_type == "write"
    
    def test_classify_busy_errors(self):
        """Test classification of busy errors."""
        # Database busy error
        busy_error = aiosqlite.OperationalError("database is busy")
        classified = classify_sqlite_error(busy_error, "busy_operation")
        
        assert isinstance(classified, DatabaseBusyError)
        assert classified.resource_type == "connection"
        
        # Transaction error
        transaction_error = aiosqlite.OperationalError("cannot start a transaction within a transaction")
        classified = classify_sqlite_error(transaction_error, "transaction_op")
        
        assert isinstance(classified, DatabaseBusyError)
        assert classified.resource_type == "transaction"
    
    def test_classify_connection_errors(self):
        """Test classification of connection errors."""
        connection_errors = [
            "unable to open database",
            "disk i/o error",
            "database disk image is malformed",
            "no such database"
        ]
        
        for error_msg in connection_errors:
            raw_error = aiosqlite.OperationalError(error_msg)
            classified = classify_sqlite_error(raw_error, "connect_op")
            
            assert isinstance(classified, DatabaseConnectionError)
            assert classified.operation_name == "connect_op"
    
    def test_classify_schema_errors(self):
        """Test classification of schema errors."""
        schema_errors = [
            "no such table: users",
            "no such column: email",
            "table users already exists",
            "syntax error near 'SELECT'"
        ]
        
        for error_msg in schema_errors:
            raw_error = aiosqlite.OperationalError(error_msg)
            classified = classify_sqlite_error(raw_error, "schema_op")
            
            assert isinstance(classified, DatabaseSchemaError)
            assert classified.operation_name == "schema_op"
    
    def test_classify_integrity_errors(self):
        """Test classification of integrity constraint errors."""
        constraint_tests = [
            ("UNIQUE constraint failed: users.email", "unique"),
            ("FOREIGN KEY constraint failed", "foreign_key"),
            ("PRIMARY KEY constraint failed", "primary_key"),
            ("CHECK constraint failed: age_check", "check")
        ]
        
        for error_msg, expected_type in constraint_tests:
            raw_error = aiosqlite.OperationalError(error_msg)
            classified = classify_sqlite_error(raw_error, "constraint_op")
            
            assert isinstance(classified, DatabaseIntegrityError)
            assert classified.constraint_type == expected_type
            assert classified.operation_name == "constraint_op"
    
    def test_classify_unknown_error(self):
        """Test classification of unknown errors."""
        unknown_error = aiosqlite.OperationalError("some unknown error")
        classified = classify_sqlite_error(unknown_error, "unknown_op")
        
        assert isinstance(classified, DatabaseError)
        assert not isinstance(classified, (DatabaseLockError, DatabaseBusyError))
        assert classified.operation_name == "unknown_op"
        assert classified.error_context["original_error_type"] == "OperationalError"


class TestRetryabilityDetection:
    """Test the retryability detection logic."""
    
    def test_retryable_database_errors(self):
        """Test detection of retryable database errors."""
        retryable_errors = [
            DatabaseLockError("locked"),
            DatabaseBusyError("busy"), 
            DatabaseTimeoutError("timeout")
        ]
        
        for error in retryable_errors:
            assert is_retryable_error(error) is True
    
    def test_non_retryable_database_errors(self):
        """Test detection of non-retryable database errors."""
        non_retryable_errors = [
            DatabaseConnectionError("connection failed"),
            DatabaseSchemaError("schema error"),
            DatabaseIntegrityError("constraint failed"),
            DatabaseError("generic error")  # Base class defaults to non-retryable
        ]
        
        for error in non_retryable_errors:
            assert is_retryable_error(error) is False
    
    def test_retryable_raw_exceptions(self):
        """Test retryability detection for raw exceptions."""
        retryable_messages = [
            "database is locked",
            "database is busy",
            "sqlite_busy",
            "sqlite_locked",
            "cannot start a transaction within a transaction"
        ]
        
        for message in retryable_messages:
            error = aiosqlite.OperationalError(message)
            assert is_retryable_error(error) is True
    
    def test_non_retryable_raw_exceptions(self):
        """Test non-retryability detection for raw exceptions."""
        non_retryable_messages = [
            "syntax error",
            "no such table",
            "constraint failed",
            "disk i/o error"
        ]
        
        for message in non_retryable_messages:
            error = aiosqlite.OperationalError(message)
            assert is_retryable_error(error) is False


class TestErrorStatistics:
    """Test error classification statistics."""
    
    def test_error_classification_stats(self):
        """Test error classification statistics generation."""
        errors = [
            aiosqlite.OperationalError("database is locked"),
            aiosqlite.OperationalError("database is locked"),  # Duplicate
            aiosqlite.OperationalError("database is busy"),
            aiosqlite.OperationalError("syntax error"),
            aiosqlite.OperationalError("UNIQUE constraint failed"),
        ]
        
        stats = get_error_classification_stats(errors)
        
        assert stats["total_errors"] == 5
        assert stats["error_types"]["DatabaseLockError"] == 2
        assert stats["error_types"]["DatabaseBusyError"] == 1
        assert stats["error_types"]["DatabaseSchemaError"] == 1
        assert stats["error_types"]["DatabaseIntegrityError"] == 1
        assert stats["retryable_count"] == 3  # 2 lock + 1 busy
        assert stats["non_retryable_count"] == 2  # 1 schema + 1 integrity
        
        # Check most common errors
        most_common = stats["most_common_errors"]
        assert len(most_common) > 0
        assert most_common[0][0] == "database is locked"  # Most frequent
        assert most_common[0][1] == 2  # Frequency
    
    def test_empty_error_list(self):
        """Test statistics with empty error list."""
        stats = get_error_classification_stats([])
        
        assert stats["total_errors"] == 0
        assert stats["error_types"] == {}
        assert stats["retryable_count"] == 0
        assert stats["non_retryable_count"] == 0
        assert stats["most_common_errors"] == []
    
    def test_mixed_error_types(self):
        """Test statistics with mixed error types."""
        errors = [
            DatabaseLockError("lock error"),  # Already classified
            aiosqlite.OperationalError("database is busy"),  # Will be classified
            Exception("generic error"),  # Non-database error
        ]
        
        stats = get_error_classification_stats(errors)
        
        assert stats["total_errors"] == 3
        assert stats["retryable_count"] == 2  # Lock + busy
        assert stats["non_retryable_count"] == 1  # Generic
