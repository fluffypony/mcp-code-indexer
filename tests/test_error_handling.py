"""
Tests for error handling and logging functionality.

This module tests the comprehensive error handling system including
structured logging, MCP error responses, and async exception management.
"""

import asyncio
import json
import logging
from unittest.mock import Mock, patch
import pytest
import pytest_asyncio

from src.mcp_code_indexer.error_handler import (
    ErrorHandler, MCPError, DatabaseError, ValidationError, FileSystemError,
    ResourceError, ErrorCategory, setup_error_handling, StructuredFormatter
)
from src.mcp_code_indexer.middleware.error_middleware import ToolMiddleware, AsyncTaskManager
from src.mcp_code_indexer.logging_config import setup_logging, get_logger


class TestMCPErrors:
    """Test custom MCP error classes."""
    
    def test_mcp_error_creation(self):
        """Test creating basic MCP error."""
        error = MCPError(
            message="Test error",
            category=ErrorCategory.VALIDATION,
            code=-32602,
            details={"field": "test"}
        )
        
        assert str(error) == "Test error"
        assert error.category == ErrorCategory.VALIDATION
        assert error.code == -32602
        assert error.details["field"] == "test"
        assert error.timestamp is not None
    
    def test_database_error(self):
        """Test database-specific error."""
        error = DatabaseError("Database connection failed", details={"host": "localhost"})
        
        assert error.category == ErrorCategory.DATABASE
        assert error.code == -32603
        assert error.details["host"] == "localhost"
    
    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError("Invalid input", details={"field": "email"})
        
        assert error.category == ErrorCategory.VALIDATION
        assert error.code == -32602
        assert error.details["field"] == "email"
    
    def test_filesystem_error(self):
        """Test file system error."""
        error = FileSystemError("File not found", path="/test/path")
        
        assert error.category == ErrorCategory.FILE_SYSTEM
        assert error.details["path"] == "/test/path"
    
    def test_resource_error(self):
        """Test resource error."""
        error = ResourceError("Memory limit exceeded", details={"limit": "1GB"})
        
        assert error.category == ErrorCategory.RESOURCE
        assert error.details["limit"] == "1GB"


class TestStructuredFormatter:
    """Test the structured JSON formatter."""
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert data["module"] == "test_module"
        assert data["function"] == "test_function"
        assert data["line"] == 42
        assert "timestamp" in data
    
    def test_structured_data_formatting(self):
        """Test formatting with structured data."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Test error",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.structured_data = {
            "error_type": "TestError",
            "context": {"user_id": 123}
        }
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["error_type"] == "TestError"
        assert data["context"]["user_id"] == 123
    
    def test_exception_formatting(self):
        """Test formatting with exception info."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
            record.module = "test_module"
            record.funcName = "test_function"
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert "exception" in data
        assert "ValueError: Test exception" in data["exception"]


class TestErrorHandler:
    """Test the ErrorHandler class."""
    
    def test_error_handler_creation(self):
        """Test creating error handler."""
        logger = get_logger("test")
        handler = ErrorHandler(logger)
        
        assert handler.logger == logger
    
    def test_log_error_basic(self):
        """Test basic error logging."""
        logger = Mock()
        handler = ErrorHandler(logger)
        
        error = ValueError("Test error")
        handler.log_error(error, context={"test": "data"}, tool_name="test_tool")
        
        logger.error.assert_called_once()
        call_args = logger.error.call_args
        assert "MCP Error occurred" in call_args[0][0]
        
        structured_data = call_args[1]["extra"]["structured_data"]
        assert structured_data["error_type"] == "ValueError"
        assert structured_data["error_message"] == "Test error"
        assert structured_data["tool_name"] == "test_tool"
        assert structured_data["context"]["test"] == "data"
    
    def test_log_mcp_error(self):
        """Test logging MCP-specific error."""
        logger = Mock()
        handler = ErrorHandler(logger)
        
        error = ValidationError("Invalid data", details={"field": "email"})
        handler.log_error(error)
        
        logger.error.assert_called_once()
        call_args = logger.error.call_args
        structured_data = call_args[1]["extra"]["structured_data"]
        
        assert structured_data["category"] == "validation"
        assert structured_data["code"] == -32602
        assert structured_data["details"]["field"] == "email"
    
    def test_create_mcp_error_response(self):
        """Test creating MCP error response."""
        logger = Mock()
        handler = ErrorHandler(logger)
        
        error = DatabaseError("Connection failed")
        response = handler.create_mcp_error_response(
            error, "test_tool", {"arg1": "value1"}
        )
        
        assert response.type == "text"
        data = json.loads(response.text)
        
        assert data["error"]["code"] == -32603
        assert data["error"]["message"] == "Connection failed"
        assert data["error"]["category"] == "database"
        assert data["tool"] == "test_tool"
        assert data["arguments"]["arg1"] == "value1"
    
    def test_sanitize_arguments(self):
        """Test argument sanitization."""
        logger = Mock()
        handler = ErrorHandler(logger)
        
        arguments = {
            "safe_arg": "safe_value",
            "password": "secret123",
            "auth_token": "token123",
            "long_text": "x" * 150,
            "normal_text": "normal"
        }
        
        sanitized = handler._sanitize_arguments(arguments)
        
        assert sanitized["safe_arg"] == "safe_value"
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["auth_token"] == "[REDACTED]"
        assert sanitized["long_text"].endswith("...")
        assert len(sanitized["long_text"]) == 103  # 100 chars + "..."
        assert sanitized["normal_text"] == "normal"
    
    @pytest_asyncio.fixture
    async def test_handle_async_task_error(self):
        """Test handling async task errors."""
        logger = Mock()
        handler = ErrorHandler(logger)
        
        async def failing_task():
            raise ValueError("Task failed")
        
        task = asyncio.create_task(failing_task())
        
        try:
            await task
        except ValueError:
            pass
        
        await handler.handle_async_task_error(task, "test_task", {"context": "test"})
        
        logger.error.assert_called_once()
        call_args = logger.error.call_args
        structured_data = call_args[1]["extra"]["structured_data"]
        assert structured_data["task_name"] == "test_task"
        assert structured_data["context"]["context"] == "test"


class TestToolMiddleware:
    """Test the ToolMiddleware class."""
    
    def test_tool_middleware_creation(self):
        """Test creating tool middleware."""
        error_handler = Mock()
        middleware = ToolMiddleware(error_handler)
        
        assert middleware.error_handler == error_handler
    
    @pytest_asyncio.fixture
    async def test_wrap_tool_handler_success(self):
        """Test wrapping tool handler for successful execution."""
        error_handler = Mock()
        middleware = ToolMiddleware(error_handler)
        
        @middleware.wrap_tool_handler("test_tool")
        async def test_handler(arguments):
            return [Mock(text="success")]
        
        result = await test_handler({"arg1": "value1"})
        
        assert len(result) == 1
        assert result[0].text == "success"
    
    @pytest_asyncio.fixture
    async def test_wrap_tool_handler_error(self):
        """Test wrapping tool handler for error cases."""
        error_handler = Mock()
        error_handler.log_error = Mock()
        error_handler.create_mcp_error_response = Mock(return_value=Mock(text="error"))
        
        middleware = ToolMiddleware(error_handler)
        
        @middleware.wrap_tool_handler("test_tool")
        async def failing_handler(arguments):
            raise ValueError("Handler failed")
        
        result = await failing_handler({"arg1": "value1"})
        
        assert len(result) == 1
        error_handler.log_error.assert_called_once()
        error_handler.create_mcp_error_response.assert_called_once()
    
    @pytest_asyncio.fixture
    async def test_validate_tool_arguments(self):
        """Test argument validation decorator."""
        error_handler = Mock()
        middleware = ToolMiddleware(error_handler)
        
        @middleware.validate_tool_arguments(["required1", "required2"], ["optional1"])
        async def test_handler(arguments):
            return "success"
        
        # Test valid arguments
        result = await test_handler({
            "required1": "value1",
            "required2": "value2",
            "optional1": "optional_value"
        })
        assert result == "success"
        
        # Test missing required field
        with pytest.raises(ValidationError, match="Missing required fields"):
            await test_handler({"required1": "value1"})
        
        # Test unexpected field
        with pytest.raises(ValidationError, match="Unexpected fields"):
            await test_handler({
                "required1": "value1",
                "required2": "value2",
                "unexpected": "value"
            })


class TestAsyncTaskManager:
    """Test the AsyncTaskManager class."""
    
    def test_task_manager_creation(self):
        """Test creating async task manager."""
        error_handler = Mock()
        manager = AsyncTaskManager(error_handler)
        
        assert manager.error_handler == error_handler
        assert len(manager._tasks) == 0
    
    @pytest_asyncio.fixture
    async def test_create_task(self):
        """Test creating and managing tasks."""
        error_handler = Mock()
        manager = AsyncTaskManager(error_handler)
        
        async def test_coroutine():
            await asyncio.sleep(0.1)
            return "completed"
        
        task = manager.create_task(test_coroutine(), "test_task")
        
        assert len(manager._tasks) == 1
        assert manager.active_task_count == 1
        
        result = await task
        assert result == "completed"
        
        # Give time for cleanup
        await asyncio.sleep(0.1)
        assert len(manager._tasks) == 0
    
    @pytest_asyncio.fixture
    async def test_task_error_handling(self):
        """Test task error handling."""
        error_handler = Mock()
        error_handler.handle_async_task_error = Mock()
        manager = AsyncTaskManager(error_handler)
        
        async def failing_coroutine():
            raise ValueError("Task error")
        
        task = manager.create_task(failing_coroutine(), "failing_task")
        
        try:
            await task
        except ValueError:
            pass
        
        # Give time for error handling
        await asyncio.sleep(0.1)
        error_handler.handle_async_task_error.assert_called_once()
    
    @pytest_asyncio.fixture
    async def test_wait_for_all(self):
        """Test waiting for all tasks to complete."""
        error_handler = Mock()
        manager = AsyncTaskManager(error_handler)
        
        results = []
        
        async def test_task(value, delay):
            await asyncio.sleep(delay)
            results.append(value)
        
        manager.create_task(test_task("first", 0.1), "task1")
        manager.create_task(test_task("second", 0.2), "task2")
        
        await manager.wait_for_all()
        
        assert "first" in results
        assert "second" in results
        assert manager.active_task_count == 0
    
    @pytest_asyncio.fixture
    async def test_cancel_all(self):
        """Test canceling all tasks."""
        error_handler = Mock()
        manager = AsyncTaskManager(error_handler)
        
        async def long_task():
            await asyncio.sleep(10)  # Long delay
        
        task1 = manager.create_task(long_task(), "task1")
        task2 = manager.create_task(long_task(), "task2")
        
        assert manager.active_task_count == 2
        
        manager.cancel_all()
        
        assert len(manager._tasks) == 0
        assert task1.cancelled()
        assert task2.cancelled()


class TestLoggingConfiguration:
    """Test logging configuration."""
    
    def test_setup_logging_basic(self, tmp_path):
        """Test basic logging setup."""
        log_file = tmp_path / "test.log"
        logger = setup_logging(
            log_level="DEBUG",
            log_file=log_file,
            enable_file_logging=True
        )
        
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2  # Console + file
        
        # Test logging
        logger.info("Test message")
        
        # Check file was created
        assert log_file.exists()
        
        # Check log content
        log_content = log_file.read_text()
        assert "Test message" in log_content
    
    def test_setup_logging_console_only(self):
        """Test logging setup with console only."""
        logger = setup_logging(log_level="WARNING", enable_file_logging=False)
        
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1  # Console only
    
    def test_get_logger(self):
        """Test getting named logger."""
        logger = get_logger("test.module")
        assert logger.name == "test.module"


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""
    
    def test_setup_error_handling(self):
        """Test complete error handling setup."""
        logger = get_logger("test")
        error_handler = setup_error_handling(logger)
        
        assert isinstance(error_handler, ErrorHandler)
        assert error_handler.logger == logger
    
    @pytest_asyncio.fixture
    async def test_full_error_workflow(self):
        """Test complete error handling workflow."""
        logger = get_logger("test")
        error_handler = setup_error_handling(logger)
        middleware = ToolMiddleware(error_handler)
        
        @middleware.wrap_tool_handler("integration_test")
        async def test_tool(arguments):
            if arguments.get("should_fail"):
                raise DatabaseError("Simulated database error")
            return [Mock(text="success")]
        
        # Test successful case
        result = await test_tool({"should_fail": False})
        assert len(result) == 1
        
        # Test error case
        result = await test_tool({"should_fail": True})
        assert len(result) == 1
        # Should contain error response, not success
