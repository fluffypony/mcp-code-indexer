"""
Tests for database connection recovery and resilience scenarios.

This module tests connection recovery mechanisms, pool refresh functionality,
and edge cases in database connection management.
"""

import asyncio
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from datetime import datetime

import aiosqlite

from src.mcp_code_indexer.database.database import DatabaseManager
from src.mcp_code_indexer.database.connection_health import ConnectionHealthMonitor, DatabaseMetricsCollector
from src.mcp_code_indexer.database.retry_handler import ConnectionRecoveryManager


class TestConnectionRecovery:
    """Test connection recovery and resilience mechanisms."""
    
    @pytest.fixture
    async def temp_db_manager(self):
        """Create a temporary database manager for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        db_manager = DatabaseManager(db_path, pool_size=3)
        await db_manager.initialize()
        
        yield db_manager
        
        await db_manager.close_pool()
        if db_path.exists():
            os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_connection_pool_refresh(self, temp_db_manager):
        """Test connection pool refresh functionality."""
        # Get initial pool state
        initial_stats = temp_db_manager.get_database_stats()
        initial_pool_size = initial_stats["connection_pool"]["current_size"]
        
        # Manually trigger pool refresh
        await temp_db_manager.close_pool()
        
        # Pool should be empty after close
        post_close_stats = temp_db_manager.get_database_stats()
        assert post_close_stats["connection_pool"]["current_size"] == 0
        
        # Creating a new connection should work after refresh
        async with temp_db_manager.get_connection() as conn:
            cursor = await conn.execute("SELECT 1")
            result = await cursor.fetchone()
            assert result[0] == 1
    
    @pytest.mark.asyncio
    async def test_health_monitor_basic_functionality(self, temp_db_manager):
        """Test basic health monitoring functionality."""
        # Health monitor should be automatically started
        assert temp_db_manager._health_monitor is not None
        
        # Perform manual health check
        health_result = await temp_db_manager._health_monitor.check_health()
        
        assert health_result.is_healthy is True
        assert health_result.response_time_ms > 0
        assert health_result.error_message is None
        assert health_result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_health_monitor_with_timeout(self):
        """Test health monitor timeout handling."""
        # Create a health monitor with very short timeout
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        try:
            db_manager = DatabaseManager(db_path, pool_size=2)
            await db_manager.initialize()
            
            # Create health monitor with very short timeout
            health_monitor = ConnectionHealthMonitor(
                db_manager, 
                check_interval=1.0, 
                timeout_seconds=0.001  # Very short timeout
            )
            
            # This should timeout
            health_result = await health_monitor.check_health()
            
            assert health_result.is_healthy is False
            assert "timeout" in health_result.error_message.lower()
            
            await db_manager.close_pool()
            
        finally:
            if db_path.exists():
                os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_metrics_collector_operation_recording(self):
        """Test database metrics collection for operations."""
        collector = DatabaseMetricsCollector()
        
        # Record some operations
        collector.record_operation("create_project", 100.5, True, 3)
        collector.record_operation("create_project", 150.2, True, 3)
        collector.record_operation("create_project", 200.0, False, 3)
        collector.record_operation("batch_insert", 500.0, True, 3)
        
        # Get metrics
        metrics = collector.get_operation_metrics()
        
        # Verify create_project metrics
        create_metrics = metrics["create_project"]
        assert create_metrics["total_operations"] == 3
        assert create_metrics["successful_operations"] == 2
        assert create_metrics["failed_operations"] == 1
        assert create_metrics["min_duration_ms"] == 100.5
        assert create_metrics["max_duration_ms"] == 200.0
        assert abs(create_metrics["avg_duration_ms"] - 150.23) < 0.1
        
        # Verify batch_insert metrics
        batch_metrics = metrics["batch_insert"]
        assert batch_metrics["total_operations"] == 1
        assert batch_metrics["successful_operations"] == 1
        assert batch_metrics["avg_duration_ms"] == 500.0
    
    @pytest.mark.asyncio
    async def test_metrics_collector_locking_events(self):
        """Test locking event recording and frequency analysis."""
        collector = DatabaseMetricsCollector()
        
        # Record some locking events
        collector.record_locking_event("create_project", "database is locked")
        collector.record_locking_event("create_project", "database is locked")
        collector.record_locking_event("batch_insert", "database is busy")
        
        # Get locking frequency
        locking_stats = collector.get_locking_frequency()
        
        assert locking_stats["total_events"] == 3
        assert len(locking_stats["most_frequent_operations"]) >= 1
        
        # Check most frequent operation
        most_frequent = locking_stats["most_frequent_operations"][0]
        assert most_frequent["operation"] == "create_project"
        assert most_frequent["count"] == 2
    
    @pytest.mark.asyncio
    async def test_recovery_manager_failure_tracking(self, temp_db_manager):
        """Test connection recovery manager failure tracking."""
        recovery_manager = temp_db_manager._recovery_manager
        
        # Initially no failures
        assert recovery_manager._recovery_stats["consecutive_failures"] == 0
        
        # Simulate failures
        test_error = Exception("Test failure")
        
        # First two failures shouldn't trigger recovery
        result1 = await recovery_manager.handle_persistent_failure("test_op", test_error)
        assert result1 is False
        assert recovery_manager._recovery_stats["consecutive_failures"] == 1
        
        result2 = await recovery_manager.handle_persistent_failure("test_op", test_error)
        assert result2 is False
        assert recovery_manager._recovery_stats["consecutive_failures"] == 2
        
        # Third failure should trigger recovery
        result3 = await recovery_manager.handle_persistent_failure("test_op", test_error)
        assert result3 is True
        assert recovery_manager._recovery_stats["consecutive_failures"] == 0
        assert recovery_manager._recovery_stats["pool_refreshes"] == 1
    
    @pytest.mark.asyncio
    async def test_recovery_manager_reset_on_success(self, temp_db_manager):
        """Test that recovery manager resets failure count on success."""
        recovery_manager = temp_db_manager._recovery_manager
        
        # Simulate some failures
        test_error = Exception("Test failure")
        await recovery_manager.handle_persistent_failure("test_op", test_error)
        await recovery_manager.handle_persistent_failure("test_op", test_error)
        
        assert recovery_manager._recovery_stats["consecutive_failures"] == 2
        
        # Reset on success
        recovery_manager.reset_failure_count()
        
        assert recovery_manager._recovery_stats["consecutive_failures"] == 0
    
    @pytest.mark.asyncio
    async def test_health_monitor_metrics_tracking(self, temp_db_manager):
        """Test health monitor metrics and history tracking."""
        health_monitor = temp_db_manager._health_monitor
        
        # Perform several health checks
        for _ in range(5):
            await health_monitor.check_health()
        
        # Get health status
        status = health_monitor.get_health_status()
        
        # Verify metrics
        metrics = status["metrics"]
        assert metrics["total_checks"] >= 5
        assert metrics["successful_checks"] >= 5
        assert metrics["avg_response_time_ms"] > 0
        
        # Verify recent history
        recent_history = health_monitor.get_recent_history(3)
        assert len(recent_history) >= 3
        
        for check in recent_history:
            assert "timestamp" in check
            assert "is_healthy" in check
            assert "response_time_ms" in check
    
    @pytest.mark.asyncio
    async def test_health_monitor_failure_threshold(self):
        """Test health monitor failure threshold and pool refresh."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        try:
            db_manager = DatabaseManager(db_path, pool_size=2)
            await db_manager.initialize()
            
            # Create health monitor with low failure threshold
            health_monitor = ConnectionHealthMonitor(
                db_manager,
                check_interval=1.0,
                failure_threshold=2
            )
            
            # Simulate health check failures by corrupting database path temporarily
            original_db_path = db_manager.db_path
            
            # Force failures by using invalid path
            db_manager.db_path = Path("/invalid/path/database.db")
            
            # Perform health checks that should fail
            for _ in range(3):
                await health_monitor.check_health()
            
            # Restore original path
            db_manager.db_path = original_db_path
            
            # Verify failure tracking
            metrics = health_monitor.get_health_status()["metrics"]
            assert metrics["failed_checks"] >= 2
            
            await db_manager.close_pool()
            
        finally:
            if db_path.exists():
                os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_connection_exhaustion_recovery(self, temp_db_manager):
        """Test recovery from connection pool exhaustion."""
        # Create more connections than pool size to test exhaustion
        connections = []
        
        try:
            # Try to get more connections than pool allows
            for i in range(temp_db_manager.pool_size + 2):
                conn_ctx = temp_db_manager.get_connection()
                conn = await conn_ctx.__aenter__()
                connections.append((conn_ctx, conn))
            
            # Pool should handle this gracefully by creating new connections
            assert len(connections) == temp_db_manager.pool_size + 2
            
            # All connections should work
            for conn_ctx, conn in connections:
                cursor = await conn.execute("SELECT 1")
                result = await cursor.fetchone()
                assert result[0] == 1
                
        finally:
            # Clean up all connections
            for conn_ctx, conn in connections:
                await conn_ctx.__aexit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_database_corruption_handling(self):
        """Test handling of database corruption scenarios."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        try:
            # Create and initialize database
            db_manager = DatabaseManager(db_path, pool_size=2)
            await db_manager.initialize()
            
            # Close properly
            await db_manager.close_pool()
            
            # Corrupt the database file
            with open(db_path, 'wb') as f:
                f.write(b"corrupted data")
            
            # Try to reinitialize
            new_db_manager = DatabaseManager(db_path, pool_size=2)
            
            # This should handle corruption gracefully
            with pytest.raises((aiosqlite.DatabaseError, sqlite3.DatabaseError)):
                await new_db_manager.initialize()
            
            await new_db_manager.close_pool()
            
        finally:
            if db_path.exists():
                os.unlink(db_path)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_pool_refresh(self):
        """Test concurrent pool refresh operations."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        try:
            db_manager = DatabaseManager(db_path, pool_size=3)
            await db_manager.initialize()
            
            # Start multiple pool refresh operations concurrently
            refresh_tasks = [
                db_manager.close_pool(),
                db_manager.close_pool(),
                db_manager.close_pool()
            ]
            
            # This should not cause errors
            await asyncio.gather(*refresh_tasks, return_exceptions=True)
            
            # Database should still be functional
            async with db_manager.get_connection() as conn:
                cursor = await conn.execute("SELECT 1")
                result = await cursor.fetchone()
                assert result[0] == 1
                
        finally:
            await db_manager.close_pool()
            if db_path.exists():
                os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_health_monitor_stop_start_cycle(self):
        """Test health monitor stop/start cycle."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        
        try:
            db_manager = DatabaseManager(db_path, pool_size=2)
            await db_manager.initialize()
            
            health_monitor = db_manager._health_monitor
            
            # Monitor should be running
            assert health_monitor._is_monitoring is True
            
            # Stop monitoring
            await health_monitor.stop_monitoring()
            assert health_monitor._is_monitoring is False
            
            # Restart monitoring
            await health_monitor.start_monitoring()
            assert health_monitor._is_monitoring is True
            
            # Perform health check after restart
            health_result = await health_monitor.check_health()
            assert health_result.is_healthy is True
            
            await db_manager.close_pool()
            
        finally:
            if db_path.exists():
                os.unlink(db_path)


if __name__ == "__main__":
    # Run tests with asyncio
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
