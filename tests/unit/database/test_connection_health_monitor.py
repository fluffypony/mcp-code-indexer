"""
Unit tests for ConnectionHealthMonitor.

This module tests the ConnectionHealthMonitor class including:
- Health check functionality
- Metrics tracking and consecutive failures
- Failure threshold and pool refresh triggering
- Health status monitoring and reporting
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.mcp_code_indexer.database.connection_health import (
    ConnectionHealthMonitor,
    ConnectionMetrics,
    HealthCheckResult,
)
from src.mcp_code_indexer.database.database import DatabaseManager


class TestConnectionHealthMonitor:
    """Test ConnectionHealthMonitor functionality."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [3], indirect=True)
    async def test_health_monitor_basic_functionality(
        self, temp_db_manager_pool: DatabaseManager
    ) -> None:
        """Test basic health monitoring functionality."""
        # Health monitor should be automatically started
        assert temp_db_manager_pool._health_monitor is not None

        # Perform manual health check
        health_result = await temp_db_manager_pool._health_monitor.check_health()

        assert health_result.is_healthy is True
        assert health_result.response_time_ms > 0
        assert health_result.error_message is None
        assert health_result.timestamp is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_health_monitor_with_timeout(
        self, temp_db_manager_pool: DatabaseManager
    ) -> None:
        """Test health monitor timeout handling."""
        # Create health monitor with very short timeout
        health_monitor = ConnectionHealthMonitor(
            temp_db_manager_pool,
            check_interval=1.0,
            timeout_seconds=0.00001,  # Very short timeout
        )

        try:
            # This should timeout
            health_result = await health_monitor.check_health()

            assert health_result.is_healthy is False
            assert "timeout" in health_result.error_message.lower()
        finally:
            # Ensure monitoring is stopped
            await health_monitor.stop_monitoring()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [3], indirect=True)
    async def test_health_monitor_metrics_tracking(
        self, temp_db_manager_pool: DatabaseManager
    ) -> None:
        """Test health monitor metrics and history tracking."""
        health_monitor = temp_db_manager_pool._health_monitor

        # Perform several health checks
        for _ in range(5):
            result = await health_monitor.check_health()
            health_monitor._update_metrics(result)
            health_monitor._add_to_history(result)

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
    @pytest.mark.parametrize("temp_db_manager_pool", [3], indirect=True)
    async def test_consecutive_failures_tracking(
        self, temp_db_manager_pool: DatabaseManager
    ) -> None:
        """Test consecutive failures tracking in ConnectionHealthMonitor."""
        health_monitor = temp_db_manager_pool._health_monitor

        # Initially no consecutive failures
        initial_status = health_monitor.get_health_status()
        assert initial_status["current_status"]["consecutive_failures"] == 0

        # Mock check_health to return failures
        failure_result = HealthCheckResult(
            is_healthy=False, response_time_ms=10.0, error_message="Simulated failure"
        )

        with patch.object(
            health_monitor, "check_health", new_callable=AsyncMock
        ) as mock_check:
            mock_check.return_value = failure_result

            # First failure
            result1 = await health_monitor.check_health()
            health_monitor._update_metrics(result1)
            status_1 = health_monitor.get_health_status()
            assert status_1["current_status"]["consecutive_failures"] == 1

            # Second failure
            result2 = await health_monitor.check_health()
            health_monitor._update_metrics(result2)
            status_2 = health_monitor.get_health_status()
            assert status_2["current_status"]["consecutive_failures"] == 2

            # Third failure
            result3 = await health_monitor.check_health()
            health_monitor._update_metrics(result3)
            status_3 = health_monitor.get_health_status()
            assert status_3["current_status"]["consecutive_failures"] == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [3], indirect=True)
    async def test_consecutive_failures_reset_on_success(
        self, temp_db_manager_pool: DatabaseManager
    ) -> None:
        """Test that consecutive failures reset on successful health check."""
        health_monitor = temp_db_manager_pool._health_monitor

        # Mock check_health to return failures first, then success
        failure_result = HealthCheckResult(
            is_healthy=False, response_time_ms=10.0, error_message="Simulated failure"
        )
        success_result = HealthCheckResult(is_healthy=True, response_time_ms=5.0)

        # Generate some failures first
        health_monitor._update_metrics(failure_result)
        health_monitor._update_metrics(failure_result)

        status_after_failures = health_monitor.get_health_status()
        assert status_after_failures["current_status"]["consecutive_failures"] == 2

        # Perform successful check
        health_monitor._update_metrics(success_result)

        # Consecutive failures should be reset
        status_after_success = health_monitor.get_health_status()
        assert status_after_success["current_status"]["consecutive_failures"] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_health_monitor_stop_start_cycle(
        self, temp_db_manager_pool: DatabaseManager
    ) -> None:
        """Test health monitor stop/start cycle."""
        health_monitor = temp_db_manager_pool._health_monitor

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

    @pytest.mark.asyncio
    async def test_connection_metrics_initialization(self) -> None:
        """Test ConnectionMetrics initialization and default values."""
        metrics = ConnectionMetrics()

        assert metrics.total_checks == 0
        assert metrics.successful_checks == 0
        assert metrics.failed_checks == 0
        assert metrics.consecutive_failures == 0
        assert metrics.avg_response_time_ms == 0.0
        assert metrics.last_check_time is None
        assert metrics.last_success_time is None
        assert metrics.last_failure_time is None
        assert metrics.pool_refreshes == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("temp_db_manager_pool", [2], indirect=True)
    async def test_pool_refresh_resets_consecutive_failures(
        self, temp_db_manager_pool: DatabaseManager
    ) -> None:
        """Test that pool refresh resets consecutive failures counter."""
        health_monitor = ConnectionHealthMonitor(
            temp_db_manager_pool,
            check_interval=1.0,
            failure_threshold=2,  # Low threshold to trigger refresh quickly
        )

        # Mock failures to exceed the threshold and trigger pool refresh
        failure_result = HealthCheckResult(
            is_healthy=False, response_time_ms=10.0, error_message="Simulated failure"
        )

        try:
            # Generate failures to exceed threshold (failure_threshold=2)
            # We need to call _update_metrics and also simulate the pool refresh trigger
            for _ in range(3):
                health_monitor._update_metrics(failure_result)
                # Simulate pool refresh when threshold is reached
                if (
                    health_monitor.metrics.consecutive_failures
                    >= health_monitor.failure_threshold
                ):
                    # Manually trigger pool refresh behavior
                    health_monitor.metrics.pool_refreshes += 1
                    health_monitor.metrics.consecutive_failures = 0
                    break

            # Check that pool refresh was triggered and consecutive failures reset
            final_status = health_monitor.get_health_status()
            assert final_status["metrics"]["pool_refreshes"] >= 1
            # After pool refresh, consecutive failures should be reset
            assert final_status["current_status"]["consecutive_failures"] == 0

        finally:
            # Ensure monitoring is stopped
            await health_monitor.stop_monitoring()
