"""
Database connection health monitoring and metrics collection.

This module provides proactive monitoring of database connections with automatic
pool refresh capabilities and performance metrics tracking.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a database health check."""

    is_healthy: bool
    response_time_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConnectionMetrics:
    """Metrics for database connection monitoring."""

    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    consecutive_failures: int = 0
    avg_response_time_ms: float = 0.0
    last_check_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    pool_refreshes: int = 0


class ConnectionHealthMonitor:
    """Monitors database connection health with periodic checks and metrics."""

    def __init__(
        self,
        database_manager,
        check_interval: float = 30.0,
        failure_threshold: int = 3,
        timeout_seconds: float = 5.0,
    ):
        """
        Initialize connection health monitor.

        Args:
            database_manager: DatabaseManager instance to monitor
            check_interval: Health check interval in seconds
            failure_threshold: Consecutive failures before pool refresh
            timeout_seconds: Timeout for health check queries
        """
        self.database_manager = database_manager
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds

        self.metrics = ConnectionMetrics()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        self._health_history: List[HealthCheckResult] = []
        self._max_history_size = 100

    async def start_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self._is_monitoring:
            logger.warning("Health monitoring is already running")
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(
            (
                f"Started database health monitoring with "
                f"{self.check_interval}s interval"
            ),
            extra={
                "structured_data": {
                    "health_monitoring": {
                        "action": "started",
                        "check_interval": self.check_interval,
                        "failure_threshold": self.failure_threshold,
                    }
                }
            },
        )

    async def stop_monitoring(self) -> None:
        """Stop periodic health monitoring."""
        if not self._is_monitoring:
            return

        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        logger.info("Stopped database health monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs periodic health checks."""
        while self._is_monitoring:
            try:
                # Perform health check
                health_result = await self.check_health()

                # Update metrics
                self._update_metrics(health_result)

                # Store in history
                self._add_to_history(health_result)

                # Check if pool refresh is needed
                if self.metrics.consecutive_failures >= self.failure_threshold:
                    await self._handle_persistent_failures()

                # Log periodic health status
                if self.metrics.total_checks % 10 == 0:  # Every 10 checks
                    self._log_health_summary()

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

            # Wait for next check
            await asyncio.sleep(self.check_interval)

    async def check_health(self) -> HealthCheckResult:
        """
        Perform a single health check on the database.

        Returns:
            HealthCheckResult with check status and timing
        """
        start_time = time.time()

        try:
            # Use a timeout for the health check
            async with asyncio.timeout(self.timeout_seconds):
                async with self.database_manager.get_connection() as conn:
                    # Simple query to test connectivity
                    cursor = await conn.execute("SELECT 1")
                    result = await cursor.fetchone()

                    if result and result[0] == 1:
                        response_time = (time.time() - start_time) * 1000
                        return HealthCheckResult(
                            is_healthy=True, response_time_ms=response_time
                        )
                    else:
                        return HealthCheckResult(
                            is_healthy=False,
                            response_time_ms=(time.time() - start_time) * 1000,
                            error_message="Unexpected query result",
                        )

        except asyncio.TimeoutError:
            return HealthCheckResult(
                is_healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=(f"Health check timeout after {self.timeout_seconds}s"),
            )

        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

    def _update_metrics(self, health_result: HealthCheckResult) -> None:
        """Update connection metrics based on health check result."""
        self.metrics.total_checks += 1
        self.metrics.last_check_time = health_result.timestamp

        if health_result.is_healthy:
            self.metrics.successful_checks += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = health_result.timestamp
        else:
            self.metrics.failed_checks += 1
            self.metrics.consecutive_failures += 1
            self.metrics.last_failure_time = health_result.timestamp

        # Update average response time
        if self.metrics.total_checks > 0:
            current_avg = self.metrics.avg_response_time_ms
            new_avg = (
                current_avg * (self.metrics.total_checks - 1)
                + health_result.response_time_ms
            ) / self.metrics.total_checks
            self.metrics.avg_response_time_ms = new_avg

    def _add_to_history(self, health_result: HealthCheckResult) -> None:
        """Add health check result to history, maintaining size limit."""
        self._health_history.append(health_result)

        # Trim history if it exceeds max size
        if len(self._health_history) > self._max_history_size:
            self._health_history = self._health_history[-self._max_history_size :]

    async def _handle_persistent_failures(self) -> None:
        """Handle persistent health check failures by refreshing pool."""
        logger.warning(
            (
                f"Detected {self.metrics.consecutive_failures} consecutive "
                f"failures, refreshing connection pool"
            ),
            extra={
                "structured_data": {
                    "pool_refresh": {
                        "consecutive_failures": self.metrics.consecutive_failures,
                        "failure_threshold": self.failure_threshold,
                        "action": "pool_refresh_triggered",
                    }
                }
            },
        )

        try:
            # Refresh the connection pool
            await self.database_manager.close_pool()
            self.metrics.pool_refreshes += 1
            self.metrics.consecutive_failures = 0

            # Perform immediate health check after refresh
            health_result = await self.check_health()
            if health_result.is_healthy:
                logger.info("Connection pool refresh successful, health check passed")
            else:
                logger.error(
                    f"Connection pool refresh failed, health check error: "
                    f"{health_result.error_message}"
                )

        except Exception as e:
            logger.error(f"Failed to refresh connection pool: {e}")

    def _log_health_summary(self) -> None:
        """Log a summary of health monitoring statistics."""
        success_rate = (
            (self.metrics.successful_checks / self.metrics.total_checks * 100)
            if self.metrics.total_checks > 0
            else 0
        )

        logger.info(
            (
                f"Health monitoring summary: {success_rate:.1f}% success rate "
                f"over {self.metrics.total_checks} checks"
            ),
            extra={
                "structured_data": {
                    "health_summary": {
                        "total_checks": self.metrics.total_checks,
                        "success_rate_percent": success_rate,
                        "avg_response_time_ms": self.metrics.avg_response_time_ms,
                        "consecutive_failures": self.metrics.consecutive_failures,
                        "pool_refreshes": self.metrics.pool_refreshes,
                    }
                }
            },
        )

    def get_health_status(self, include_retry_stats: bool = True) -> Dict:
        """
        Get current health status and metrics.

        Args:
            include_retry_stats: Whether to include retry executor statistics

        Returns:
            Dictionary with health status, metrics, recent history, and retry stats
        """
        # Get recent health status (last 5 checks)
        recent_checks = self._health_history[-5:] if self._health_history else []
        recent_success_rate = (
            sum(1 for check in recent_checks if check.is_healthy)
            / len(recent_checks)
            * 100
            if recent_checks
            else 0
        )

        health_status = {
            "is_monitoring": self._is_monitoring,
            "current_status": {
                "is_healthy": (recent_checks[-1].is_healthy if recent_checks else True),
                "consecutive_failures": self.metrics.consecutive_failures,
                "recent_success_rate_percent": recent_success_rate,
            },
            "metrics": {
                "total_checks": self.metrics.total_checks,
                "successful_checks": self.metrics.successful_checks,
                "failed_checks": self.metrics.failed_checks,
                "avg_response_time_ms": self.metrics.avg_response_time_ms,
                "pool_refreshes": self.metrics.pool_refreshes,
                "last_check_time": (
                    self.metrics.last_check_time.isoformat()
                    if self.metrics.last_check_time
                    else None
                ),
                "last_success_time": (
                    self.metrics.last_success_time.isoformat()
                    if self.metrics.last_success_time
                    else None
                ),
                "last_failure_time": (
                    self.metrics.last_failure_time.isoformat()
                    if self.metrics.last_failure_time
                    else None
                ),
            },
            "configuration": {
                "check_interval": self.check_interval,
                "failure_threshold": self.failure_threshold,
                "timeout_seconds": self.timeout_seconds,
            },
        }

        # Include retry executor statistics if available
        if include_retry_stats and hasattr(self.database_manager, "_retry_executor"):
            retry_executor = self.database_manager._retry_executor
            if retry_executor:
                health_status["retry_statistics"] = retry_executor.get_retry_stats()

        # Include database-level statistics if available
        if hasattr(self.database_manager, "get_database_stats"):
            try:
                db_stats = self.database_manager.get_database_stats()
                health_status["database_statistics"] = db_stats
            except Exception as e:
                logger.warning(f"Failed to get database statistics: {e}")

        return health_status

    def get_recent_history(self, count: int = 10) -> List[Dict]:
        """
        Get recent health check history.

        Args:
            count: Number of recent checks to return

        Returns:
            List of health check results as dictionaries
        """
        recent_checks = self._health_history[-count:] if self._health_history else []
        return [
            {
                "timestamp": check.timestamp.isoformat(),
                "is_healthy": check.is_healthy,
                "response_time_ms": check.response_time_ms,
                "error_message": (
                    check.error_message[:500] + "..."
                    if check.error_message and len(check.error_message) > 500
                    else check.error_message
                ),
            }
            for check in recent_checks
        ]

    def get_comprehensive_diagnostics(self) -> Dict:
        """
        Get comprehensive database health diagnostics for monitoring.

        This method provides detailed diagnostics suitable for the
        check_database_health MCP tool.

        Returns:
            Comprehensive health diagnostics including retry metrics,
            performance data, and resilience statistics
        """
        # Get base health status with retry stats
        base_status = self.get_health_status(include_retry_stats=True)

        # Add detailed performance analysis
        diagnostics = {
            **base_status,
            "performance_analysis": {
                "health_check_performance": {
                    "avg_response_time_ms": self.metrics.avg_response_time_ms,
                    "response_time_threshold_exceeded": (
                        self.metrics.avg_response_time_ms > 100
                    ),
                    "recent_performance_trend": self._get_performance_trend(),
                },
                "failure_analysis": {
                    "failure_rate_percent": (
                        (self.metrics.failed_checks / self.metrics.total_checks * 100)
                        if self.metrics.total_checks > 0
                        else 0
                    ),
                    "consecutive_failures": self.metrics.consecutive_failures,
                    "approaching_failure_threshold": (
                        self.metrics.consecutive_failures >= self.failure_threshold - 1
                    ),
                    "pool_refresh_frequency": self.metrics.pool_refreshes,
                },
            },
            "resilience_indicators": {
                "overall_health_score": self._calculate_health_score(),
                "retry_effectiveness": self._analyze_retry_effectiveness(),
                "connection_stability": self._assess_connection_stability(),
                "recommendations": self._generate_health_recommendations(),
            },
            "recent_history": self.get_recent_history(count=5),
        }

        return diagnostics

    def _get_performance_trend(self) -> str:
        """Analyze recent performance trend."""
        if len(self._health_history) < 5:
            return "insufficient_data"

        recent_times = [
            check.response_time_ms
            for check in self._health_history[-5:]
            if check.is_healthy
        ]

        if len(recent_times) < 2:
            return "insufficient_healthy_checks"

        # Simple trend analysis
        if recent_times[-1] > recent_times[0] * 1.5:
            return "degrading"
        elif recent_times[-1] < recent_times[0] * 0.7:
            return "improving"
        else:
            return "stable"

    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        if self.metrics.total_checks == 0:
            return 100.0

        # Base score from success rate
        success_rate = (
            self.metrics.successful_checks / self.metrics.total_checks
        ) * 100

        # Penalize consecutive failures
        failure_penalty = min(self.metrics.consecutive_failures * 10, 50)

        # Penalize high response times
        response_penalty = min(max(0, self.metrics.avg_response_time_ms - 50) / 10, 20)

        # Calculate final score
        score = success_rate - failure_penalty - response_penalty
        return max(0.0, min(100.0, score))

    def _analyze_retry_effectiveness(self) -> Dict:
        """Analyze retry mechanism effectiveness."""
        if not hasattr(self.database_manager, "_retry_executor"):
            return {"status": "no_retry_executor"}

        retry_executor = self.database_manager._retry_executor
        if not retry_executor:
            return {"status": "retry_executor_not_initialized"}

        retry_stats = retry_executor.get_retry_stats()

        return {
            "status": "active",
            "effectiveness_score": retry_stats.get("success_rate_percent", 0),
            "retry_frequency": retry_stats.get("retry_rate_percent", 0),
            "avg_attempts_per_operation": retry_stats.get(
                "average_attempts_per_operation", 0
            ),
            "is_effective": retry_stats.get("success_rate_percent", 0) > 85,
        }

    def _assess_connection_stability(self) -> Dict:
        """Assess connection stability."""
        stability_score = 100.0

        # Penalize pool refreshes
        if self.metrics.pool_refreshes > 0:
            stability_score -= min(self.metrics.pool_refreshes * 15, 60)

        # Penalize consecutive failures
        if self.metrics.consecutive_failures > 0:
            stability_score -= min(self.metrics.consecutive_failures * 20, 80)

        return {
            "stability_score": max(0.0, stability_score),
            "pool_refreshes": self.metrics.pool_refreshes,
            "consecutive_failures": self.metrics.consecutive_failures,
            "is_stable": stability_score > 70,
        }

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health recommendations based on current metrics."""
        recommendations = []

        # High failure rate
        if self.metrics.total_checks > 0:
            failure_rate = (
                self.metrics.failed_checks / self.metrics.total_checks
            ) * 100
            if failure_rate > 20:
                recommendations.append(
                    (
                        f"High failure rate ({failure_rate:.1f}%) - "
                        f"check database configuration"
                    )
                )

        # High response times
        if self.metrics.avg_response_time_ms > 100:
            recommendations.append(
                (
                    f"High response times "
                    f"({self.metrics.avg_response_time_ms:.1f}ms) - "
                    f"consider optimizing queries"
                )
            )

        # Approaching failure threshold
        if self.metrics.consecutive_failures >= self.failure_threshold - 1:
            recommendations.append(
                "Approaching failure threshold - pool refresh imminent"
            )

        # Frequent pool refreshes
        if self.metrics.pool_refreshes > 3:
            recommendations.append(
                (
                    "Frequent pool refreshes detected - investigate "
                    "underlying connection issues"
                )
            )

        # No recent successful checks
        if (
            self.metrics.last_success_time
            and datetime.utcnow() - self.metrics.last_success_time
            > timedelta(minutes=5)
        ):
            recommendations.append(
                (
                    "No successful health checks in last 5 minutes - "
                    "database may be unavailable"
                )
            )

        if not recommendations:
            recommendations.append("Database health is optimal")

        return recommendations


class DatabaseMetricsCollector:
    """Collects and aggregates database performance metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self._operation_metrics = {}
        self._locking_events = []
        self._max_events_history = 50

    def record_operation(
        self,
        operation_name: str,
        duration_ms: float,
        success: bool,
        connection_pool_size: int,
    ) -> None:
        """
        Record a database operation for metrics.

        Args:
            operation_name: Name of the database operation
            duration_ms: Operation duration in milliseconds
            success: Whether the operation succeeded
            connection_pool_size: Current connection pool size
        """
        if operation_name not in self._operation_metrics:
            self._operation_metrics[operation_name] = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "total_duration_ms": 0.0,
                "avg_duration_ms": 0.0,
                "min_duration_ms": float("inf"),
                "max_duration_ms": 0.0,
            }

        metrics = self._operation_metrics[operation_name]
        metrics["total_operations"] += 1
        metrics["total_duration_ms"] += duration_ms

        if success:
            metrics["successful_operations"] += 1
        else:
            metrics["failed_operations"] += 1

        # Update duration statistics
        metrics["avg_duration_ms"] = (
            metrics["total_duration_ms"] / metrics["total_operations"]
        )
        metrics["min_duration_ms"] = min(metrics["min_duration_ms"], duration_ms)
        metrics["max_duration_ms"] = max(metrics["max_duration_ms"], duration_ms)

    def record_locking_event(self, operation_name: str, error_message: str) -> None:
        """
        Record a database locking event.

        Args:
            operation_name: Name of the operation that encountered locking
            error_message: Error message from the locking event
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation_name": operation_name,
            "error_message": error_message[:1000] if error_message else None,
        }

        self._locking_events.append(event)

        # Trim history
        if len(self._locking_events) > self._max_events_history:
            self._locking_events = self._locking_events[-self._max_events_history :]

    def get_operation_metrics(self) -> Dict:
        """Get aggregated operation metrics."""
        return {
            operation: metrics.copy()
            for operation, metrics in self._operation_metrics.items()
        }

    def get_locking_frequency(self) -> Dict:
        """Get locking event frequency statistics."""
        if not self._locking_events:
            return {
                "total_events": 0,
                "events_last_hour": 0,
                "most_frequent_operations": [],
            }

        # Count events in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_events = [
            event
            for event in self._locking_events
            if datetime.fromisoformat(event["timestamp"]) > one_hour_ago
        ]

        # Count by operation
        operation_counts = {}
        for event in self._locking_events:
            op = event["operation_name"]
            operation_counts[op] = operation_counts.get(op, 0) + 1

        # Sort by frequency
        most_frequent = sorted(
            operation_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Truncate error messages to prevent massive responses
        recent_events_truncated = []
        for event in self._locking_events[-10:]:  # Last 10 events
            truncated_event = {
                "timestamp": event["timestamp"],
                "operation_name": event["operation_name"],
                "error_message": (
                    event["error_message"][:500] + "..."
                    if len(event["error_message"]) > 500
                    else event["error_message"]
                ),
            }
            recent_events_truncated.append(truncated_event)

        return {
            "total_events": len(self._locking_events),
            "events_last_hour": len(recent_events),
            "most_frequent_operations": [
                {"operation": op, "count": count} for op, count in most_frequent
            ],
            "recent_events": recent_events_truncated,
        }
