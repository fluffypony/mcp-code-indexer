"""
HTTP logging middleware for MCP Code Indexer.

Provides request/response logging and monitoring for HTTP transport.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

try:
    from fastapi import Request, Response
    from fastapi.responses import StreamingResponse
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as e:
    raise ImportError(
        "HTTP middleware dependencies not installed. "
        "This should not happen as they are now required dependencies. "
        "Please reinstall mcp-code-indexer."
    ) from e

logger = logging.getLogger(__name__)


class HTTPLoggingMiddleware(BaseHTTPMiddleware):
    """
    HTTP request/response logging middleware.
    
    Logs HTTP requests and responses with performance metrics
    and structured data for monitoring.
    """
    
    def __init__(self, app: Any, log_level: str = "INFO"):
        """
        Initialize HTTP logging middleware.
        
        Args:
            app: FastAPI application instance
            log_level: Logging level for HTTP requests
        """
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper())
        self.logger = logger.getChild("http_access")
        self.logger.setLevel(self.log_level)
        
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process HTTP request and log access information.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        request_id = self._generate_request_id(request)
        
        # Extract client information
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        # Log incoming request
        self.logger.info(
            f"HTTP {request.method} {request.url.path}",
            extra={
                "structured_data": {
                    "event_type": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                    "content_length": request.headers.get("content-length"),
                    "content_type": request.headers.get("content-type"),
                }
            },
        )
        
        # Add request ID to request state for use in handlers
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            self._log_response(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                process_time=process_time,
                response=response,
            )
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            
            self.logger.error(
                f"HTTP {request.method} {request.url.path} - ERROR",
                extra={
                    "structured_data": {
                        "event_type": "http_error",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "process_time": process_time,
                    }
                },
            )
            
            raise
    
    def _generate_request_id(self, request: Request) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
            
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
            
        # Fall back to direct client IP
        if hasattr(request.client, "host"):
            return request.client.host
            
        return "unknown"
    
    def _log_response(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        process_time: float,
        response: Response,
    ) -> None:
        """Log HTTP response information."""
        # Determine log level based on status code
        if status_code >= 500:
            log_level = logging.ERROR
        elif status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = self.log_level
        
        # Extract response information
        content_length = response.headers.get("content-length")
        content_type = response.headers.get("content-type")
        
        # Check if this is a streaming response
        is_streaming = isinstance(response, StreamingResponse)
        
        self.logger.log(
            log_level,
            f"HTTP {method} {path} - {status_code}",
            extra={
                "structured_data": {
                    "event_type": "http_response",
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "process_time": process_time,
                    "content_length": content_length,
                    "content_type": content_type,
                    "is_streaming": is_streaming,
                    "performance_category": self._categorize_performance(process_time),
                }
            },
        )
    
    def _categorize_performance(self, process_time: float) -> str:
        """Categorize request performance."""
        if process_time < 0.1:
            return "fast"
        elif process_time < 1.0:
            return "normal"
        elif process_time < 5.0:
            return "slow"
        else:
            return "very_slow"


class HTTPMetricsCollector:
    """
    Collect HTTP metrics for monitoring.
    
    Tracks request counts, response times, and error rates.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.response_times = []
        self.max_response_times = 1000  # Keep last 1000 response times
        
    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time: float,
    ) -> None:
        """Record HTTP request metrics."""
        self.request_count += 1
        self.total_response_time += response_time
        
        # Track error rates
        if status_code >= 400:
            self.error_count += 1
            
        # Track response times
        self.response_times.append(response_time)
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        if not self.request_count:
            return {
                "request_count": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "avg_response_time": 0.0,
                "p95_response_time": 0.0,
                "p99_response_time": 0.0,
            }
        
        # Calculate percentiles
        sorted_times = sorted(self.response_times)
        p95_index = int(0.95 * len(sorted_times))
        p99_index = int(0.99 * len(sorted_times))
        
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count,
            "avg_response_time": self.total_response_time / self.request_count,
            "p95_response_time": sorted_times[p95_index] if sorted_times else 0.0,
            "p99_response_time": sorted_times[p99_index] if sorted_times else 0.0,
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.response_times.clear()
