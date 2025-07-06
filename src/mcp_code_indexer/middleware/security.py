"""
HTTP security middleware for MCP Code Indexer.

Provides security features like rate limiting, request validation,
and security headers for HTTP transport.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Any

try:
    from fastapi import HTTPException, Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as e:
    raise ImportError(
        "HTTP middleware dependencies not installed. "
        "This should not happen as they are now required dependencies. "
        "Please reinstall mcp-code-indexer."
    ) from e

logger = logging.getLogger(__name__)


class HTTPSecurityMiddleware(BaseHTTPMiddleware):
    """
    HTTP security middleware providing rate limiting and security headers.

    Implements rate limiting per client IP and adds security headers
    to all responses.
    """

    def __init__(
        self,
        app,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        enable_security_headers: bool = True,
    ):
        """
        Initialize HTTP security middleware.

        Args:
            app: FastAPI application instance
            rate_limit_requests: Number of requests allowed per window
            rate_limit_window: Time window for rate limiting in seconds
            max_request_size: Maximum request size in bytes
            enable_security_headers: Whether to add security headers
        """
        super().__init__(app)
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window = rate_limit_window
        self.max_request_size = max_request_size
        self.enable_security_headers = enable_security_headers

        # Rate limiting storage
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        self.rate_limit_lock = asyncio.Lock()

        self.logger = logger.getChild("http_security")

        self.logger.info(
            "HTTP security middleware initialized",
            extra={
                "structured_data": {
                    "rate_limit_requests": rate_limit_requests,
                    "rate_limit_window": rate_limit_window,
                    "max_request_size": max_request_size,
                    "security_headers_enabled": enable_security_headers,
                }
            },
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process HTTP request with security checks.

        Args:
            request: FastAPI request object
            call_next: Next middleware in chain

        Returns:
            HTTP response

        Raises:
            HTTPException: If security checks fail
        """
        client_ip = self._get_client_ip(request)

        # Check request size
        await self._check_request_size(request)

        # Check rate limits
        await self._check_rate_limit(client_ip, request.url.path)

        # Process request
        response = await call_next(request)

        # Add security headers
        if self.enable_security_headers:
            self._add_security_headers(response)

        return response

    async def _check_request_size(self, request: Request) -> None:
        """
        Check if request size is within limits.

        Args:
            request: FastAPI request object

        Raises:
            HTTPException: If request is too large
        """
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    self.logger.warning(
                        f"Request too large: {size} bytes",
                        extra={
                            "structured_data": {
                                "client_ip": self._get_client_ip(request),
                                "request_size": size,
                                "max_size": self.max_request_size,
                                "path": request.url.path,
                            }
                        },
                    )
                    raise HTTPException(
                        status_code=413,
                        detail=f"Request too large. Maximum size: {self.max_request_size} bytes",
                    )
            except ValueError:
                # Invalid content-length header
                pass

    async def _check_rate_limit(self, client_ip: str, path: str) -> None:
        """
        Check if client has exceeded rate limits.

        Args:
            client_ip: Client IP address
            path: Request path

        Raises:
            HTTPException: If rate limit exceeded
        """
        current_time = time.time()

        async with self.rate_limit_lock:
            # Get or create request queue for this client
            client_requests = self.request_counts[client_ip]

            # Remove old requests outside the window
            while (
                client_requests
                and client_requests[0] < current_time - self.rate_limit_window
            ):
                client_requests.popleft()

            # Check if rate limit exceeded
            if len(client_requests) >= self.rate_limit_requests:
                self.logger.warning(
                    f"Rate limit exceeded for {client_ip}",
                    extra={
                        "structured_data": {
                            "client_ip": client_ip,
                            "path": path,
                            "request_count": len(client_requests),
                            "rate_limit": self.rate_limit_requests,
                            "window": self.rate_limit_window,
                        }
                    },
                )

                # Calculate retry after
                oldest_request = client_requests[0]
                retry_after = (
                    int(oldest_request + self.rate_limit_window - current_time) + 1
                )

                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )

            # Add current request
            client_requests.append(current_time)

    def _add_security_headers(self, response: Response) -> None:
        """
        Add security headers to response.

        Args:
            response: FastAPI response object
        """
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent framing (clickjacking protection)
        response.headers["X-Frame-Options"] = "DENY"

        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy (basic)
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # Strict Transport Security (if HTTPS)
        # Note: Only add HSTS if serving over HTTPS
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

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

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """
        Get current rate limit statistics.

        Returns:
            Dictionary with rate limit statistics
        """
        current_time = time.time()
        active_clients = 0
        total_requests = 0

        for client_ip, requests in self.request_counts.items():
            # Count active requests (within window)
            active_requests = sum(
                1
                for req_time in requests
                if req_time > current_time - self.rate_limit_window
            )

            if active_requests > 0:
                active_clients += 1
                total_requests += active_requests

        return {
            "active_clients": active_clients,
            "total_active_requests": total_requests,
            "rate_limit": self.rate_limit_requests,
            "window_seconds": self.rate_limit_window,
        }

    async def cleanup_old_entries(self) -> None:
        """
        Clean up old rate limit entries.

        Should be called periodically to prevent memory leaks.
        """
        current_time = time.time()
        cutoff_time = current_time - self.rate_limit_window * 2

        async with self.rate_limit_lock:
            # Remove old entries
            clients_to_remove = []

            for client_ip, requests in self.request_counts.items():
                # Remove old requests
                while requests and requests[0] < cutoff_time:
                    requests.popleft()

                # Mark empty clients for removal
                if not requests:
                    clients_to_remove.append(client_ip)

            # Remove empty clients
            for client_ip in clients_to_remove:
                del self.request_counts[client_ip]

            self.logger.debug(
                f"Cleaned up rate limit entries: removed {len(clients_to_remove)} inactive clients"
            )


class RequestValidator:
    """
    Utility class for request validation.

    Provides methods for validating request content and format.
    """

    @staticmethod
    def validate_json_size(data: str, max_size: int = 1024 * 1024) -> bool:
        """
        Validate JSON data size.

        Args:
            data: JSON data as string
            max_size: Maximum allowed size in bytes

        Returns:
            True if valid, False otherwise
        """
        return len(data.encode("utf-8")) <= max_size

    @staticmethod
    def validate_mcp_request(data: dict) -> bool:
        """
        Validate MCP request format.

        Args:
            data: Request data dictionary

        Returns:
            True if valid MCP request, False otherwise
        """
        # Check required fields
        required_fields = ["jsonrpc", "method"]
        for field in required_fields:
            if field not in data:
                return False

        # Check JSON-RPC version
        if data.get("jsonrpc") != "2.0":
            return False

        # Validate method format
        method = data.get("method")
        if not isinstance(method, str) or not method:
            return False

        return True

    @staticmethod
    def sanitize_user_input(data: str, max_length: int = 10000) -> str:
        """
        Sanitize user input for logging and processing.

        Args:
            data: Input data to sanitize
            max_length: Maximum length to allow

        Returns:
            Sanitized input string
        """
        if not isinstance(data, str):
            data = str(data)

        # Truncate if too long
        if len(data) > max_length:
            data = data[:max_length] + "..."

        # Remove potentially dangerous characters
        # This is basic sanitization - adjust based on needs
        dangerous_chars = ["\x00", "\x08", "\x0b", "\x0c", "\x0e", "\x0f"]
        for char in dangerous_chars:
            data = data.replace(char, "")

        return data
