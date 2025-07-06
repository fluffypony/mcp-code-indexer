"""
HTTP authentication middleware for MCP Code Indexer.

Provides Bearer token authentication for HTTP transport.
"""

import logging
from typing import Optional

try:
    from fastapi import HTTPException, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as e:
    raise ImportError(
        "HTTP middleware dependencies not installed. "
        "This should not happen as they are now required dependencies. "
        "Please reinstall mcp-code-indexer."
    ) from e

logger = logging.getLogger(__name__)


class HTTPAuthMiddleware(BaseHTTPMiddleware):
    """
    HTTP authentication middleware using Bearer tokens.

    Validates Bearer tokens for protected endpoints while allowing
    public endpoints to pass through.
    """

    def __init__(
        self,
        app,
        auth_token: Optional[str] = None,
        public_paths: Optional[list] = None,
    ):
        """
        Initialize HTTP authentication middleware.

        Args:
            app: FastAPI application instance
            auth_token: Expected Bearer token for authentication
            public_paths: List of paths that don't require authentication
        """
        super().__init__(app)
        self.auth_token = auth_token
        self.public_paths = public_paths or ["/health", "/docs", "/openapi.json"]
        self.logger = logger.getChild("http_auth")

        # Only enable auth if token is provided
        self.auth_enabled = auth_token is not None

        if self.auth_enabled:
            self.logger.info("HTTP authentication enabled")
        else:
            self.logger.info("HTTP authentication disabled")

    async def dispatch(self, request: Request, call_next):
        """
        Process HTTP request and validate authentication.

        Args:
            request: FastAPI request object
            call_next: Next middleware in chain

        Returns:
            HTTP response

        Raises:
            HTTPException: If authentication fails
        """
        # Skip authentication for public paths
        if not self.auth_enabled or request.url.path in self.public_paths:
            return await call_next(request)

        # Extract Authorization header
        auth_header = request.headers.get("authorization")

        if not auth_header:
            self.logger.warning(
                f"Missing authorization header for {request.url.path}",
                extra={
                    "structured_data": {
                        "path": request.url.path,
                        "client_ip": self._get_client_ip(request),
                        "user_agent": request.headers.get("user-agent", ""),
                    }
                },
            )
            raise HTTPException(
                status_code=401,
                detail="Authorization header required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate Bearer token format
        if not auth_header.startswith("Bearer "):
            self.logger.warning(
                f"Invalid authorization format for {request.url.path}",
                extra={
                    "structured_data": {
                        "path": request.url.path,
                        "auth_format": auth_header.split(" ")[0]
                        if " " in auth_header
                        else auth_header,
                        "client_ip": self._get_client_ip(request),
                    }
                },
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization format. Use: Bearer <token>",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Extract and validate token
        token = auth_header[7:]  # Remove "Bearer " prefix

        if token != self.auth_token:
            self.logger.warning(
                f"Invalid token for {request.url.path}",
                extra={
                    "structured_data": {
                        "path": request.url.path,
                        "token_prefix": token[:8] + "..." if len(token) > 8 else token,
                        "client_ip": self._get_client_ip(request),
                    }
                },
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Token is valid, proceed with request
        self.logger.debug(
            f"Authentication successful for {request.url.path}",
            extra={
                "structured_data": {
                    "path": request.url.path,
                    "client_ip": self._get_client_ip(request),
                }
            },
        )

        return await call_next(request)

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


class TokenValidator:
    """
    Utility class for token validation logic.

    Provides methods for validating different token formats
    and managing token-based authentication.
    """

    @staticmethod
    def validate_bearer_token(token: str, expected_token: str) -> bool:
        """
        Validate Bearer token against expected value.

        Args:
            token: Token to validate
            expected_token: Expected token value

        Returns:
            True if token is valid, False otherwise
        """
        if not token or not expected_token:
            return False

        # Simple string comparison for now
        # In production, consider using constant-time comparison
        return token == expected_token

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate a random token for authentication.

        Args:
            length: Length of token to generate

        Returns:
            Random token string
        """
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def mask_token(token: str, visible_chars: int = 8) -> str:
        """
        Mask token for logging purposes.

        Args:
            token: Token to mask
            visible_chars: Number of characters to show

        Returns:
            Masked token string
        """
        if not token:
            return ""

        if len(token) <= visible_chars:
            return "*" * len(token)

        return token[:visible_chars] + "..." + "*" * (len(token) - visible_chars)
