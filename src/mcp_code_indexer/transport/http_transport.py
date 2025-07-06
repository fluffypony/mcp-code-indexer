"""
HTTP transport implementation for MCP Code Indexer.

Provides HTTP/REST API access to MCP tools using FastAPI with
Server-Sent Events for streaming responses.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, ValidationError
    import uvicorn
except ImportError as e:
    raise ImportError(
        "HTTP transport dependencies not installed. "
        "This should not happen as they are now required dependencies. "
        "Please reinstall mcp-code-indexer."
    ) from e

from .base import Transport

# Import middleware
from ..middleware.logging import HTTPLoggingMiddleware, HTTPMetricsCollector
from ..middleware.auth import HTTPAuthMiddleware
from ..middleware.security import HTTPSecurityMiddleware

logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    """MCP JSON-RPC request model."""

    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None


class MCPResponse(BaseModel):
    """MCP JSON-RPC response model."""

    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


class HTTPTransport(Transport):
    """
    HTTP transport implementation using FastAPI.

    Provides REST API endpoints for MCP tools with optional authentication
    and Server-Sent Events for streaming responses.
    """

    def __init__(
        self,
        server_instance: Any,
        host: str = "127.0.0.1",
        port: int = 7557,
        auth_token: Optional[str] = None,
        cors_origins: Optional[List[str]] = None,
    ):
        """
        Initialize HTTP transport.

        Args:
            server_instance: The MCPCodeIndexServer instance
            host: Host to bind the server to
            port: Port to bind the server to
            auth_token: Optional Bearer token for authentication
            cors_origins: List of allowed CORS origins
        """
        super().__init__(server_instance)
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.cors_origins = cors_origins or ["*"]

        # Connection management
        self.active_connections: Dict[str, asyncio.Queue] = {}
        self.app: Optional[FastAPI] = None

        # Metrics collection
        self.metrics = HTTPMetricsCollector()

    async def initialize(self) -> None:
        """Initialize FastAPI application and routes."""
        self.app = await self._create_app()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """FastAPI lifespan context manager."""
        self.logger.info("HTTP transport starting up")
        yield
        self.logger.info("HTTP transport shutting down")
        await self.cleanup()

    async def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="MCP Code Indexer HTTP API",
            description="HTTP interface for MCP Code Indexer tools",
            version="1.0.0",
            lifespan=self._lifespan,
        )

        # Add middleware stack (in reverse order of execution)

        # Security middleware (outermost)
        app.add_middleware(HTTPSecurityMiddleware)

        # Logging middleware
        app.add_middleware(HTTPLoggingMiddleware)

        # Authentication middleware
        if self.auth_token:
            app.add_middleware(
                HTTPAuthMiddleware,
                auth_token=self.auth_token,
                public_paths=["/health", "/docs", "/openapi.json", "/metrics"],
            )

        # CORS middleware (innermost, applied first)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

        # Authentication dependency
        security = HTTPBearer(auto_error=False) if self.auth_token else None

        async def verify_token(
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        ) -> bool:
            """Verify Bearer token if authentication is enabled."""
            if not self.auth_token:
                return True

            if not credentials or credentials.credentials != self.auth_token:
                raise HTTPException(
                    status_code=401, detail="Invalid authentication token"
                )
            return True

        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "transport": "http"}

        @app.get("/metrics")
        async def get_metrics(authenticated: bool = Depends(verify_token)):
            """Get HTTP transport metrics."""
            metrics = {}

            metrics["http"] = self.metrics.get_metrics()

            # Add connection stats
            metrics["connections"] = {
                "active_sse_connections": len(self.active_connections),
                "connection_ids": list(self.active_connections.keys()),
            }

            return metrics

        @app.get("/tools")
        async def list_tools(authenticated: bool = Depends(verify_token)):
            """List available MCP tools."""
            tools = await self.server._handle_list_tools()
            return {"tools": tools}

        @app.post("/mcp", response_model=MCPResponse)
        async def handle_mcp_request(
            request: MCPRequest, authenticated: bool = Depends(verify_token)
        ):
            """Handle MCP JSON-RPC requests."""
            try:
                # Route to appropriate tool handler
                if request.method == "tools/call":
                    tool_name = request.params.get("name")
                    tool_arguments = request.params.get("arguments", {})

                    # Map tool names to handler methods
                    tool_handlers = {
                        "get_file_description": self.server._handle_get_file_description,
                        "update_file_description": self.server._handle_update_file_description,
                        "check_codebase_size": self.server._handle_check_codebase_size,
                        "find_missing_descriptions": self.server._handle_find_missing_descriptions,
                        "search_descriptions": self.server._handle_search_descriptions,
                        "get_all_descriptions": self.server._handle_get_codebase_overview,
                        "get_codebase_overview": self.server._handle_get_condensed_overview,
                        "update_codebase_overview": self.server._handle_update_codebase_overview,
                        "get_word_frequency": self.server._handle_get_word_frequency,
                        "search_codebase_overview": self.server._handle_search_codebase_overview,
                        "check_database_health": self.server._handle_check_database_health,
                    }

                    if tool_name not in tool_handlers:
                        return MCPResponse(
                            id=request.id,
                            error={
                                "code": -32601,
                                "message": f"Unknown tool: {tool_name}",
                            },
                        )

                    # Execute tool handler
                    result = await tool_handlers[tool_name](tool_arguments)

                    return MCPResponse(id=request.id, result=result)

                else:
                    return MCPResponse(
                        id=request.id,
                        error={
                            "code": -32601,
                            "message": f"Unknown method: {request.method}",
                        },
                    )

            except ValidationError as e:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32602, "message": f"Invalid params: {str(e)}"},
                )
            except Exception as e:
                self.logger.error(
                    "Tool execution failed",
                    extra={
                        "structured_data": {
                            "method": request.method,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        }
                    },
                )
                return MCPResponse(
                    id=request.id,
                    error={"code": -32603, "message": f"Internal error: {str(e)}"},
                )

        @app.get("/events/{connection_id}")
        async def server_sent_events(
            connection_id: str, authenticated: bool = Depends(verify_token)
        ):
            """Server-Sent Events endpoint for streaming responses."""

            async def event_stream():
                # Create connection queue
                queue = asyncio.Queue()
                self.active_connections[connection_id] = queue

                try:
                    while True:
                        # Wait for events from the queue
                        try:
                            event_data = await asyncio.wait_for(
                                queue.get(), timeout=30.0
                            )
                            yield f"data: {json.dumps(event_data)}\\n\\n"
                        except asyncio.TimeoutError:
                            # Send keepalive
                            yield f"data: {json.dumps({'type': 'keepalive'})}\\n\\n"

                except asyncio.CancelledError:
                    pass
                finally:
                    # Clean up connection
                    self.active_connections.pop(connection_id, None)

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        return app

    async def run(self) -> None:
        """
        Run the HTTP transport server.

        Starts the uvicorn ASGI server with the configured FastAPI application.
        """
        if not self.app:
            await self.initialize()

        self.logger.info(
            "Starting HTTP transport",
            extra={
                "structured_data": {
                    "transport": "http",
                    "host": self.host,
                    "port": self.port,
                    "auth_enabled": self.auth_token is not None,
                    "cors_origins": self.cors_origins,
                }
            },
        )

        # Configure uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True,
            loop="asyncio",
        )

        server = uvicorn.Server(config)

        try:
            await server.serve()
        except Exception as e:
            self.logger.error(
                "HTTP transport failed",
                extra={
                    "structured_data": {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                },
            )
            raise

    async def cleanup(self) -> None:
        """Clean up HTTP transport resources."""
        # Close all active SSE connections
        for connection_id, queue in self.active_connections.items():
            try:
                await queue.put({"type": "disconnect"})
            except Exception as e:
                self.logger.warning(f"Error closing connection {connection_id}: {e}")

        self.active_connections.clear()
        self.logger.info("HTTP transport cleanup completed")

    async def send_event(self, connection_id: str, event_data: Dict[str, Any]) -> bool:
        """
        Send an event to a specific SSE connection.

        Args:
            connection_id: Connection identifier
            event_data: Event data to send

        Returns:
            True if event was sent successfully, False otherwise
        """
        if connection_id not in self.active_connections:
            return False

        try:
            await self.active_connections[connection_id].put(event_data)
            return True
        except Exception as e:
            self.logger.warning(f"Error sending event to {connection_id}: {e}")
            return False
