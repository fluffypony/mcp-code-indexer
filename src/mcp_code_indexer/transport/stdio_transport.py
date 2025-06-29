"""
Stdio transport implementation for MCP Code Indexer.

Wraps the existing stdio transport functionality from the mcp library
into the transport abstraction.
"""

import asyncio
import logging
from typing import Any

from mcp.server.stdio import stdio_server

from .base import Transport

logger = logging.getLogger(__name__)


class StdioTransport(Transport):
    """
    Stdio transport implementation using MCP's built-in stdio server.
    
    This transport maintains compatibility with the existing stdio-based
    MCP protocol while fitting into the transport abstraction.
    """
    
    def __init__(self, server_instance: Any):
        """
        Initialize stdio transport.
        
        Args:
            server_instance: The MCPCodeIndexServer instance
        """
        super().__init__(server_instance)
        
    async def run(self) -> None:
        """
        Run the stdio transport server.
        
        Uses the mcp library's stdio_server context manager to handle
        JSON-RPC communication over stdin/stdout with retry logic.
        """
        self.logger.info(
            "Starting stdio transport",
            extra={
                "structured_data": {
                    "transport": "stdio",
                    "server_type": type(self.server).__name__,
                }
            },
        )
        
        max_retries = 5
        base_delay = 2.0  # seconds

        for attempt in range(max_retries + 1):
            try:
                async with stdio_server() as (read_stream, write_stream):
                    self.logger.info(
                        f"stdio_server context established (attempt {attempt + 1})"
                    )
                    initialization_options = self.server.server.create_initialization_options()
                    self.logger.debug(f"Initialization options: {initialization_options}")

                    await self.server._run_session_with_retry(
                        read_stream, write_stream, initialization_options
                    )
                    return  # Success, exit retry loop

            except KeyboardInterrupt:
                self.logger.info("Server stopped by user interrupt")
                return

            except Exception as e:
                import traceback

                # Check if this is a wrapped validation error
                error_str = str(e)
                is_validation_error = (
                    "ValidationError" in error_str
                    or "Field required" in error_str
                    or "Input should be" in error_str
                    or "pydantic_core._pydantic_core.ValidationError" in error_str
                )

                if is_validation_error:
                    self.logger.warning(
                        f"Detected validation error in session "
                        f"(attempt {attempt + 1}): Malformed client request",
                        extra={
                            "structured_data": {
                                "error_type": "ValidationError",
                                "error_message": (
                                    "Client sent malformed request "
                                    "(likely missing clientInfo)"
                                ),
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "will_retry": attempt < max_retries,
                            }
                        },
                    )

                    if attempt < max_retries:
                        delay = base_delay * (
                            2 ** min(attempt, 3)
                        )  # Cap exponential growth
                        self.logger.info(f"Retrying server in {delay} seconds...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        self.logger.warning(
                            "Max retries exceeded for validation errors. Server is "
                            "robust against malformed requests."
                        )
                        return
                else:
                    # This is a genuine fatal error
                    self.logger.error(
                        f"Fatal server error: {e}",
                        extra={
                            "structured_data": {
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "traceback": traceback.format_exc(),
                            }
                        },
                    )
                    raise
