"""
Base transport abstraction for MCP Code Indexer.

Provides common interface and functionality for different transport methods.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class Transport(ABC):
    """
    Abstract base class for MCP transport implementations.

    Defines the common interface that all transports must implement
    to work with the MCPCodeIndexServer.
    """

    def __init__(self, server_instance: Any):
        """
        Initialize transport with server instance.

        Args:
            server_instance: The MCPCodeIndexServer instance
        """
        self.server = server_instance
        self.logger = logger.getChild(self.__class__.__name__)

    @abstractmethod
    async def run(self) -> None:
        """
        Run the transport server.

        This method should handle the main server loop and connection
        management for the specific transport type.
        """
        pass

    async def initialize(self) -> None:
        """
        Initialize transport-specific resources.

        Called before run() to set up any transport-specific state.
        Default implementation does nothing.
        """
        pass

    async def cleanup(self) -> None:
        """
        Clean up transport-specific resources.

        Called when shutting down to clean up connections and resources.
        Default implementation does nothing.
        """
        pass

    async def _run_with_retry(
        self, max_retries: int = 3, base_delay: float = 1.0
    ) -> None:
        """
        Run transport with exponential backoff retry logic.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
        """
        retry_count = 0

        while retry_count <= max_retries:
            try:
                await self.run()
                break
            except Exception as e:
                retry_count += 1

                if retry_count > max_retries:
                    self.logger.error(
                        "Transport failed after %d attempts",
                        max_retries,
                        extra={
                            "structured_data": {
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "retry_count": retry_count,
                            }
                        },
                    )
                    raise

                delay = base_delay * (2 ** (retry_count - 1))
                self.logger.warning(
                    "Transport failed, retrying in %.1f seconds (attempt %d/%d)",
                    delay,
                    retry_count,
                    max_retries,
                    extra={
                        "structured_data": {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "retry_count": retry_count,
                            "delay": delay,
                        }
                    },
                )

                await asyncio.sleep(delay)
