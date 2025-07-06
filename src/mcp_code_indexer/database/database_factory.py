"""
Database factory for managing multiple database instances.

This module provides a factory that creates and manages DatabaseManager instances
for both global and local databases, handling initialization and connection pooling.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from .database import DatabaseManager
from .path_resolver import DatabasePathResolver

logger = logging.getLogger(__name__)


class DatabaseFactory:
    """
    Factory for creating and managing DatabaseManager instances.

    Maintains a cache of database managers for different database paths
    to avoid creating multiple instances for the same database.
    """

    def __init__(
        self,
        global_db_path: Path,
        pool_size: int = 3,
        retry_count: int = 5,
        timeout: float = 10.0,
        enable_wal_mode: bool = True,
        health_check_interval: float = 30.0,
        retry_min_wait: float = 0.1,
        retry_max_wait: float = 2.0,
        retry_jitter: float = 0.2,
    ):
        """
        Initialize the database factory.

        Args:
            global_db_path: Path to the global database
            pool_size: Database connection pool size
            retry_count: Maximum retry attempts for database operations
            timeout: Database operation timeout in seconds
            enable_wal_mode: Whether to enable WAL mode for SQLite
            health_check_interval: Health check interval in seconds
            retry_min_wait: Minimum wait time between retries
            retry_max_wait: Maximum wait time between retries
            retry_jitter: Maximum jitter for retry delays
        """
        self.global_db_path = global_db_path
        self.db_config = {
            "pool_size": pool_size,
            "retry_count": retry_count,
            "timeout": timeout,
            "enable_wal_mode": enable_wal_mode,
            "health_check_interval": health_check_interval,
            "retry_min_wait": retry_min_wait,
            "retry_max_wait": retry_max_wait,
            "retry_jitter": retry_jitter,
        }

        self.path_resolver = DatabasePathResolver(global_db_path)
        self._database_managers: Dict[str, DatabaseManager] = {}
        self._initialized_dbs: set = set()

    async def get_database_manager(
        self, folder_path: Optional[str] = None
    ) -> DatabaseManager:
        """
        Get a database manager for the appropriate database (local or global).

        Args:
            folder_path: Project folder path to check for local database

        Returns:
            DatabaseManager instance for the appropriate database
        """
        db_path = self.path_resolver.resolve_database_path(folder_path)
        db_key = str(db_path)

        # Return existing manager if available
        if db_key in self._database_managers:
            return self._database_managers[db_key]

        # Create new database manager
        db_manager = DatabaseManager(db_path=db_path, **self.db_config)

        # Initialize if not already done
        if db_key not in self._initialized_dbs:
            await db_manager.initialize()
            self._initialized_dbs.add(db_key)
            logger.info(f"Initialized database: {db_path}")

        self._database_managers[db_key] = db_manager
        return db_manager

    async def close_all(self) -> None:
        """Close all database managers and their connection pools."""
        for db_manager in self._database_managers.values():
            await db_manager.close_pool()
        self._database_managers.clear()
        self._initialized_dbs.clear()
        logger.info("Closed all database connections")

    def get_path_resolver(self) -> DatabasePathResolver:
        """Get the database path resolver."""
        return self.path_resolver

    def list_active_databases(self) -> Dict[str, str]:
        """
        List all active database connections.

        Returns:
            Dictionary mapping database paths to their types (global/local)
        """
        result = {}
        for db_path in self._database_managers.keys():
            if db_path == str(self.global_db_path):
                result[db_path] = "global"
            else:
                result[db_path] = "local"
        return result
