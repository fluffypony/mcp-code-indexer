"""
Database migration utilities for upgrading existing databases.

This module provides utilities for migrating databases to new schemas
and ensuring backward compatibility during upgrades.
"""

import asyncio
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)


class DeprecationWarning(UserWarning):
    """Custom deprecation warning for database methods."""
    pass


def deprecated(reason: str):
    """
    Decorator to mark methods as deprecated.
    
    Args:
        reason: Explanation of what to use instead
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


async def check_database_version(db_path: Path) -> Optional[str]:
    """
    Check the current database schema version.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Schema version string or None if not found
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            # Check if schema_version table exists
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            table_exists = await cursor.fetchone()
            
            if not table_exists:
                return None
            
            # Get current version
            cursor = await db.execute("SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1")
            row = await cursor.fetchone()
            
            return row[0] if row else None
            
    except Exception as e:
        logger.warning(f"Could not check database version: {e}")
        return None


async def ensure_schema_version_table(db_path: Path) -> None:
    """
    Ensure the schema_version table exists for tracking migrations.
    
    Args:
        db_path: Path to SQLite database
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                description TEXT,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def record_schema_version(db_path: Path, version: str, description: str = "") -> None:
    """
    Record a new schema version in the database.
    
    Args:
        db_path: Path to SQLite database
        version: Version string
        description: Description of changes
    """
    await ensure_schema_version_table(db_path)
    
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO schema_version (version, description) VALUES (?, ?)",
            (version, description)
        )
        await db.commit()
        
    logger.info(f"Recorded schema version {version}: {description}")


async def migrate_to_retry_executor_schema(db_path: Path) -> bool:
    """
    Migrate database to support the new retry executor schema.
    
    This migration ensures the database is compatible with the enhanced
    retry executor and adds any necessary schema updates.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        True if migration was applied, False if already up to date
    """
    current_version = await check_database_version(db_path)
    target_version = "2.0.0-retry-executor"
    
    if current_version == target_version:
        logger.info("Database already at target version for retry executor")
        return False
    
    logger.info(f"Migrating database from version {current_version} to {target_version}")
    
    try:
        async with aiosqlite.connect(db_path) as db:
            # Enable WAL mode for better concurrent access
            await db.execute("PRAGMA journal_mode=WAL")
            
            # Optimize database settings for retry executor
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA cache_size=-64000")  # 64MB cache
            await db.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
            await db.execute("PRAGMA temp_store=MEMORY")
            
            # Add any new indexes that might help with concurrent access
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_projects_remote_origin 
                ON projects(remote_origin)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_descriptions_branch_path 
                ON file_descriptions(project_name, branch, file_path)
            """)
            
            await db.commit()
            
        # Record the migration
        await record_schema_version(
            db_path, 
            target_version, 
            "Enhanced retry executor compatibility and concurrent access optimizations"
        )
        
        logger.info("Successfully migrated database for retry executor")
        return True
        
    except Exception as e:
        logger.error(f"Failed to migrate database: {e}")
        raise


async def validate_database_health(db_path: Path) -> Dict[str, Any]:
    """
    Validate database health and return diagnostic information.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dictionary with health information
    """
    health_info = {
        "database_exists": db_path.exists(),
        "readable": False,
        "writable": False,
        "wal_mode": False,
        "schema_version": None,
        "table_count": 0,
        "integrity_check": "unknown"
    }
    
    if not health_info["database_exists"]:
        return health_info
    
    try:
        async with aiosqlite.connect(db_path) as db:
            health_info["readable"] = True
            
            # Check WAL mode
            cursor = await db.execute("PRAGMA journal_mode")
            journal_mode = await cursor.fetchone()
            health_info["wal_mode"] = journal_mode and journal_mode[0].lower() == "wal"
            
            # Get schema version
            health_info["schema_version"] = await check_database_version(db_path)
            
            # Count tables
            cursor = await db.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            )
            count_row = await cursor.fetchone()
            health_info["table_count"] = count_row[0] if count_row else 0
            
            # Test write capability
            try:
                await db.execute("CREATE TEMP TABLE test_write (id INTEGER)")
                await db.execute("DROP TABLE test_write")
                health_info["writable"] = True
            except Exception:
                health_info["writable"] = False
            
            # Basic integrity check
            try:
                cursor = await db.execute("PRAGMA quick_check")
                result = await cursor.fetchone()
                health_info["integrity_check"] = result[0] if result else "unknown"
            except Exception as e:
                health_info["integrity_check"] = f"failed: {e}"
                
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        health_info["error"] = str(e)
    
    return health_info


async def cleanup_deprecated_retry_data(db_path: Path) -> Dict[str, int]:
    """
    Clean up any deprecated retry-related data or temporary tables.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dictionary with cleanup statistics
    """
    cleanup_stats = {
        "temp_tables_removed": 0,
        "old_indexes_removed": 0,
        "orphaned_records_removed": 0
    }
    
    try:
        async with aiosqlite.connect(db_path) as db:
            # Remove any temporary retry-related tables that might exist
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'retry_%_temp'"
            )
            temp_tables = await cursor.fetchall()
            
            for table in temp_tables:
                table_name = table[0]
                await db.execute(f"DROP TABLE IF EXISTS {table_name}")
                cleanup_stats["temp_tables_removed"] += 1
                logger.info(f"Removed deprecated temporary table: {table_name}")
            
            # Remove old retry statistics tables if they exist
            old_tables = ["retry_statistics", "connection_metrics", "old_retry_log"]
            for table_name in old_tables:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                if await cursor.fetchone():
                    await db.execute(f"DROP TABLE {table_name}")
                    cleanup_stats["temp_tables_removed"] += 1
                    logger.info(f"Removed deprecated table: {table_name}")
            
            await db.commit()
            
    except Exception as e:
        logger.warning(f"Cleanup of deprecated retry data failed: {e}")
    
    return cleanup_stats


class DatabaseMigrationManager:
    """Manages database migrations and compatibility."""
    
    def __init__(self, db_path: Path):
        """
        Initialize migration manager.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.current_version = None
        
    async def check_and_migrate(self) -> Dict[str, Any]:
        """
        Check database version and perform necessary migrations.
        
        Returns:
            Dictionary with migration results
        """
        migration_results = {
            "current_version": await check_database_version(self.db_path),
            "migrations_applied": [],
            "health_check": await validate_database_health(self.db_path),
            "cleanup_performed": False
        }
        
        # Apply retry executor migration if needed
        if await migrate_to_retry_executor_schema(self.db_path):
            migration_results["migrations_applied"].append("retry-executor-compatibility")
        
        # Perform cleanup of deprecated data
        cleanup_stats = await cleanup_deprecated_retry_data(self.db_path)
        if sum(cleanup_stats.values()) > 0:
            migration_results["cleanup_performed"] = True
            migration_results["cleanup_stats"] = cleanup_stats
        
        migration_results["final_version"] = await check_database_version(self.db_path)
        
        return migration_results
    
    async def force_migration(self) -> bool:
        """
        Force migration to latest schema version.
        
        Returns:
            True if successful
        """
        try:
            await migrate_to_retry_executor_schema(self.db_path)
            await cleanup_deprecated_retry_data(self.db_path)
            return True
        except Exception as e:
            logger.error(f"Forced migration failed: {e}")
            return False


# Deprecated wrapper functions for backward compatibility
@deprecated("Use DatabaseManager.get_write_connection_with_retry() instead")
async def legacy_get_write_connection(db_manager, operation_name: str = "legacy_operation"):
    """
    Legacy wrapper for write connection access.
    
    This function is deprecated and provided only for backward compatibility.
    Use DatabaseManager.get_write_connection_with_retry() instead.
    """
    async with db_manager.get_write_connection_with_retry(operation_name) as conn:
        yield conn


@deprecated("Use RetryExecutor.execute_with_retry() instead")
async def legacy_retry_operation(operation, max_retries: int = 3):
    """
    Legacy wrapper for retry operations.
    
    This function is deprecated and provided only for backward compatibility.
    Use RetryExecutor.execute_with_retry() instead.
    """
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            if attempt >= max_retries - 1:
                raise
            await asyncio.sleep(0.1 * (2 ** attempt))
