"""
Database operations for the MCP Code Indexer.

This module provides async database operations using aiosqlite with proper
connection management, transaction handling, and performance optimizations.
"""

import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, AsyncIterator

import asyncio
import random
import aiosqlite

from mcp_code_indexer.database.models import (
    Project, FileDescription, MergeConflict, SearchResult,
    CodebaseSizeInfo, ProjectOverview, WordFrequencyResult, WordFrequencyTerm
)
from mcp_code_indexer.database.retry_executor import (
    RetryExecutor, create_retry_executor
)
from mcp_code_indexer.database.exceptions import (
    DatabaseError, DatabaseLockError, classify_sqlite_error, is_retryable_error
)
from mcp_code_indexer.database.connection_health import (
    ConnectionHealthMonitor, DatabaseMetricsCollector
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations with async support.
    
    Provides high-level operations for projects, file descriptions, search,
    and caching with proper transaction management and error handling.
    """
    
    def __init__(self, 
                 db_path: Path, 
                 pool_size: int = 3,
                 retry_count: int = 5,
                 timeout: float = 10.0,
                 enable_wal_mode: bool = True,
                 health_check_interval: float = 30.0,
                 retry_min_wait: float = 0.1,
                 retry_max_wait: float = 2.0,
                 retry_jitter: float = 0.2):
        """Initialize database manager with path to SQLite database."""
        self.db_path = db_path
        self.pool_size = pool_size
        self.retry_count = retry_count
        self.timeout = timeout
        self.enable_wal_mode = enable_wal_mode
        self.health_check_interval = health_check_interval
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.retry_jitter = retry_jitter
        self._connection_pool: List[aiosqlite.Connection] = []
        self._pool_lock = None  # Will be initialized in async context
        self._write_lock = None  # Write serialization lock, initialized in async context
        
        # Retry and recovery components - configure with provided settings
        self._retry_executor = create_retry_executor(
            max_attempts=retry_count,
            min_wait_seconds=retry_min_wait,
            max_wait_seconds=retry_max_wait,
            jitter_max_seconds=retry_jitter
        )
        
        # Health monitoring and metrics
        self._health_monitor = None  # Initialized in async context
        self._metrics_collector = DatabaseMetricsCollector()
        
    async def initialize(self) -> None:
        """Initialize database schema and configuration."""
        import asyncio
        
        # Initialize locks
        self._pool_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        
        # Connection recovery is now handled by the retry executor
        
        # Initialize health monitoring with configured interval
        self._health_monitor = ConnectionHealthMonitor(
            self, 
            check_interval=self.health_check_interval,
            timeout_seconds=self.timeout
        )
        await self._health_monitor.start_monitoring()
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Database initialization now uses the modern retry executor directly
        
        # Apply migrations in order
        migrations_dir = Path(__file__).parent.parent.parent.parent / "migrations"
        migration_files = sorted(migrations_dir.glob("*.sql"))
        
        async with aiosqlite.connect(self.db_path) as db:
            # Enable row factory for easier data access
            db.row_factory = aiosqlite.Row
            
            # Configure WAL mode and optimizations for concurrent access
            await self._configure_database_optimizations(db, include_wal_mode=self.enable_wal_mode)
            
            # Apply each migration
            for migration_file in migration_files:
                logger.info(f"Applying migration: {migration_file.name}")
                with open(migration_file, 'r') as f:
                    migration_sql = f.read()
                
                await db.executescript(migration_sql)
                await db.commit()
            
        logger.info(f"Database initialized at {self.db_path} with {len(migration_files)} migrations")
    
    async def _configure_database_optimizations(self, db: aiosqlite.Connection, include_wal_mode: bool = True) -> None:
        """
        Configure SQLite optimizations for concurrent access and performance.
        
        Args:
            db: Database connection to configure
            include_wal_mode: Whether to set WAL mode (only needed once per database)
        """
        optimizations = []
        
        # WAL mode is database-level, only set during initialization
        if include_wal_mode:
            optimizations.append("PRAGMA journal_mode = WAL")
            logger.info("Enabling WAL mode for database concurrency")
        
        # Connection-level optimizations that can be set per connection
        optimizations.extend([
            "PRAGMA synchronous = NORMAL",      # Balance durability/performance  
            "PRAGMA cache_size = -64000",       # 64MB cache
            "PRAGMA temp_store = MEMORY",       # Use memory for temp tables
            "PRAGMA mmap_size = 268435456",     # 256MB memory mapping
            "PRAGMA busy_timeout = 10000",      # 10 second timeout (reduced from 30s)
            "PRAGMA optimize"                   # Enable query planner optimizations
        ])
        
        # WAL-specific settings (only if WAL mode is being set)
        if include_wal_mode:
            optimizations.append("PRAGMA wal_autocheckpoint = 1000")  # Checkpoint after 1000 pages
        
        for pragma in optimizations:
            try:
                await db.execute(pragma)
                logger.debug(f"Applied optimization: {pragma}")
            except Exception as e:
                logger.warning(f"Failed to apply optimization '{pragma}': {e}")
        
        await db.commit()
        if include_wal_mode:
            logger.info("Database optimizations configured for concurrent access with WAL mode")
        else:
            logger.debug("Connection optimizations applied")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Get a database connection from pool or create new one."""
        conn = None
        
        # Try to get from pool
        if self._pool_lock:
            async with self._pool_lock:
                if self._connection_pool:
                    conn = self._connection_pool.pop()
        
        # Create new connection if none available
        if conn is None:
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            
            # Apply connection-level optimizations (WAL mode already set during initialization)
            await self._configure_database_optimizations(conn, include_wal_mode=False)
        
        try:
            yield conn
        finally:
            # Return to pool if pool not full, otherwise close
            returned_to_pool = False
            if self._pool_lock and len(self._connection_pool) < self.pool_size:
                async with self._pool_lock:
                    if len(self._connection_pool) < self.pool_size:
                        self._connection_pool.append(conn)
                        returned_to_pool = True
            
            if not returned_to_pool:
                await conn.close()
    
    async def close_pool(self) -> None:
        """Close all connections in the pool and stop monitoring."""
        # Stop health monitoring
        if self._health_monitor:
            await self._health_monitor.stop_monitoring()
        
        # Close connections
        if self._pool_lock:
            async with self._pool_lock:
                for conn in self._connection_pool:
                    await conn.close()
                self._connection_pool.clear()
    
    @asynccontextmanager
    async def get_write_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """
        Get a database connection with write serialization.
        
        This ensures only one write operation occurs at a time across the entire
        application, preventing database locking issues in multi-client scenarios.
        """
        if self._write_lock is None:
            raise RuntimeError("DatabaseManager not initialized - call initialize() first")
        
        async with self._write_lock:
            async with self.get_connection() as conn:
                yield conn
    
    @asynccontextmanager
    async def get_write_connection_with_retry(self, operation_name: str = "write_operation") -> AsyncIterator[aiosqlite.Connection]:
        """
        Get a database connection with write serialization and automatic retry logic.
        
        This uses the new RetryExecutor to properly handle retry logic without
        the broken yield-in-retry-loop pattern that caused generator errors.
        
        Args:
            operation_name: Name of the operation for logging and monitoring
        """
        if self._write_lock is None:
            raise RuntimeError("DatabaseManager not initialized - call initialize() first")
        
        async def get_write_connection():
            """Inner function to get connection - will be retried by executor."""
            async with self._write_lock:
                async with self.get_connection() as conn:
                    return conn
        
        try:
            # Use retry executor to handle connection acquisition with retries
            connection = await self._retry_executor.execute_with_retry(
                get_write_connection, 
                operation_name
            )
            
            try:
                yield connection
                
                # Success - retry executor handles all failure tracking
                    
            except Exception as e:
                # Error handling is managed by the retry executor
                raise
                
        except DatabaseError:
            # Re-raise our custom database errors as-is
            raise
        except Exception as e:
            # Classify and wrap other exceptions
            classified_error = classify_sqlite_error(e, operation_name)
            logger.error(
                f"Database operation '{operation_name}' failed: {classified_error.message}",
                extra={"structured_data": classified_error.to_dict()}
            )
            raise classified_error
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database performance and reliability statistics.
        
        Returns:
            Dictionary with retry stats, recovery stats, health status, and metrics
        """
        stats = {
            "connection_pool": {
                "configured_size": self.pool_size,
                "current_size": len(self._connection_pool)
            },
            "retry_executor": self._retry_executor.get_retry_stats() if self._retry_executor else {},
        }
        
        # Legacy retry handler removed - retry executor stats are included above
        
        if self._health_monitor:
            stats["health_status"] = self._health_monitor.get_health_status()
        
        if self._metrics_collector:
            stats["operation_metrics"] = self._metrics_collector.get_operation_metrics()
            stats["locking_frequency"] = self._metrics_collector.get_locking_frequency()
        
        return stats
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Perform an immediate health check and return detailed status.
        
        Returns:
            Dictionary with health check result and current metrics
        """
        if not self._health_monitor:
            return {"error": "Health monitoring not initialized"}
        
        # Perform immediate health check
        health_result = await self._health_monitor.check_health()
        
        return {
            "health_check": {
                "is_healthy": health_result.is_healthy,
                "response_time_ms": health_result.response_time_ms,
                "error_message": health_result.error_message,
                "timestamp": health_result.timestamp.isoformat()
            },
            "overall_status": self._health_monitor.get_health_status(),
            "recent_history": self._health_monitor.get_recent_history()
        }
    
    @asynccontextmanager
    async def get_immediate_transaction(
        self, 
        operation_name: str = "immediate_transaction",
        timeout_seconds: float = 10.0
    ) -> AsyncIterator[aiosqlite.Connection]:
        """
        Get a database connection with BEGIN IMMEDIATE transaction and timeout.
        
        This ensures write locks are acquired immediately, preventing lock escalation
        failures that can occur with DEFERRED transactions.
        
        Args:
            operation_name: Name of the operation for monitoring
            timeout_seconds: Transaction timeout in seconds
        """
        async with self.get_write_connection_with_retry(operation_name) as conn:
            try:
                # Start immediate transaction with timeout
                async with asyncio.timeout(timeout_seconds):
                    await conn.execute("BEGIN IMMEDIATE")
                    yield conn
                    await conn.commit()
            except asyncio.TimeoutError:
                logger.warning(
                    f"Transaction timeout after {timeout_seconds}s for {operation_name}",
                    extra={
                        "structured_data": {
                            "transaction_timeout": {
                                "operation": operation_name,
                                "timeout_seconds": timeout_seconds
                            }
                        }
                    }
                )
                await conn.rollback()
                raise
            except Exception as e:
                logger.error(f"Transaction failed for {operation_name}: {e}")
                await conn.rollback()
                raise
    
    async def execute_transaction_with_retry(
        self,
        operation_func,
        operation_name: str = "transaction_operation",
        max_retries: int = 3,
        timeout_seconds: float = 10.0
    ) -> Any:
        """
        Execute a database operation within a transaction with automatic retry.
        
        Uses the new RetryExecutor for robust retry handling with proper error
        classification and exponential backoff.
        
        Args:
            operation_func: Async function that takes a connection and performs the operation
            operation_name: Name of the operation for logging
            max_retries: Maximum retry attempts (overrides default retry executor config)
            timeout_seconds: Transaction timeout in seconds
            
        Returns:
            Result from operation_func
            
        Example:
            async def my_operation(conn):
                await conn.execute("INSERT INTO ...", (...))
                return "success"
            
            result = await db.execute_transaction_with_retry(my_operation, "insert_data")
        """
        
        async def execute_transaction():
            """Inner function to execute transaction - will be retried by executor."""
            try:
                async with self.get_immediate_transaction(operation_name, timeout_seconds) as conn:
                    result = await operation_func(conn)
                    
                # Record successful operation metrics
                if self._metrics_collector:
                    self._metrics_collector.record_operation(
                        operation_name, 
                        timeout_seconds * 1000,  # Convert to ms
                        True,
                        len(self._connection_pool)
                    )
                
                return result
                
            except (aiosqlite.OperationalError, asyncio.TimeoutError) as e:
                # Record locking event for metrics
                if self._metrics_collector and "locked" in str(e).lower():
                    self._metrics_collector.record_locking_event(operation_name, str(e))
                
                # Classify the error for better handling
                classified_error = classify_sqlite_error(e, operation_name)
                
                # Record failed operation metrics for non-retryable errors
                if not is_retryable_error(classified_error):
                    if self._metrics_collector:
                        self._metrics_collector.record_operation(
                            operation_name,
                            timeout_seconds * 1000,
                            False,
                            len(self._connection_pool)
                        )
                
                raise classified_error
        
        try:
            # Create a temporary retry executor with custom max_retries if different from default
            if max_retries != self._retry_executor.config.max_attempts:
                from mcp_code_indexer.database.retry_executor import RetryConfig, RetryExecutor
                temp_config = RetryConfig(
                    max_attempts=max_retries,
                    min_wait_seconds=self._retry_executor.config.min_wait_seconds,
                    max_wait_seconds=self._retry_executor.config.max_wait_seconds,
                    jitter_max_seconds=self._retry_executor.config.jitter_max_seconds
                )
                temp_executor = RetryExecutor(temp_config)
                return await temp_executor.execute_with_retry(execute_transaction, operation_name)
            else:
                return await self._retry_executor.execute_with_retry(execute_transaction, operation_name)
                
        except DatabaseError as e:
            # Record failed operation metrics for final failure
            if self._metrics_collector:
                self._metrics_collector.record_operation(
                    operation_name,
                    timeout_seconds * 1000,
                    False,
                    len(self._connection_pool)
                )
            raise
    
    # Project operations
    
    async def create_project(self, project: Project) -> None:
        """Create a new project record."""
        async with self.get_write_connection_with_retry("create_project") as db:
            await db.execute(
                """
                INSERT INTO projects (id, name, remote_origin, upstream_origin, aliases, created, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.id,
                    project.name,
                    project.remote_origin,
                    project.upstream_origin,
                    json.dumps(project.aliases),
                    project.created,
                    project.last_accessed
                )
            )
            await db.commit()
            logger.debug(f"Created project: {project.id}")
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                "SELECT * FROM projects WHERE id = ?",
                (project_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                return Project(
                    id=row['id'],
                    name=row['name'],
                    remote_origin=row['remote_origin'],
                    upstream_origin=row['upstream_origin'],
                    aliases=json.loads(row['aliases']),
                    created=datetime.fromisoformat(row['created']),
                    last_accessed=datetime.fromisoformat(row['last_accessed'])
                )
            return None
    
    async def find_project_by_origin(self, origin_url: str) -> Optional[Project]:
        """Find project by remote or upstream origin URL."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM projects 
                WHERE remote_origin = ? OR upstream_origin = ?
                LIMIT 1
                """,
                (origin_url, origin_url)
            )
            row = await cursor.fetchone()
            
            if row:
                return Project(
                    id=row['id'],
                    name=row['name'],
                    remote_origin=row['remote_origin'],
                    upstream_origin=row['upstream_origin'],
                    aliases=json.loads(row['aliases']),
                    created=datetime.fromisoformat(row['created']),
                    last_accessed=datetime.fromisoformat(row['last_accessed'])
                )
            return None

    async def find_matching_project(
        self, 
        project_name: str, 
        remote_origin: Optional[str] = None,
        upstream_origin: Optional[str] = None,
        folder_path: Optional[str] = None
    ) -> Optional[Project]:
        """
        Find project by matching criteria.
        
        Args:
            project_name: Name of the project
            remote_origin: Remote origin URL
            upstream_origin: Upstream origin URL  
            folder_path: Project folder path
            
        Returns:
            Matching project or None
        """
        projects = await self.get_all_projects()
        normalized_name = project_name.lower()
        
        best_match = None
        best_score = 0
        
        for project in projects:
            score = 0
            match_factors = []
            
            # Check name match (case-insensitive)
            if project.name.lower() == normalized_name:
                score += 1
                match_factors.append("name")
            
            # Check remote origin match
            if remote_origin and project.remote_origin == remote_origin:
                score += 1
                match_factors.append("remote_origin")
            
            # Check upstream origin match
            if upstream_origin and project.upstream_origin == upstream_origin:
                score += 1
                match_factors.append("upstream_origin")
            
            # Check folder path in aliases
            if folder_path and folder_path in project.aliases:
                score += 1
                match_factors.append("folder_path")
            
            # Enhanced matching: If name matches and no remote origins are provided,
            # consider it a strong match to prevent duplicates
            if (score == 1 and "name" in match_factors and 
                not remote_origin and not project.remote_origin and
                not upstream_origin and not project.upstream_origin):
                logger.info(f"Name-only match with no remotes for project {project.name} - treating as strong match")
                score = 2  # Boost score to strong match level
                match_factors.append("no_remotes_boost")
            
            # If we have 2+ matches, this is a strong candidate
            if score >= 2:
                if score > best_score:
                    best_score = score
                    best_match = project
                    logger.info(f"Strong match for project {project.name} (score: {score}, factors: {match_factors})")
        
        return best_match

    async def get_or_create_project(
        self,
        project_name: str,
        folder_path: str,
        remote_origin: Optional[str] = None,
        upstream_origin: Optional[str] = None
    ) -> Project:
        """
        Get or create a project using intelligent matching.
        
        Args:
            project_name: Name of the project
            folder_path: Project folder path
            remote_origin: Remote origin URL
            upstream_origin: Upstream origin URL
            
        Returns:
            Existing or newly created project
        """
        # Try to find existing project
        project = await self.find_matching_project(
            project_name, remote_origin, upstream_origin, folder_path
        )
        
        if project:
            # Update aliases if folder path not already included
            if folder_path not in project.aliases:
                project.aliases.append(folder_path)
                await self.update_project(project)
                logger.info(f"Added folder path {folder_path} to project {project.name} aliases")
            
            # Update access time
            await self.update_project_access_time(project.id)
            return project
        
        # Create new project
        from ..database.models import Project
        import uuid
        
        new_project = Project(
            id=str(uuid.uuid4()),
            name=project_name,
            remote_origin=remote_origin,
            upstream_origin=upstream_origin,
            aliases=[folder_path],
            created=datetime.utcnow(),
            last_accessed=datetime.utcnow()
        )
        
        await self.create_project(new_project)
        logger.info(f"Created new project: {new_project.name} ({new_project.id})")
        return new_project
    
    async def update_project_access_time(self, project_id: str) -> None:
        """Update the last accessed time for a project."""
        async with self.get_write_connection_with_retry("update_project_access_time") as db:
            await db.execute(
                "UPDATE projects SET last_accessed = ? WHERE id = ?",
                (datetime.utcnow(), project_id)
            )
            await db.commit()
    
    async def update_project(self, project: Project) -> None:
        """Update an existing project record."""
        async with self.get_write_connection_with_retry("update_project") as db:
            await db.execute(
                """
                UPDATE projects 
                SET name = ?, remote_origin = ?, upstream_origin = ?, aliases = ?, last_accessed = ?
                WHERE id = ?
                """,
                (
                    project.name,
                    project.remote_origin,
                    project.upstream_origin,
                    json.dumps(project.aliases),
                    project.last_accessed,
                    project.id
                )
            )
            await db.commit()
            logger.debug(f"Updated project: {project.id}")
    
    async def get_all_projects(self) -> List[Project]:
        """Get all projects in the database."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                "SELECT id, name, remote_origin, upstream_origin, aliases, created, last_accessed FROM projects"
            )
            rows = await cursor.fetchall()
            
            projects = []
            for row in rows:
                aliases = json.loads(row[4]) if row[4] else []
                project = Project(
                    id=row[0],
                    name=row[1],
                    remote_origin=row[2],
                    upstream_origin=row[3],
                    aliases=aliases,
                    created=row[5],
                    last_accessed=row[6]
                )
                projects.append(project)
            
            return projects
    
    async def get_branch_file_counts(self, project_id: str) -> Dict[str, int]:
        """Get file counts per branch for a project."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT branch, COUNT(*) as file_count 
                FROM file_descriptions 
                WHERE project_id = ? 
                GROUP BY branch
                """,
                (project_id,)
            )
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}
    
    # File description operations
    
    async def create_file_description(self, file_desc: FileDescription) -> None:
        """Create or update a file description."""
        async with self.get_write_connection_with_retry("create_file_description") as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO file_descriptions 
                (project_id, branch, file_path, description, file_hash, last_modified, version, source_project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_desc.project_id,
                    file_desc.branch,
                    file_desc.file_path,
                    file_desc.description,
                    file_desc.file_hash,
                    file_desc.last_modified,
                    file_desc.version,
                    file_desc.source_project_id
                )
            )
            await db.commit()
            logger.debug(f"Saved file description: {file_desc.file_path}")
    
    async def get_file_description(
        self, 
        project_id: str, 
        branch: str, 
        file_path: str
    ) -> Optional[FileDescription]:
        """Get file description by project, branch, and path."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM file_descriptions 
                WHERE project_id = ? AND branch = ? AND file_path = ?
                """,
                (project_id, branch, file_path)
            )
            row = await cursor.fetchone()
            
            if row:
                return FileDescription(
                    project_id=row['project_id'],
                    branch=row['branch'],
                    file_path=row['file_path'],
                    description=row['description'],
                    file_hash=row['file_hash'],
                    last_modified=datetime.fromisoformat(row['last_modified']),
                    version=row['version'],
                    source_project_id=row['source_project_id']
                )
            return None
    
    async def get_all_file_descriptions(
        self, 
        project_id: str, 
        branch: str
    ) -> List[FileDescription]:
        """Get all file descriptions for a project and branch."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM file_descriptions 
                WHERE project_id = ? AND branch = ?
                ORDER BY file_path
                """,
                (project_id, branch)
            )
            rows = await cursor.fetchall()
            
            return [
                FileDescription(
                    project_id=row['project_id'],
                    branch=row['branch'],
                    file_path=row['file_path'],
                    description=row['description'],
                    file_hash=row['file_hash'],
                    last_modified=datetime.fromisoformat(row['last_modified']),
                    version=row['version'],
                    source_project_id=row['source_project_id']
                )
                for row in rows
            ]
    
    async def batch_create_file_descriptions(self, file_descriptions: List[FileDescription]) -> None:
        """Batch create multiple file descriptions efficiently with optimized transactions."""
        if not file_descriptions:
            return
        
        async def batch_operation(conn: aiosqlite.Connection) -> None:
            data = [
                (
                    fd.project_id,
                    fd.branch,
                    fd.file_path,
                    fd.description,
                    fd.file_hash,
                    fd.last_modified,
                    fd.version,
                    fd.source_project_id
                )
                for fd in file_descriptions
            ]
            
            await conn.executemany(
                """
                INSERT OR REPLACE INTO file_descriptions 
                (project_id, branch, file_path, description, file_hash, last_modified, version, source_project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data
            )
            logger.debug(f"Batch created {len(file_descriptions)} file descriptions")
        
        await self.execute_transaction_with_retry(
            batch_operation,
            f"batch_create_file_descriptions_{len(file_descriptions)}_files",
            timeout_seconds=30.0  # Longer timeout for batch operations
        )
    
    # Search operations
    
    async def search_file_descriptions(
        self,
        project_id: str,
        branch: str,
        query: str,
        max_results: int = 20
    ) -> List[SearchResult]:
        """Search file descriptions using FTS5."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT 
                    fd.project_id,
                    fd.branch,
                    fd.file_path,
                    fd.description,
                    bm25(file_descriptions_fts) as rank
                FROM file_descriptions_fts
                JOIN file_descriptions fd ON fd.rowid = file_descriptions_fts.rowid
                WHERE file_descriptions_fts MATCH ? 
                  AND fd.project_id = ? 
                  AND fd.branch = ?
                ORDER BY bm25(file_descriptions_fts)
                LIMIT ?
                """,
                (query, project_id, branch, max_results)
            )
            rows = await cursor.fetchall()
            
            return [
                SearchResult(
                    project_id=row['project_id'],
                    branch=row['branch'],
                    file_path=row['file_path'],
                    description=row['description'],
                    relevance_score=row['rank']
                )
                for row in rows
            ]
    
    # Token cache operations
    
    async def get_cached_token_count(self, cache_key: str) -> Optional[int]:
        """Get cached token count if not expired."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT token_count FROM token_cache 
                WHERE cache_key = ? AND (expires IS NULL OR expires > ?)
                """,
                (cache_key, datetime.utcnow())
            )
            row = await cursor.fetchone()
            return row['token_count'] if row else None
    
    async def cache_token_count(
        self, 
        cache_key: str, 
        token_count: int, 
        ttl_hours: int = 24
    ) -> None:
        """Cache token count with TTL."""
        expires = datetime.utcnow() + timedelta(hours=ttl_hours)
        
        async with self.get_write_connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO token_cache (cache_key, token_count, expires)
                VALUES (?, ?, ?)
                """,
                (cache_key, token_count, expires)
            )
            await db.commit()
    
    async def cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        async with self.get_write_connection() as db:
            await db.execute(
                "DELETE FROM token_cache WHERE expires < ?",
                (datetime.utcnow(),)
            )
            await db.commit()
    
    # Utility operations
    
    async def get_file_count(self, project_id: str, branch: str) -> int:
        """Get count of files in a project branch."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                "SELECT COUNT(*) as count FROM file_descriptions WHERE project_id = ? AND branch = ?",
                (project_id, branch)
            )
            row = await cursor.fetchone()
            return row['count'] if row else 0
    
    # Upstream inheritance operations
    
    async def inherit_from_upstream(self, project: Project, target_branch: str = "main") -> int:
        """
        Inherit file descriptions from upstream repository.
        
        Args:
            project: Target project that should inherit descriptions
            target_branch: Branch to inherit descriptions into
            
        Returns:
            Number of descriptions inherited
        """
        if not project.upstream_origin:
            return 0
        
        # Find upstream project
        upstream_project = await self.find_project_by_origin(project.upstream_origin)
        if not upstream_project:
            logger.debug(f"No upstream project found for {project.upstream_origin}")
            return 0
        
        # Get upstream descriptions
        upstream_descriptions = await self.get_all_file_descriptions(
            upstream_project.id, target_branch
        )
        
        if not upstream_descriptions:
            logger.debug(f"No upstream descriptions found in branch {target_branch}")
            return 0
        
        # Get existing descriptions to avoid overwriting
        existing_descriptions = await self.get_all_file_descriptions(
            project.id, target_branch
        )
        existing_paths = {desc.file_path for desc in existing_descriptions}
        
        # Create new descriptions for files that don't exist locally
        inherited_descriptions = []
        for upstream_desc in upstream_descriptions:
            if upstream_desc.file_path not in existing_paths:
                new_desc = FileDescription(
                    project_id=project.id,
                    branch=target_branch,
                    file_path=upstream_desc.file_path,
                    description=upstream_desc.description,
                    file_hash=None,  # Don't copy hash as local file may differ
                    last_modified=datetime.utcnow(),
                    version=1,
                    source_project_id=upstream_project.id  # Track inheritance source
                )
                inherited_descriptions.append(new_desc)
        
        if inherited_descriptions:
            await self.batch_create_file_descriptions(inherited_descriptions)
            logger.info(f"Inherited {len(inherited_descriptions)} descriptions from upstream")
        
        return len(inherited_descriptions)
    
    async def check_upstream_inheritance_needed(self, project: Project) -> bool:
        """
        Check if a project needs upstream inheritance.
        
        Args:
            project: Project to check
            
        Returns:
            True if project has upstream but no descriptions yet
        """
        if not project.upstream_origin:
            return False
        
        # Check if project has any descriptions
        file_count = await self.get_file_count(project.id, "main")
        return file_count == 0
    
    # Project Overview operations
    
    async def create_project_overview(self, overview: ProjectOverview) -> None:
        """Create or update a project overview."""
        async with self.get_write_connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO project_overviews 
                (project_id, branch, overview, last_modified, total_files, total_tokens)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    overview.project_id,
                    overview.branch,
                    overview.overview,
                    overview.last_modified,
                    overview.total_files,
                    overview.total_tokens
                )
            )
            await db.commit()
            logger.debug(f"Created/updated overview for project {overview.project_id}, branch {overview.branch}")
    
    async def get_project_overview(self, project_id: str, branch: str) -> Optional[ProjectOverview]:
        """Get project overview by ID and branch."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                "SELECT * FROM project_overviews WHERE project_id = ? AND branch = ?",
                (project_id, branch)
            )
            row = await cursor.fetchone()
            
            if row:
                return ProjectOverview(
                    project_id=row['project_id'],
                    branch=row['branch'],
                    overview=row['overview'],
                    last_modified=datetime.fromisoformat(row['last_modified']),
                    total_files=row['total_files'],
                    total_tokens=row['total_tokens']
                )
            return None
    
    async def cleanup_missing_files(self, project_id: str, branch: str, project_root: Path) -> List[str]:
        """
        Remove descriptions for files that no longer exist on disk.
        
        Args:
            project_id: Project identifier
            branch: Branch name
            project_root: Path to project root directory
            
        Returns:
            List of file paths that were cleaned up
        """
        removed_files = []
        
        async def cleanup_operation(conn: aiosqlite.Connection) -> List[str]:
            # Get all file descriptions for this project/branch
            cursor = await conn.execute(
                "SELECT file_path FROM file_descriptions WHERE project_id = ? AND branch = ?",
                (project_id, branch)
            )
            
            rows = await cursor.fetchall()
            
            # Check which files no longer exist
            to_remove = []
            for row in rows:
                file_path = row['file_path']
                full_path = project_root / file_path
                
                if not full_path.exists():
                    to_remove.append(file_path)
            
            # Remove descriptions for missing files
            if to_remove:
                await conn.executemany(
                    "DELETE FROM file_descriptions WHERE project_id = ? AND branch = ? AND file_path = ?",
                    [(project_id, branch, path) for path in to_remove]
                )
                logger.info(f"Cleaned up {len(to_remove)} missing files from {project_id}/{branch}")
            
            return to_remove
        
        removed_files = await self.execute_transaction_with_retry(
            cleanup_operation,
            f"cleanup_missing_files_{project_id}_{branch}",
            timeout_seconds=60.0  # Longer timeout for file system operations
        )
        
        return removed_files
    
    async def analyze_word_frequency(self, project_id: str, branch: str, limit: int = 200) -> WordFrequencyResult:
        """
        Analyze word frequency across all file descriptions for a project/branch.
        
        Args:
            project_id: Project identifier
            branch: Branch name
            limit: Maximum number of top terms to return
            
        Returns:
            WordFrequencyResult with top terms and statistics
        """
        from collections import Counter
        import re
        
        # Load stop words from bundled file
        stop_words_path = Path(__file__).parent.parent / "data" / "stop_words_english.txt"
        stop_words = set()
        
        if stop_words_path.exists():
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Each line contains just the stop word
                    word = line.strip().lower()
                    if word:  # Skip empty lines
                        stop_words.add(word)
        
        # Add common programming keywords to stop words
        programming_keywords = {
            'if', 'else', 'for', 'while', 'do', 'break', 'continue', 'return',
            'function', 'class', 'def', 'var', 'let', 'const', 'public', 'private',
            'static', 'async', 'await', 'import', 'export', 'from', 'true', 'false',
            'null', 'undefined', 'this', 'that', 'self', 'super', 'new', 'delete'
        }
        stop_words.update(programming_keywords)
        
        async with self.get_connection() as db:
            # Get all descriptions for this project/branch
            cursor = await db.execute(
                "SELECT description FROM file_descriptions WHERE project_id = ? AND branch = ?",
                (project_id, branch)
            )
            
            rows = await cursor.fetchall()
            
            # Combine all descriptions
            all_text = " ".join(row['description'] for row in rows)
            
            # Tokenize and filter
            words = re.findall(r'\b[a-zA-Z]{2,}\b', all_text.lower())
            filtered_words = [word for word in words if word not in stop_words]
            
            # Count frequencies
            word_counts = Counter(filtered_words)
            
            # Create result
            top_terms = [
                WordFrequencyTerm(term=term, frequency=count)
                for term, count in word_counts.most_common(limit)
            ]
            
            return WordFrequencyResult(
            top_terms=top_terms,
            total_terms_analyzed=len(filtered_words),
            total_unique_terms=len(word_counts)
            )
    
    async def cleanup_empty_projects(self) -> int:
        """
        Remove projects that have no file descriptions and no project overview.
        
        Returns:
            Number of projects removed
        """
        async with self.get_write_connection() as db:
            # Find projects with no descriptions and no overview
            cursor = await db.execute("""
                SELECT p.id, p.name 
                FROM projects p
                LEFT JOIN file_descriptions fd ON p.id = fd.project_id
                LEFT JOIN project_overviews po ON p.id = po.project_id
                WHERE fd.project_id IS NULL AND po.project_id IS NULL
            """)
            
            empty_projects = await cursor.fetchall()
            
            if not empty_projects:
                return 0
            
            removed_count = 0
            for project in empty_projects:
                project_id = project['id']
                project_name = project['name']
                
                # Remove from projects table (cascading will handle related data)
                await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
                removed_count += 1
                
                logger.info(f"Removed empty project: {project_name} (ID: {project_id})")
            
            await db.commit()
            return removed_count
    
    async def get_project_map_data(self, project_identifier: str, branch: str = None) -> dict:
        """
        Get all data needed to generate a project map.
        
        Args:
            project_identifier: Project name or ID
            branch: Branch name (optional, will use first available if not specified)
            
        Returns:
            Dictionary containing project info, overview, and file descriptions
        """
        async with self.get_connection() as db:
            # Try to find project by ID first, then by name
            if len(project_identifier) == 36 and '-' in project_identifier:
                # Looks like a UUID
                cursor = await db.execute(
                    "SELECT * FROM projects WHERE id = ?", 
                    (project_identifier,)
                )
            else:
                # Search by name
                cursor = await db.execute(
                    "SELECT * FROM projects WHERE LOWER(name) = LOWER(?)", 
                    (project_identifier,)
                )
            
            project_row = await cursor.fetchone()
            if not project_row:
                return None
            
            # Handle aliases JSON parsing
            project_dict = dict(project_row)
            if isinstance(project_dict['aliases'], str):
                import json
                project_dict['aliases'] = json.loads(project_dict['aliases'])
            
            project = Project(**project_dict)
            
            # If no branch specified, find the first available branch
            if not branch:
                cursor = await db.execute(
                    "SELECT DISTINCT branch FROM file_descriptions WHERE project_id = ? LIMIT 1",
                    (project.id,)
                )
                branch_row = await cursor.fetchone()
                if branch_row:
                    branch = branch_row['branch']
                else:
                    branch = 'main'  # Default fallback
            
            # Get project overview
            cursor = await db.execute(
                "SELECT * FROM project_overviews WHERE project_id = ? AND branch = ?",
                (project.id, branch)
            )
            overview_row = await cursor.fetchone()
            project_overview = ProjectOverview(**overview_row) if overview_row else None
            
            # Get all file descriptions for this project/branch
            cursor = await db.execute(
                """SELECT * FROM file_descriptions 
                   WHERE project_id = ? AND branch = ? 
                   ORDER BY file_path""",
                (project.id, branch)
            )
            file_rows = await cursor.fetchall()
            file_descriptions = [FileDescription(**row) for row in file_rows]
            
            return {
                'project': project,
                'branch': branch,
                'overview': project_overview,
                'files': file_descriptions
            }
