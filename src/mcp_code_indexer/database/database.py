"""
Database operations for the MCP Code Indexer.

This module provides async database operations using aiosqlite with proper
connection management, transaction handling, and performance optimizations.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Callable

import aiosqlite

from mcp_code_indexer.cleanup_manager import CleanupManager
from mcp_code_indexer.database.connection_health import (
    ConnectionHealthMonitor,
    DatabaseMetricsCollector,
)
from mcp_code_indexer.database.exceptions import (
    DatabaseError,
    classify_sqlite_error,
    is_retryable_error,
)
from mcp_code_indexer.database.models import (
    FileDescription,
    IndexMeta,
    Project,
    ProjectOverview,
    SearchResult,
    SyncStatus,
    WordFrequencyResult,
    WordFrequencyTerm,
)
from mcp_code_indexer.database.retry_executor import create_retry_executor
from mcp_code_indexer.query_preprocessor import preprocess_search_query

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations with async support.

    Provides high-level operations for projects, file descriptions, search,
    and caching with proper transaction management and error handling.
    """

    def __init__(
        self,
        db_path: Path,
        pool_size: int = 3,
        retry_count: int = 5,
        timeout: float = 10.0,
        enable_wal_mode: bool = True,
        health_check_interval: float = 30.0,
        retry_min_wait: float = 0.1,
        retry_max_wait: float = 2.0,
        retry_jitter: float = 0.2,
    ):
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
        self._pool_lock: Optional[asyncio.Lock] = (
            None  # Will be initialized in async context
        )
        self._write_lock: Optional[asyncio.Lock] = (
            None  # Write serialization lock, async context
        )

        # Retry and recovery components - configure with provided settings
        self._retry_executor = create_retry_executor(
            max_attempts=retry_count,
            min_wait_seconds=retry_min_wait,
            max_wait_seconds=retry_max_wait,
            jitter_max_seconds=retry_jitter,
        )

        # Health monitoring and metrics
        self._health_monitor: Optional[ConnectionHealthMonitor] = (
            None  # Initialized in async context
        )
        self._metrics_collector = DatabaseMetricsCollector()

        # Cleanup manager for retention policies
        self._cleanup_manager: Optional[CleanupManager] = (
            None  # Initialized in async context
        )

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
            timeout_seconds=self.timeout,
        )
        await self._health_monitor.start_monitoring()

        # Initialize cleanup manager
        self._cleanup_manager = CleanupManager(self, retention_months=6)

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Database initialization now uses the modern retry executor directly

        # Apply migrations in order
        # Migrations are now bundled with the package
        migrations_dir = Path(__file__).parent.parent / "migrations"
        if not migrations_dir.exists():
            raise RuntimeError(
                f"Could not find migrations directory at {migrations_dir}"
            )
        migration_files = sorted(migrations_dir.glob("*.sql"))

        async with aiosqlite.connect(self.db_path) as db:
            # Enable row factory for easier data access
            db.row_factory = aiosqlite.Row

            # Configure WAL mode and optimizations for concurrent access
            await self._configure_database_optimizations(
                db, include_wal_mode=self.enable_wal_mode
            )

            # Create migrations tracking table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            await db.commit()

            # Get list of already applied migrations
            cursor = await db.execute("SELECT filename FROM migrations")
            applied_migrations = {row[0] for row in await cursor.fetchall()}

            # Apply each migration that hasn't been applied yet
            for migration_file in migration_files:
                migration_name = migration_file.name
                if migration_name in applied_migrations:
                    logger.info(f"Skipping already applied migration: {migration_name}")
                    continue

                logger.info(f"Applying migration: {migration_name}")
                try:
                    migration_sql = migration_file.read_text(encoding="utf-8")
                except AttributeError:
                    # Fallback for regular file objects
                    with open(migration_file, "r", encoding="utf-8") as f:
                        migration_sql = f.read()

                try:
                    await db.executescript(migration_sql)

                    # Record that migration was applied
                    await db.execute(
                        "INSERT INTO migrations (filename) VALUES (?)",
                        (migration_name,),
                    )
                    await db.commit()
                    logger.info(f"Successfully applied migration: {migration_name}")
                except Exception as e:
                    logger.error(f"Failed to apply migration {migration_name}: {e}")
                    await db.rollback()
                    raise

        logger.info(
            (
                f"Database initialized at {self.db_path} with "
                f"{len(migration_files)} total migrations"
            )
        )

    async def _configure_database_optimizations(
        self,
        db: aiosqlite.Connection,
        include_wal_mode: bool = True,
    ) -> None:
        """
        Configure SQLite optimizations for concurrent access and performance.

        Args:
            db: Database connection to configure
            include_wal_mode: Whether to set WAL mode (only needed once per
                database)
        """
        optimizations = []

        # WAL mode is database-level, only set during initialization
        if include_wal_mode:
            optimizations.append("PRAGMA journal_mode = WAL")
            logger.info("Enabling WAL mode for database concurrency")

        # Connection-level optimizations that can be set per connection
        optimizations.extend(
            [
                "PRAGMA synchronous = NORMAL",  # Balance durability/performance
                "PRAGMA cache_size = -64000",  # 64MB cache
                "PRAGMA temp_store = MEMORY",  # Use memory for temp tables
                "PRAGMA mmap_size = 268435456",  # 256MB memory mapping
                "PRAGMA busy_timeout = 10000",  # 10s timeout (reduced from 30s)
                "PRAGMA optimize",  # Enable query planner optimizations
            ]
        )

        # WAL-specific settings (only if WAL mode is being set)
        if include_wal_mode:
            optimizations.append(
                "PRAGMA wal_autocheckpoint = 1000"
            )  # Checkpoint after 1000 pages

        for pragma in optimizations:
            try:
                await db.execute(pragma)
                logger.debug(f"Applied optimization: {pragma}")
            except Exception as e:
                logger.warning(f"Failed to apply optimization '{pragma}': {e}")

        await db.commit()
        if include_wal_mode:
            logger.info(
                "Database optimizations configured for concurrent access with WAL mode"
            )
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

            # Apply connection-level optimizations (WAL mode set during init)
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

        This ensures only one write operation occurs at a time across the
        entire application, preventing database locking issues in
        multi-client scenarios.
        """
        if self._write_lock is None:
            raise RuntimeError(
                "DatabaseManager not initialized - call initialize() first"
            )

        async with self._write_lock:
            async with self.get_connection() as conn:
                yield conn

    @asynccontextmanager
    async def get_write_connection_with_retry(
        self, operation_name: str = "write_operation"
    ) -> AsyncIterator[aiosqlite.Connection]:
        """
        Get a database connection with write serialization.

        Ensures the write lock is held throughout the duration of the context
        to prevent race conditions and database locking errors.

        Args:
            operation_name: Name of the operation for logging and
                monitoring
        """
        if self._write_lock is None:
            raise RuntimeError(
                "DatabaseManager not initialized - call initialize() first"
            )

        # Acquire lock for exclusive write access - hold it for entire context
        async with self._write_lock:
            async with self.get_connection() as conn:
                yield conn

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database performance and reliability statistics.

        Returns:
            Dictionary with retry stats, recovery stats, health status,
            and metrics
        """
        stats = {
            "connection_pool": {
                "configured_size": self.pool_size,
                "current_size": len(self._connection_pool),
            },
            "retry_executor": (
                self._retry_executor.get_retry_stats() if self._retry_executor else {}
            ),
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
                "timestamp": health_result.timestamp.isoformat(),
            },
            "overall_status": self._health_monitor.get_health_status(),
            "recent_history": self._health_monitor.get_recent_history(),
        }

    @asynccontextmanager
    async def get_immediate_transaction(
        self,
        operation_name: str = "immediate_transaction",
        timeout_seconds: float = 10.0,
    ) -> AsyncIterator[aiosqlite.Connection]:
        """
        Get a database connection with BEGIN IMMEDIATE transaction and
        timeout.

        This ensures write locks are acquired immediately, preventing lock
        escalation failures that can occur with DEFERRED transactions.

        Args:
            operation_name: Name of the operation for monitoring
            timeout_seconds: Transaction timeout in seconds
        """
        async with self.get_write_connection_with_retry(operation_name) as conn:
            try:
                # Start immediate transaction with timeout
                await asyncio.wait_for(
                    conn.execute("BEGIN IMMEDIATE"), timeout=timeout_seconds
                )
                yield conn
                await conn.commit()
            except asyncio.TimeoutError:
                logger.warning(
                    (
                        f"Transaction timeout after {timeout_seconds}s for "
                        f"{operation_name}"
                    ),
                    extra={
                        "structured_data": {
                            "transaction_timeout": {
                                "operation": operation_name,
                                "timeout_seconds": timeout_seconds,
                            }
                        }
                    },
                )
                await conn.rollback()
                raise
            except Exception as e:
                logger.error(f"Transaction failed for {operation_name}: {e}")
                await conn.rollback()
                raise

    async def execute_transaction_with_retry(
        self,
        operation_func: Callable[[aiosqlite.Connection], Any],
        operation_name: str = "transaction_operation",
        max_retries: int = 3,
        timeout_seconds: float = 10.0,
    ) -> Any:
        """
        Execute a database operation within a transaction with automatic
        retry.

        Uses the new RetryExecutor for robust retry handling with proper
        error classification and exponential backoff.

        Args:
            operation_func: Async function that takes a connection and
                performs the operation
            operation_name: Name of the operation for logging
            max_retries: Maximum retry attempts (overrides default retry
                executor config)
            timeout_seconds: Transaction timeout in seconds

        Returns:
            Result from operation_func

        Example:
            async def my_operation(conn):
                await conn.execute("INSERT INTO ...", (...))
                return "success"

            result = await db.execute_transaction_with_retry(
                my_operation, "insert_data"
            )
        """

        async def execute_transaction() -> Any:
            """Inner function to execute transaction - retried by executor."""
            try:
                async with self.get_immediate_transaction(
                    operation_name, timeout_seconds
                ) as conn:
                    result = await operation_func(conn)

                # Record successful operation metrics
                if self._metrics_collector:
                    self._metrics_collector.record_operation(
                        operation_name,
                        timeout_seconds * 1000,  # Convert to ms
                        True,
                        len(self._connection_pool),
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
                            len(self._connection_pool),
                        )

                raise classified_error

        try:
            # Create a temporary retry executor with custom max_retries if different
            # from default
            if max_retries != self._retry_executor.config.max_attempts:
                from mcp_code_indexer.database.retry_executor import (
                    RetryConfig,
                    RetryExecutor,
                )

                temp_config = RetryConfig(
                    max_attempts=max_retries,
                    min_wait_seconds=self._retry_executor.config.min_wait_seconds,
                    max_wait_seconds=self._retry_executor.config.max_wait_seconds,
                    jitter_max_seconds=self._retry_executor.config.jitter_max_seconds,
                )
                temp_executor = RetryExecutor(temp_config)
                return await temp_executor.execute_with_retry(
                    execute_transaction, operation_name
                )
            else:
                return await self._retry_executor.execute_with_retry(
                    execute_transaction, operation_name
                )

        except DatabaseError:
            # Record failed operation metrics for final failure
            if self._metrics_collector:
                self._metrics_collector.record_operation(
                    operation_name,
                    timeout_seconds * 1000,
                    False,
                    len(self._connection_pool),
                )
            raise

    # Project operations

    async def create_project(self, project: Project) -> None:
        """Create a new project record."""
        async with self.get_write_connection_with_retry("create_project") as db:
            await db.execute(
                """
                INSERT INTO projects (id, name, aliases, created, last_accessed)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    project.id,
                    project.name,
                    json.dumps(project.aliases),
                    project.created,
                    project.last_accessed,
                ),
            )
            await db.commit()
            logger.debug(f"Created project: {project.id}")

    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            )
            row = await cursor.fetchone()

            if row:
                return Project(
                    id=row["id"],
                    name=row["name"],
                    aliases=json.loads(row["aliases"]),
                    created=datetime.fromisoformat(row["created"]),
                    last_accessed=datetime.fromisoformat(row["last_accessed"]),
                )
            return None

    async def find_matching_project(
        self, project_name: str, folder_path: Optional[str] = None
    ) -> Optional[Project]:
        """
        Find project by matching criteria.

        Args:
            project_name: Name of the project
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
                score += 2  # Name match is primary identifier
                match_factors.append("name")

            # Check folder path in aliases
            if folder_path and folder_path in project.aliases:
                score += 1
                match_factors.append("folder_path")

            # If we have a name match, it's a strong candidate
            if score >= 2:
                if score > best_score:
                    best_score = score
                    best_match = project
                    logger.info(
                        (
                            f"Match for project {project.name} "
                            f"(score: {score}, factors: {match_factors})"
                        )
                    )

        return best_match

    async def get_or_create_project(
        self, project_name: str, folder_path: str
    ) -> Project:
        """
        Get or create a project using intelligent matching.

        Args:
            project_name: Name of the project
            folder_path: Project folder path

        Returns:
            Existing or newly created project
        """
        # Try to find existing project
        project = await self.find_matching_project(project_name, folder_path)

        if project:
            # Update aliases if folder path not already included
            if folder_path not in project.aliases:
                project.aliases.append(folder_path)
                await self.update_project(project)
                logger.info(
                    f"Added folder path {folder_path} to project {project.name} aliases"
                )

            # Update access time
            await self.update_project_access_time(project.id)
            return project

        # Create new project
        import uuid

        from ..database.models import Project

        new_project = Project(
            id=str(uuid.uuid4()),
            name=project_name,
            aliases=[folder_path],
            created=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
        )

        await self.create_project(new_project)
        logger.info(f"Created new project: {new_project.name} ({new_project.id})")
        return new_project

    async def update_project_access_time(self, project_id: str) -> None:
        """Update the last accessed time for a project."""
        async with self.get_write_connection_with_retry(
            "update_project_access_time"
        ) as db:
            await db.execute(
                "UPDATE projects SET last_accessed = ? WHERE id = ?",
                (datetime.utcnow(), project_id),
            )
            await db.commit()

    async def update_project(self, project: Project) -> None:
        """Update an existing project record."""
        async with self.get_write_connection_with_retry("update_project") as db:
            await db.execute(
                """
                UPDATE projects
                SET name = ?, aliases = ?, last_accessed = ?
                WHERE id = ?
                """,
                (
                    project.name,
                    json.dumps(project.aliases),
                    project.last_accessed,
                    project.id,
                ),
            )
            await db.commit()
            logger.debug(f"Updated project: {project.id}")

    async def set_project_vector_mode(self, project_id: str, enabled: bool) -> None:
        """Set the vector_mode for a specific project."""
        async with self.get_write_connection_with_retry(
            "set_project_vector_mode"
        ) as db:
            await db.execute(
                "UPDATE projects SET vector_mode = ? WHERE id = ?",
                (int(enabled), project_id),
            )

            # Check if the project was actually updated
            cursor = await db.execute("SELECT changes()")
            changes = await cursor.fetchone()
            if changes[0] == 0:
                raise ValueError(f"Project not found: {project_id}")

            await db.commit()
            logger.debug(f"Set vector_mode={enabled} for project: {project_id}")

    async def get_all_projects(self) -> List[Project]:
        """Get all projects in the database."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                "SELECT id, name, aliases, created, last_accessed, COALESCE(vector_mode, 0) FROM projects"
            )
            rows = await cursor.fetchall()

            projects = []
            for row in rows:
                aliases = json.loads(row[2]) if row[2] else []
                project = Project(
                    id=row[0],
                    name=row[1],
                    aliases=aliases,
                    created=row[3],
                    last_accessed=row[4],
                    vector_mode=bool(row[5]),
                )
                projects.append(project)

            return projects

    async def get_vector_enabled_projects(self) -> List[Project]:
        """Get projects that have vector mode enabled."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                "SELECT id, name, aliases, created, last_accessed, vector_mode FROM projects WHERE vector_mode = 1"
            )
            rows = await cursor.fetchall()

            projects = []
            for row in rows:
                aliases = json.loads(row[2]) if row[2] else []
                project = Project(
                    id=row[0],
                    name=row[1],
                    aliases=aliases,
                    created=row[3],
                    last_accessed=row[4],
                    vector_mode=bool(row[5]),
                )
                projects.append(project)

            return projects

    # File description operations

    async def create_file_description(self, file_desc: FileDescription) -> None:
        """Create or update a file description."""
        async with self.get_write_connection_with_retry(
            "create_file_description"
        ) as db:
            await db.execute(
                """
                INSERT INTO file_descriptions
                (
                    project_id, file_path, description, file_hash, last_modified,
                    version, source_project_id, to_be_cleaned
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, file_path) DO UPDATE SET
                    description=excluded.description,
                    file_hash=excluded.file_hash,
                    last_modified=excluded.last_modified,
                    version=excluded.version,
                    source_project_id=excluded.source_project_id,
                    to_be_cleaned=excluded.to_be_cleaned
                """,
                (
                    file_desc.project_id,
                    file_desc.file_path,
                    file_desc.description,
                    file_desc.file_hash,
                    file_desc.last_modified,
                    file_desc.version,
                    file_desc.source_project_id,
                    file_desc.to_be_cleaned,
                ),
            )
            await db.commit()
            logger.debug(f"Saved file description: {file_desc.file_path}")

    async def get_file_description(
        self, project_id: str, file_path: str
    ) -> Optional[FileDescription]:
        """Get file description by project and path."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM file_descriptions
                WHERE project_id = ? AND file_path = ? AND to_be_cleaned IS NULL
                """,
                (project_id, file_path),
            )
            row = await cursor.fetchone()

            if row:
                return FileDescription(
                    id=row["id"],
                    project_id=row["project_id"],
                    file_path=row["file_path"],
                    description=row["description"],
                    file_hash=row["file_hash"],
                    last_modified=datetime.fromisoformat(row["last_modified"]),
                    version=row["version"],
                    source_project_id=row["source_project_id"],
                    to_be_cleaned=row["to_be_cleaned"],
                )
            return None

    async def get_all_file_descriptions(self, project_id: str) -> List[FileDescription]:
        """Get all file descriptions for a project."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT * FROM file_descriptions
                WHERE project_id = ? AND to_be_cleaned IS NULL
                ORDER BY file_path
                """,
                (project_id,),
            )
            rows = await cursor.fetchall()

            return [
                FileDescription(
                    id=row["id"],
                    project_id=row["project_id"],
                    file_path=row["file_path"],
                    description=row["description"],
                    file_hash=row["file_hash"],
                    last_modified=datetime.fromisoformat(row["last_modified"]),
                    version=row["version"],
                    source_project_id=row["source_project_id"],
                    to_be_cleaned=row["to_be_cleaned"],
                )
                for row in rows
            ]

    async def batch_create_file_descriptions(
        self, file_descriptions: List[FileDescription]
    ) -> None:
        """
        Batch create multiple file descriptions efficiently with optimized transactions.
        """
        if not file_descriptions:
            return

        async def batch_operation(conn: aiosqlite.Connection) -> None:
            data = [
                (
                    fd.project_id,
                    fd.file_path,
                    fd.description,
                    fd.file_hash,
                    fd.last_modified,
                    fd.version,
                    fd.source_project_id,
                    fd.to_be_cleaned,
                )
                for fd in file_descriptions
            ]

            await conn.executemany(
                """
                INSERT INTO file_descriptions
                (
                    project_id, file_path, description, file_hash, last_modified,
                    version, source_project_id, to_be_cleaned
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, file_path) DO UPDATE SET
                    description=excluded.description,
                    file_hash=excluded.file_hash,
                    last_modified=excluded.last_modified,
                    version=excluded.version,
                    source_project_id=excluded.source_project_id,
                    to_be_cleaned=excluded.to_be_cleaned
                """,
                data,
            )
            logger.debug(f"Batch created {len(file_descriptions)} file descriptions")

        await self.execute_transaction_with_retry(
            batch_operation,
            f"batch_create_file_descriptions_{len(file_descriptions)}_files",
            timeout_seconds=30.0,  # Longer timeout for batch operations
        )

    # Search operations

    async def search_file_descriptions(
        self, project_id: str, query: str, max_results: int = 20
    ) -> List[SearchResult]:
        """Search file descriptions using FTS5 with intelligent query preprocessing."""
        # Preprocess query for optimal FTS5 search
        preprocessed_query = preprocess_search_query(query)

        if not preprocessed_query:
            logger.debug(f"Empty query after preprocessing: '{query}'")
            return []

        logger.debug(f"Search query preprocessing: '{query}' -> '{preprocessed_query}'")

        async with self.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT
                    fd.project_id,
                    fd.file_path,
                    fd.description,
                    bm25(file_descriptions_fts) as rank
                FROM file_descriptions_fts
                JOIN file_descriptions fd ON fd.id = file_descriptions_fts.rowid
                WHERE file_descriptions_fts MATCH ?
                  AND fd.project_id = ?
                  AND fd.to_be_cleaned IS NULL
                ORDER BY bm25(file_descriptions_fts)
                LIMIT ?
                """,
                (preprocessed_query, project_id, max_results),
            )
            rows = await cursor.fetchall()

            return [
                SearchResult(
                    project_id=row["project_id"],
                    file_path=row["file_path"],
                    description=row["description"],
                    relevance_score=row["rank"],
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
                (cache_key, datetime.utcnow()),
            )
            row = await cursor.fetchone()
            return row["token_count"] if row else None

    async def cache_token_count(
        self, cache_key: str, token_count: int, ttl_hours: int = 24
    ) -> None:
        """Cache token count with TTL."""
        expires = datetime.utcnow() + timedelta(hours=ttl_hours)

        async with self.get_write_connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO token_cache (cache_key, token_count, expires)
                VALUES (?, ?, ?)
                """,
                (cache_key, token_count, expires),
            )
            await db.commit()

    async def cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        async with self.get_write_connection() as db:
            await db.execute(
                "DELETE FROM token_cache WHERE expires < ?", (datetime.utcnow(),)
            )
            await db.commit()

    # Utility operations

    async def get_file_count(self, project_id: str) -> int:
        """Get count of files in a project."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                (
                    "SELECT COUNT(*) as count FROM file_descriptions WHERE "
                    "project_id = ? AND to_be_cleaned IS NULL"
                ),
                (project_id,),
            )
            row = await cursor.fetchone()
            return row["count"] if row else 0

    # Project Overview operations

    async def create_project_overview(self, overview: ProjectOverview) -> None:
        """Create or update a project overview."""
        async with self.get_write_connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO project_overviews
                (project_id, overview, last_modified, total_files, total_tokens)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    overview.project_id,
                    overview.overview,
                    overview.last_modified,
                    overview.total_files,
                    overview.total_tokens,
                ),
            )
            await db.commit()
            logger.debug(f"Created/updated overview for project {overview.project_id}")

    async def get_project_overview(self, project_id: str) -> Optional[ProjectOverview]:
        """Get project overview by ID."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                "SELECT * FROM project_overviews WHERE project_id = ?", (project_id,)
            )
            row = await cursor.fetchone()

            if row:
                return ProjectOverview(
                    project_id=row["project_id"],
                    overview=row["overview"],
                    last_modified=datetime.fromisoformat(row["last_modified"]),
                    total_files=row["total_files"],
                    total_tokens=row["total_tokens"],
                )
            return None

    async def cleanup_missing_files(
        self, project_id: str, project_root: Path
    ) -> List[str]:
        """
        Mark descriptions for cleanup for files that no longer exist on disk.

        Args:
            project_id: Project identifier
            project_root: Path to project root directory

        Returns:
            List of file paths that were marked for cleanup
        """
        # 1. Get all active file paths (fast DB read)
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                (
                    "SELECT file_path FROM file_descriptions WHERE "
                    "project_id = ? AND to_be_cleaned IS NULL"
                ),
                (project_id,),
            )
            rows = await cursor.fetchall()
            file_paths = [row["file_path"] for row in rows]

        # 2. Check existence on disk (blocking IO - run in executor)
        def find_removed_files() -> List[str]:
            missing = []
            for file_path in file_paths:
                full_path = project_root / file_path
                if not full_path.exists():
                    missing.append(file_path)
            return missing

        loop = asyncio.get_running_loop()
        to_remove = await loop.run_in_executor(None, find_removed_files)

        # 3. Mark for cleanup (fast DB write)
        if to_remove:
            await self.cleanup_manager.mark_files_for_cleanup(project_id, to_remove)
            logger.info(
                f"Marked {len(to_remove)} missing files for cleanup from {project_id}"
            )

        return to_remove

    async def analyze_word_frequency(
        self, project_id: str, limit: int = 200
    ) -> WordFrequencyResult:
        """
        Analyze word frequency across all file descriptions for a project.

        Args:
            project_id: Project identifier
            limit: Maximum number of top terms to return

        Returns:
            WordFrequencyResult with top terms and statistics
        """
        import re
        from collections import Counter

        # Load stop words from bundled file
        stop_words_path = (
            Path(__file__).parent.parent / "data" / "stop_words_english.txt"
        )
        stop_words: set = set()

        if stop_words_path.exists():
            with open(stop_words_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Each line contains just the stop word
                    word = line.strip().lower()
                    if word:  # Skip empty lines
                        stop_words.add(word)

        # Add common programming keywords to stop words
        programming_keywords = {
            "if",
            "else",
            "for",
            "while",
            "do",
            "break",
            "continue",
            "return",
            "function",
            "class",
            "def",
            "var",
            "let",
            "const",
            "public",
            "private",
            "static",
            "async",
            "await",
            "import",
            "export",
            "from",
            "true",
            "false",
            "null",
            "undefined",
            "this",
            "that",
            "self",
            "super",
            "new",
            "delete",
        }
        stop_words.update(programming_keywords)

        # Get all descriptions for this project (fast DB read)
        async with self.get_connection() as db:
            cursor = await db.execute(
                (
                    "SELECT description FROM file_descriptions WHERE "
                    "project_id = ? AND to_be_cleaned IS NULL"
                ),
                (project_id,),
            )
            rows = await cursor.fetchall()
            descriptions = [row["description"] for row in rows]

        # Process word frequency in executor (CPU-bound work)
        def process_word_frequency() -> WordFrequencyResult:
            # Combine all descriptions
            all_text = " ".join(descriptions)

            # Tokenize and filter
            words = re.findall(r"\b[a-zA-Z]{2,}\b", all_text.lower())
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
                total_unique_terms=len(word_counts),
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, process_word_frequency)

    async def cleanup_empty_projects(self) -> int:
        """
        Remove projects that have no file descriptions and no project overview.

        Returns:
            Number of projects removed
        """
        async with self.get_write_connection() as db:
            # Find projects with no descriptions and no overview
            cursor = await db.execute(
                """
                SELECT p.id, p.name
                FROM projects p
                LEFT JOIN file_descriptions fd ON p.id = fd.project_id
                LEFT JOIN project_overviews po ON p.id = po.project_id
                WHERE fd.project_id IS NULL AND po.project_id IS NULL
            """
            )

            empty_projects = await cursor.fetchall()

            if not empty_projects:
                return 0

            removed_count = 0
            for project in empty_projects:
                project_id = project["id"]
                project_name = project["name"]

                # Remove from projects table (cascading will handle related data)
                await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
                removed_count += 1

                logger.info(f"Removed empty project: {project_name} (ID: {project_id})")

            await db.commit()
            return removed_count

    async def get_project_map_data(
        self, project_identifier: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get all data needed to generate a project map.

        Args:
            project_identifier: Project name or ID

        Returns:
            Dictionary containing project info, overview, and file descriptions
        """
        async with self.get_connection() as db:
            # Try to find project by ID first, then by name
            if len(project_identifier) == 36 and "-" in project_identifier:
                # Looks like a UUID
                cursor = await db.execute(
                    "SELECT * FROM projects WHERE id = ?", (project_identifier,)
                )
            else:
                # Search by name
                cursor = await db.execute(
                    "SELECT * FROM projects WHERE LOWER(name) = LOWER(?)",
                    (project_identifier,),
                )

            project_row = await cursor.fetchone()
            if not project_row:
                return None

            # Handle aliases JSON parsing
            project_dict = dict(project_row)
            if isinstance(project_dict["aliases"], str):
                import json

                project_dict["aliases"] = json.loads(project_dict["aliases"])

            project = Project(**project_dict)

            # Get project overview
            cursor = await db.execute(
                "SELECT * FROM project_overviews WHERE project_id = ?", (project.id,)
            )
            overview_row = await cursor.fetchone()
            project_overview = ProjectOverview(**overview_row) if overview_row else None

            # Get all file descriptions for this project
            cursor = await db.execute(
                """SELECT * FROM file_descriptions
                   WHERE project_id = ? AND to_be_cleaned IS NULL
                   ORDER BY file_path""",
                (project.id,),
            )
            file_rows = await cursor.fetchall()
            file_descriptions = [FileDescription(**row) for row in file_rows]

            return {
                "project": project,
                "overview": project_overview,
                "files": file_descriptions,
            }

    # IndexMeta operations
    async def create_index_meta(self, index_meta: IndexMeta) -> None:
        """Create or update index metadata for a project."""
        async with self.get_write_connection_with_retry("create_index_meta") as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO index_meta (
                    project_id, total_chunks, indexed_chunks, total_files, indexed_files,
                    last_sync, sync_status, error_message, queue_depth, processing_rate,
                    estimated_completion, metadata, created, last_modified
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    index_meta.project_id,
                    index_meta.total_chunks,
                    index_meta.indexed_chunks,
                    index_meta.total_files,
                    index_meta.indexed_files,
                    index_meta.last_sync,
                    index_meta.sync_status.value,
                    index_meta.error_message,
                    index_meta.queue_depth,
                    index_meta.processing_rate,
                    index_meta.estimated_completion,
                    json.dumps(index_meta.metadata),
                    index_meta.created,
                    index_meta.last_modified,
                ),
            )
            await db.commit()
            logger.debug(
                f"Created/updated index metadata for project: {index_meta.project_id}"
            )

    async def update_index_meta(self, index_meta: IndexMeta) -> None:
        """Update existing index metadata for a project."""
        async with self.get_write_connection_with_retry("update_index_meta") as db:
            await db.execute(
                """
                UPDATE index_meta
                SET total_chunks = ?, indexed_chunks = ?, total_files = ?, indexed_files = ?,
                    last_sync = ?, sync_status = ?, error_message = ?, queue_depth = ?,
                    processing_rate = ?, estimated_completion = ?, metadata = ?, last_modified = ?
                WHERE project_id = ?
                """,
                (
                    index_meta.total_chunks,
                    index_meta.indexed_chunks,
                    index_meta.total_files,
                    index_meta.indexed_files,
                    index_meta.last_sync,
                    index_meta.sync_status.value,
                    index_meta.error_message,
                    index_meta.queue_depth,
                    index_meta.processing_rate,
                    index_meta.estimated_completion,
                    json.dumps(index_meta.metadata),
                    index_meta.last_modified,
                    index_meta.project_id,
                ),
            )

            # Check if the project was actually updated
            cursor = await db.execute("SELECT changes()")
            changes = await cursor.fetchone()
            if changes[0] == 0:
                raise ValueError(
                    f"Index metadata not found for project: {index_meta.project_id}"
                )

            await db.commit()
            logger.debug(f"Updated index metadata for project: {index_meta.project_id}")

    async def get_index_meta(self, project_id: str) -> Optional[IndexMeta]:
        """Retrieve index metadata for a project."""
        async with self.get_connection() as db:
            cursor = await db.execute(
                "SELECT * FROM index_meta WHERE project_id = ?", (project_id,)
            )
            row = await cursor.fetchone()

            if row:
                # Convert row to dict for easier field access
                row_dict = dict(row)

                # Parse JSON metadata field
                metadata = (
                    json.loads(row_dict["metadata"]) if row_dict["metadata"] else {}
                )

                # Parse datetime fields
                created = (
                    datetime.fromisoformat(row_dict["created"])
                    if row_dict["created"]
                    else datetime.utcnow()
                )
                last_modified = (
                    datetime.fromisoformat(row_dict["last_modified"])
                    if row_dict["last_modified"]
                    else datetime.utcnow()
                )
                last_sync = (
                    datetime.fromisoformat(row_dict["last_sync"])
                    if row_dict["last_sync"]
                    else None
                )
                estimated_completion = (
                    datetime.fromisoformat(row_dict["estimated_completion"])
                    if row_dict["estimated_completion"]
                    else None
                )

                return IndexMeta(
                    id=row_dict["id"],
                    project_id=row_dict["project_id"],
                    total_chunks=row_dict["total_chunks"],
                    indexed_chunks=row_dict["indexed_chunks"],
                    total_files=row_dict["total_files"],
                    indexed_files=row_dict["indexed_files"],
                    last_sync=last_sync,
                    sync_status=row_dict["sync_status"],
                    error_message=row_dict["error_message"],
                    queue_depth=row_dict["queue_depth"],
                    processing_rate=row_dict["processing_rate"],
                    estimated_completion=estimated_completion,
                    metadata=metadata,
                    created=created,
                    last_modified=last_modified,
                )
            return None

    async def get_or_create_index_meta(self, project_id: str, **kwargs) -> IndexMeta:
        """
        Get existing index metadata or create new one with default values.
        
        Args:
            project_id: Project identifier
            **kwargs: Optional fields to override defaults when creating new metadata
            
        Returns:
            IndexMeta object (existing or newly created)
        """
        # Try to get existing metadata first
        existing_meta = await self.get_index_meta(project_id)
        if existing_meta:
            return existing_meta
        
        # Create new metadata with defaults, allowing kwargs to override
        default_metadata = {
            "project_id": project_id,
            "total_chunks": 0,
            "indexed_chunks": 0,
            "total_files": 0,
            "indexed_files": 0,
            "last_sync": None,
            "sync_status": SyncStatus.PENDING,
            "error_message": None,
            "queue_depth": 0,
            "processing_rate": 0.0,
            "estimated_completion": None,
            "metadata": {},
        }
        
        # Override defaults with provided kwargs
        default_metadata.update(kwargs)
        
        # Create the IndexMeta object
        new_meta = IndexMeta(**default_metadata)
        
        # Store it in the database
        await self.create_index_meta(new_meta)
        
        # Return the created metadata (fetch it back to get the assigned ID)
        result = await self.get_index_meta(project_id)
        if result is None:
            raise DatabaseError(f"Failed to create index metadata for project: {project_id}")
        
        return result

    # Cleanup operations

    @property
    def cleanup_manager(self) -> CleanupManager:
        """Get the cleanup manager instance."""
        if self._cleanup_manager is None:
            self._cleanup_manager = CleanupManager(self, retention_months=6)
        return self._cleanup_manager

    async def mark_file_for_cleanup(self, project_id: str, file_path: str) -> bool:
        """Mark a file for cleanup. Convenience method."""
        return await self.cleanup_manager.mark_file_for_cleanup(project_id, file_path)

    async def perform_cleanup(self, project_id: Optional[str] = None) -> int:
        """Perform cleanup of old records. Convenience method."""
        return await self.cleanup_manager.perform_cleanup(project_id)
