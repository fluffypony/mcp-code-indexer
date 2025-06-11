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

import aiosqlite

from mcp_code_indexer.database.models import (
    Project, FileDescription, MergeConflict, SearchResult,
    CodebaseSizeInfo
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations with async support.
    
    Provides high-level operations for projects, file descriptions, search,
    and caching with proper transaction management and error handling.
    """
    
    def __init__(self, db_path: Path, pool_size: int = 5):
        """Initialize database manager with path to SQLite database."""
        self.db_path = db_path
        self.pool_size = pool_size
        self._connection_pool: List[aiosqlite.Connection] = []
        self._pool_lock = None  # Will be initialized in async context
        
    async def initialize(self) -> None:
        """Initialize database schema and configuration."""
        import asyncio
        
        # Initialize pool lock
        self._pool_lock = asyncio.Lock()
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Apply migrations in order
        migrations_dir = Path(__file__).parent.parent.parent / "migrations"
        migration_files = sorted(migrations_dir.glob("*.sql"))
        
        async with aiosqlite.connect(self.db_path) as db:
            # Enable row factory for easier data access
            db.row_factory = aiosqlite.Row
            
            # Apply each migration
            for migration_file in migration_files:
                logger.info(f"Applying migration: {migration_file.name}")
                with open(migration_file, 'r') as f:
                    migration_sql = f.read()
                
                await db.executescript(migration_sql)
                await db.commit()
            
        logger.info(f"Database initialized at {self.db_path} with {len(migration_files)} migrations")
    
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
            
            # Apply performance settings to new connections
            await conn.execute("PRAGMA busy_timeout = 30000")  # 30 second timeout
            await conn.execute("PRAGMA synchronous = NORMAL")   # Balanced durability/performance
            await conn.execute("PRAGMA cache_size = -64000")    # 64MB cache
            await conn.execute("PRAGMA temp_store = MEMORY")    # Use memory for temp tables
        
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
        """Close all connections in the pool."""
        if self._pool_lock:
            async with self._pool_lock:
                for conn in self._connection_pool:
                    await conn.close()
                self._connection_pool.clear()
    
    # Project operations
    
    async def create_project(self, project: Project) -> None:
        """Create a new project record."""
        async with self.get_connection() as db:
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
    
    async def update_project_access_time(self, project_id: str) -> None:
        """Update the last accessed time for a project."""
        async with self.get_connection() as db:
            await db.execute(
                "UPDATE projects SET last_accessed = ? WHERE id = ?",
                (datetime.utcnow(), project_id)
            )
            await db.commit()
    
    async def update_project(self, project: Project) -> None:
        """Update an existing project record."""
        async with self.get_connection() as db:
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
    
    # File description operations
    
    async def create_file_description(self, file_desc: FileDescription) -> None:
        """Create or update a file description."""
        async with self.get_connection() as db:
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
        """Batch create multiple file descriptions efficiently."""
        if not file_descriptions:
            return
            
        async with self.get_connection() as db:
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
            
            await db.executemany(
                """
                INSERT OR REPLACE INTO file_descriptions 
                (project_id, branch, file_path, description, file_hash, last_modified, version, source_project_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data
            )
            await db.commit()
            logger.debug(f"Batch created {len(file_descriptions)} file descriptions")
    
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
                    fts.rank
                FROM file_descriptions_fts fts
                JOIN file_descriptions fd ON fd.rowid = fts.rowid
                WHERE fts MATCH ? 
                  AND fd.project_id = ? 
                  AND fd.branch = ?
                ORDER BY fts.rank
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
        
        async with self.get_connection() as db:
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
        async with self.get_connection() as db:
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
