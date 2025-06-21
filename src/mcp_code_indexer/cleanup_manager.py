"""
Cleanup Manager for MCP Code Indexer.

Handles soft deletion and retention policies for file descriptions
that are marked for cleanup. Provides periodic cleanup operations
and manual cleanup methods.
"""

import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class CleanupManager:
    """
    Manages cleanup operations for file descriptions with retention policies.

    Handles soft deletion by updating to_be_cleaned timestamps and provides
    periodic cleanup to permanently remove old records after the retention period.
    """

    def __init__(self, db_manager, retention_months: int = 6):
        """
        Initialize cleanup manager.

        Args:
            db_manager: DatabaseManager instance
            retention_months: Number of months to retain records before
                permanent deletion
        """
        self.db_manager = db_manager
        self.retention_months = retention_months

    async def mark_file_for_cleanup(self, project_id: str, file_path: str) -> bool:
        """
        Mark a specific file for cleanup by setting to_be_cleaned timestamp.

        Args:
            project_id: Project identifier
            file_path: Path to file to mark for cleanup

        Returns:
            True if file was marked, False if file not found
        """
        cleanup_timestamp = int(time.time())

        async with self.db_manager.get_write_connection_with_retry(
            "mark_file_for_cleanup"
        ) as db:
            cursor = await db.execute(
                """
                UPDATE file_descriptions
                SET to_be_cleaned = ?
                WHERE project_id = ? AND file_path = ? AND to_be_cleaned IS NULL
                """,
                (cleanup_timestamp, project_id, file_path),
            )
            await db.commit()

            # Check if any rows were affected
            return cursor.rowcount > 0

    async def mark_files_for_cleanup(
        self, project_id: str, file_paths: List[str]
    ) -> int:
        """
        Mark multiple files for cleanup in a batch operation.

        Args:
            project_id: Project identifier
            file_paths: List of file paths to mark for cleanup

        Returns:
            Number of files marked for cleanup
        """
        if not file_paths:
            return 0

        cleanup_timestamp = int(time.time())

        async def batch_operation(conn):
            data = [(cleanup_timestamp, project_id, path) for path in file_paths]
            cursor = await conn.executemany(
                """
                UPDATE file_descriptions
                SET to_be_cleaned = ?
                WHERE project_id = ? AND file_path = ? AND to_be_cleaned IS NULL
                """,
                data,
            )
            return cursor.rowcount

        marked_count = await self.db_manager.execute_transaction_with_retry(
            batch_operation,
            f"mark_files_for_cleanup_{len(file_paths)}_files",
            timeout_seconds=30.0,
        )

        logger.info(f"Marked {marked_count} files for cleanup in project {project_id}")
        return marked_count

    async def restore_file_from_cleanup(self, project_id: str, file_path: str) -> bool:
        """
        Restore a file from cleanup by clearing its to_be_cleaned timestamp.

        Args:
            project_id: Project identifier
            file_path: Path to file to restore

        Returns:
            True if file was restored, False if file not found
        """
        async with self.db_manager.get_write_connection_with_retry(
            "restore_file_from_cleanup"
        ) as db:
            cursor = await db.execute(
                """
                UPDATE file_descriptions
                SET to_be_cleaned = NULL
                WHERE project_id = ? AND file_path = ? AND to_be_cleaned IS NOT NULL
                """,
                (project_id, file_path),
            )
            await db.commit()

            return cursor.rowcount > 0

    async def get_files_to_be_cleaned(self, project_id: str) -> List[dict]:
        """
        Get list of files marked for cleanup in a project.

        Args:
            project_id: Project identifier

        Returns:
            List of dictionaries with file_path and to_be_cleaned timestamp
        """
        async with self.db_manager.get_connection() as db:
            cursor = await db.execute(
                """
                SELECT file_path, to_be_cleaned
                FROM file_descriptions
                WHERE project_id = ? AND to_be_cleaned IS NOT NULL
                ORDER BY to_be_cleaned DESC, file_path
                """,
                (project_id,),
            )
            rows = await cursor.fetchall()

            return [
                {
                    "file_path": row["file_path"],
                    "marked_for_cleanup": row["to_be_cleaned"],
                    "marked_date": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(row["to_be_cleaned"])
                    ),
                }
                for row in rows
            ]

    async def perform_cleanup(self, project_id: Optional[str] = None) -> int:
        """
        Permanently delete records that exceed the retention period.

        Args:
            project_id: If specified, only clean up this project. Otherwise
                clean all projects.

        Returns:
            Number of records permanently deleted
        """
        # Calculate cutoff timestamp (retention_months ago)
        cutoff_seconds = (
            self.retention_months * 30 * 24 * 60 * 60
        )  # Approximate months to seconds
        cutoff_timestamp = int(time.time()) - cutoff_seconds

        async def cleanup_operation(conn):
            if project_id:
                cursor = await conn.execute(
                    """
                    DELETE FROM file_descriptions
                    WHERE project_id = ? AND to_be_cleaned IS NOT NULL
                        AND to_be_cleaned < ?
                    """,
                    (project_id, cutoff_timestamp),
                )
            else:
                cursor = await conn.execute(
                    """
                    DELETE FROM file_descriptions
                    WHERE to_be_cleaned IS NOT NULL AND to_be_cleaned < ?
                    """,
                    (cutoff_timestamp,),
                )

            return cursor.rowcount

        deleted_count = await self.db_manager.execute_transaction_with_retry(
            cleanup_operation,
            f"perform_cleanup_{project_id or 'all_projects'}",
            timeout_seconds=60.0,
        )

        if deleted_count > 0:
            scope = f"project {project_id}" if project_id else "all projects"
            logger.info(f"Permanently deleted {deleted_count} old records from {scope}")

        return deleted_count

    async def get_cleanup_stats(self, project_id: Optional[str] = None) -> dict:
        """
        Get statistics about cleanup state.

        Args:
            project_id: If specified, get stats for this project only

        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_seconds = self.retention_months * 30 * 24 * 60 * 60
        cutoff_timestamp = int(time.time()) - cutoff_seconds

        async with self.db_manager.get_connection() as db:
            if project_id:
                base_where = "WHERE project_id = ?"
                params = (project_id,)
            else:
                base_where = ""
                params = ()

            # Active files
            cursor = await db.execute(
                (
                    f"SELECT COUNT(*) FROM file_descriptions {base_where} "
                    "AND to_be_cleaned IS NULL"
                ),
                params,
            )
            active_count = (await cursor.fetchone())[0]

            # Files marked for cleanup
            cursor = await db.execute(
                (
                    f"SELECT COUNT(*) FROM file_descriptions {base_where} "
                    "AND to_be_cleaned IS NOT NULL"
                ),
                params,
            )
            marked_count = (await cursor.fetchone())[0]

            # Files eligible for permanent deletion
            if project_id:
                cursor = await db.execute(
                    (
                        "SELECT COUNT(*) FROM file_descriptions WHERE project_id = ? "
                        "AND to_be_cleaned IS NOT NULL AND to_be_cleaned < ?"
                    ),
                    (project_id, cutoff_timestamp),
                )
            else:
                cursor = await db.execute(
                    (
                        "SELECT COUNT(*) FROM file_descriptions WHERE "
                        "to_be_cleaned IS NOT NULL AND to_be_cleaned < ?"
                    ),
                    (cutoff_timestamp,),
                )
            eligible_for_deletion = (await cursor.fetchone())[0]

            return {
                "active_files": active_count,
                "marked_for_cleanup": marked_count,
                "eligible_for_deletion": eligible_for_deletion,
                "retention_months": self.retention_months,
                "cutoff_date": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(cutoff_timestamp)
                ),
            }
