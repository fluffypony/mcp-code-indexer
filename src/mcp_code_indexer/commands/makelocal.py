"""
Command to migrate project data from global to local database.

This module provides functionality to extract project data from the global database
and create a local database in a project folder.
"""

import logging
from pathlib import Path
from typing import List, Optional

from mcp_code_indexer.database.database import DatabaseManager
from mcp_code_indexer.database.database_factory import DatabaseFactory
from mcp_code_indexer.database.models import Project, FileDescription, ProjectOverview

logger = logging.getLogger(__name__)


class MakeLocalCommand:
    """
    Command to migrate project data from global to local database.

    Extracts all project data, file descriptions, and project overviews
    from the global database and creates a local database in the specified folder.
    """

    def __init__(self, db_factory: DatabaseFactory):
        """
        Initialize the make local command.

        Args:
            db_factory: Database factory for creating database managers
        """
        self.db_factory = db_factory

    async def execute(
        self, folder_path: str, project_name: Optional[str] = None
    ) -> dict:
        """
        Execute the make local command.

        Args:
            folder_path: Path to the project folder where local DB will be created
            project_name: Optional project name to migrate (if None, tries to find by folder)

        Returns:
            Dictionary with operation results
        """
        folder_path_obj = Path(folder_path).resolve()

        if not folder_path_obj.exists():
            raise ValueError(f"Folder path does not exist: {folder_path}")

        if not folder_path_obj.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        # Get local database folder and path
        local_db_folder = self.db_factory.get_path_resolver().get_local_database_folder(
            folder_path
        )
        local_db_path = self.db_factory.get_path_resolver().get_local_database_path(
            folder_path
        )

        # Check if local database already exists and has data
        if (
            local_db_folder.exists()
            and local_db_path.exists()
            and local_db_path.stat().st_size > 0
        ):
            # Check if it actually has project data (not just schema)
            from sqlite3 import connect

            try:
                with connect(local_db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM projects")
                    project_count = cursor.fetchone()[0]
                    if project_count > 0:
                        raise ValueError(
                            f"Local database already contains {project_count} project(s): {local_db_path}"
                        )
            except Exception:
                # If we can't check, assume it has data to be safe
                raise ValueError(f"Local database already exists: {local_db_path}")

        # Get global database manager
        global_db_manager = await self.db_factory.get_database_manager()

        # Find the project to migrate
        project = await self._find_project_to_migrate(
            global_db_manager, folder_path, project_name
        )
        if not project:
            if project_name:
                raise ValueError(
                    f"Project '{project_name}' not found in global database"
                )
            else:
                raise ValueError(f"No project found for folder path: {folder_path}")

        logger.info(f"Found project to migrate: {project.name} (ID: {project.id})")

        # Get all project data
        file_descriptions = await global_db_manager.get_all_file_descriptions(
            project.id
        )
        project_overview = await global_db_manager.get_project_overview(project.id)

        logger.info(f"Found {len(file_descriptions)} file descriptions to migrate")
        if project_overview:
            logger.info("Found project overview to migrate")

        # Create local database folder (this ensures it exists)
        local_db_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created local database folder: {local_db_folder}")

        # Create local database manager (this will initialize schema)
        local_db_manager = await self.db_factory.get_database_manager(
            str(folder_path_obj)
        )

        # For local databases, we'll create a project with a machine-independent approach
        # We'll store the current folder path in aliases for reference, but the project
        # will be found by being the single project in the local database

        # Migrate data
        await self._migrate_project_data(
            local_db_manager, project, file_descriptions, project_overview
        )

        # Remove data from global database
        await self._remove_from_global_database(global_db_manager, project.id)

        return {
            "success": True,
            "project_name": project.name,
            "project_id": project.id,
            "local_database_path": str(local_db_path),
            "local_database_folder": str(local_db_folder),
            "migrated_files": len(file_descriptions),
            "migrated_overview": project_overview is not None,
        }

    async def _find_project_to_migrate(
        self,
        global_db_manager: DatabaseManager,
        folder_path: str,
        project_name: Optional[str],
    ) -> Optional[Project]:
        """
        Find the project to migrate from the global database.

        Args:
            global_db_manager: Global database manager
            folder_path: Project folder path
            project_name: Optional project name

        Returns:
            Project to migrate or None if not found
        """
        all_projects = await global_db_manager.get_all_projects()

        if project_name:
            # Search by name
            normalized_name = project_name.lower()
            for project in all_projects:
                if project.name.lower() == normalized_name:
                    return project
        else:
            # Search by folder path in aliases
            for project in all_projects:
                if folder_path in project.aliases:
                    return project

        return None

    async def _migrate_project_data(
        self,
        local_db_manager: DatabaseManager,
        project: Project,
        file_descriptions: List[FileDescription],
        project_overview: Optional[ProjectOverview],
    ) -> None:
        """
        Migrate project data to the local database.

        For local databases, we update the project aliases to include the current
        folder path since local database projects are found by being the single
        project in the database rather than by path matching.

        Args:
            local_db_manager: Local database manager
            project: Project to migrate
            file_descriptions: File descriptions to migrate
            project_overview: Project overview to migrate (if any)
        """
        # Update project aliases to include current folder path for reference
        # Note: This will be machine-specific but that's OK for local databases

        # Create project in local database
        await local_db_manager.create_project(project)
        logger.info(f"Created project in local database: {project.name}")

        # Migrate file descriptions
        if file_descriptions:
            await local_db_manager.batch_create_file_descriptions(file_descriptions)
            logger.info(f"Migrated {len(file_descriptions)} file descriptions")

        # Migrate project overview
        if project_overview:
            await local_db_manager.create_project_overview(project_overview)
            logger.info("Migrated project overview")

    async def _remove_from_global_database(
        self, global_db_manager: DatabaseManager, project_id: str
    ) -> None:
        """
        Remove project data from the global database.

        Args:
            global_db_manager: Global database manager
            project_id: Project ID to remove
        """
        # Remove file descriptions
        async with global_db_manager.get_write_connection_with_retry(
            "remove_project_files"
        ) as db:
            await db.execute(
                "DELETE FROM file_descriptions WHERE project_id = ?", (project_id,)
            )
            await db.commit()

        # Remove project overview
        async with global_db_manager.get_write_connection_with_retry(
            "remove_project_overview"
        ) as db:
            await db.execute(
                "DELETE FROM project_overviews WHERE project_id = ?", (project_id,)
            )
            await db.commit()

        # Remove project
        async with global_db_manager.get_write_connection_with_retry(
            "remove_project"
        ) as db:
            await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            await db.commit()

        logger.info(f"Removed project data from global database: {project_id}")
