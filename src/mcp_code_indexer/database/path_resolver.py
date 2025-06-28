"""
Database path resolution for local vs global databases.

This module provides logic to determine whether to use a local project database
or the global database based on the presence of .code-index.db files.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DatabasePathResolver:
    """
    Resolves database paths, determining whether to use local or global databases.
    
    Local databases are stored as tracker.db in .code-index folders.
    If a .code-index folder exists, it takes precedence over the global database.
    """
    
    def __init__(self, global_db_path: Path):
        """
        Initialize the path resolver with the global database path.
        
        Args:
            global_db_path: Path to the global database file
        """
        self.global_db_path = global_db_path
        
    def resolve_database_path(self, folder_path: Optional[str] = None) -> Path:
        """
        Resolve which database to use based on folder path.
        
        Args:
            folder_path: Project folder path to check for local database
            
        Returns:
            Path to the database file to use
        """
        if not folder_path:
            logger.debug("No folder path provided, using global database")
            return self.global_db_path
            
        try:
            folder_path_obj = Path(folder_path).resolve()
            local_db_folder = folder_path_obj / ".code-index"
            
            if local_db_folder.exists() and local_db_folder.is_dir():
                local_db_path = local_db_folder / "tracker.db"
                logger.debug(f"Found local database folder: {local_db_path}")
                return local_db_path
            else:
                logger.debug(f"No local database folder found at {local_db_folder}, using global database")
                return self.global_db_path
                
        except (OSError, ValueError) as e:
            logger.warning(f"Error resolving folder path '{folder_path}': {e}")
            return self.global_db_path
    
    def is_local_database(self, folder_path: Optional[str] = None) -> bool:
        """
        Check if a local database folder exists for the given folder path.
        
        Args:
            folder_path: Project folder path to check
            
        Returns:
            True if a local database folder exists, False otherwise
        """
        if not folder_path:
            return False
            
        try:
            folder_path_obj = Path(folder_path).resolve()
            local_db_folder = folder_path_obj / ".code-index"
            return local_db_folder.exists() and local_db_folder.is_dir()
        except (OSError, ValueError):
            return False
    
    def get_local_database_path(self, folder_path: str) -> Path:
        """
        Get the local database path for a folder (whether it exists or not).
        
        Args:
            folder_path: Project folder path
            
        Returns:
            Path where the local database would be located
        """
        return Path(folder_path).resolve() / ".code-index" / "tracker.db"
    
    def get_local_database_folder(self, folder_path: str) -> Path:
        """
        Get the local database folder for a project folder.
        
        Args:
            folder_path: Project folder path
            
        Returns:
            Path where the local database folder would be located
        """
        return Path(folder_path).resolve() / ".code-index"
