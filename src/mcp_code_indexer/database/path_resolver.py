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
    
    Local databases are stored as .code-index.db in project folders.
    If a local database file exists, it takes precedence over the global database.
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
            local_db_path = folder_path_obj / ".code-index.db"
            
            if local_db_path.exists():
                logger.debug(f"Found local database: {local_db_path}")
                return local_db_path
            else:
                logger.debug(f"No local database found at {local_db_path}, using global database")
                return self.global_db_path
                
        except (OSError, ValueError) as e:
            logger.warning(f"Error resolving folder path '{folder_path}': {e}")
            return self.global_db_path
    
    def is_local_database(self, folder_path: Optional[str] = None) -> bool:
        """
        Check if a local database exists for the given folder path.
        
        Args:
            folder_path: Project folder path to check
            
        Returns:
            True if a local database exists, False otherwise
        """
        if not folder_path:
            return False
            
        try:
            folder_path_obj = Path(folder_path).resolve()
            local_db_path = folder_path_obj / ".code-index.db"
            return local_db_path.exists()
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
        return Path(folder_path).resolve() / ".code-index.db"
    
    def is_empty_database_file(self, db_path: Path) -> bool:
        """
        Check if a database file is empty (0 bytes).
        
        Args:
            db_path: Path to the database file
            
        Returns:
            True if the file exists and is empty, False otherwise
        """
        try:
            return db_path.exists() and db_path.stat().st_size == 0
        except (OSError, ValueError):
            return False
    
    def should_initialize_local_database(self, folder_path: str) -> bool:
        """
        Check if a local database should be initialized.
        
        This returns True if .code-index.db exists and is empty.
        
        Args:
            folder_path: Project folder path
            
        Returns:
            True if local database should be initialized, False otherwise
        """
        local_db_path = self.get_local_database_path(folder_path)
        return self.is_empty_database_file(local_db_path)
