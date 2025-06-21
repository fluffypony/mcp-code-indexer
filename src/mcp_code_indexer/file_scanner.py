"""
File discovery and gitignore integration for the MCP Code Indexer.

This module provides functionality to scan project directories for files
while respecting .gitignore patterns and common ignore patterns. It enables
efficient discovery of files that need description tracking.
"""

import logging
from pathlib import Path
from typing import List, Set, Optional, Generator
import fnmatch

try:
    from gitignore_parser import parse_gitignore
except ImportError:
    parse_gitignore = None

logger = logging.getLogger(__name__)


# Default patterns to ignore even without .gitignore
DEFAULT_IGNORE_PATTERNS = [
    # Version control
    ".git/",
    ".svn/",
    ".hg/",
    # Dependencies and packages
    "node_modules/",
    "venv/",
    ".venv/",
    "env/",
    ".env/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    # Build artifacts
    "build/",
    "dist/",
    "target/",
    "out/",
    "bin/",
    "obj/",
    "*.o",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    # IDE and editor files
    ".vscode/",
    ".idea/",
    ".vs/",
    "*.swp",
    "*.swo",
    "*~",
    ".DS_Store",
    "Thumbs.db",
    # Testing and coverage
    "coverage/",
    "htmlcov/",
    ".pytest_cache/",
    ".coverage",
    "*.coverage",
    # Documentation builds
    "_build/",
    "docs/_build/",
    "site/",
    # Logs and temporary files
    "*.log",
    "*.tmp",
    "*.temp",
    "*.cache",
    # Package files
    "*.tar.gz",
    "*.zip",
    "*.rar",
    "*.7z",
    # Lock files
    "package-lock.json",
    "yarn.lock",
    "Pipfile.lock",
    "poetry.lock",
]

# File extensions commonly ignored for code indexing
IGNORED_EXTENSIONS = {
    # Binary files
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".o",
    ".obj",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    # Documents
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    # Media
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    # Fonts
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".eot",
}


class FileScanner:
    """
    Handles file discovery with gitignore and pattern-based filtering.

    Provides methods to scan directories while respecting .gitignore files
    and default ignore patterns to identify files suitable for description tracking.
    """

    def __init__(self, project_root: Path):
        """
        Initialize file scanner for a project.

        Args:
            project_root: Root directory of the project to scan
        """
        self.project_root = Path(project_root).resolve()
        self._gitignore_cache: dict = {}
        self._load_gitignore_patterns()

    def _load_gitignore_patterns(self) -> None:
        """Load and cache gitignore patterns from the project."""
        self._gitignore_cache.clear()

        if parse_gitignore is None:
            logger.warning(
                "gitignore_parser not available, using default patterns only"
            )
            return

        # Look for .gitignore files in the project hierarchy
        current_path = self.project_root

        while current_path != current_path.parent:
            gitignore_path = current_path / ".gitignore"

            if gitignore_path.exists():
                try:
                    gitignore_func = parse_gitignore(gitignore_path)
                    self._gitignore_cache[str(current_path)] = gitignore_func
                    logger.debug(f"Loaded .gitignore from {gitignore_path}")
                except Exception as e:
                    logger.warning(f"Failed to parse {gitignore_path}: {e}")

            current_path = current_path.parent

    def _is_ignored_by_gitignore(self, file_path: Path) -> bool:
        """Check if a file is ignored by any .gitignore file."""
        if not self._gitignore_cache:
            return False

        # Check against all loaded .gitignore patterns
        for base_path, gitignore_func in self._gitignore_cache.items():
            try:
                # gitignore_parser expects absolute paths
                if gitignore_func(str(file_path.resolve())):
                    return True
            except Exception as e:
                logger.debug(f"Error checking gitignore pattern: {e}")
                continue

        return False

    def _is_ignored_by_default_patterns(self, file_path: Path) -> bool:
        """Check if a file matches default ignore patterns."""
        try:
            resolved_file = file_path.resolve()
            resolved_root = self.project_root.resolve()
            rel_path = resolved_file.relative_to(resolved_root)
            rel_path_str = str(rel_path)
        except ValueError:
            return True

        for pattern in DEFAULT_IGNORE_PATTERNS:
            # Handle directory patterns
            if pattern.endswith("/"):
                pattern_no_slash = pattern.rstrip("/")
                # Check if any parent directory matches
                for parent in rel_path.parents:
                    if fnmatch.fnmatch(parent.name, pattern_no_slash):
                        return True
                # Check the file's parent directory
                if fnmatch.fnmatch(rel_path.parent.name, pattern_no_slash):
                    return True
            else:
                # Handle file patterns
                if fnmatch.fnmatch(rel_path_str, pattern):
                    return True
                if fnmatch.fnmatch(file_path.name, pattern):
                    return True

        return False

    def _is_ignored_by_extension(self, file_path: Path) -> bool:
        """Check if a file has an ignored extension."""
        return file_path.suffix.lower() in IGNORED_EXTENSIONS

    def should_ignore_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be ignored.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file should be ignored
        """
        # Check if it's a file (not directory)
        if not file_path.is_file():
            return True

        # Check file extension
        if self._is_ignored_by_extension(file_path):
            return True

        # Check default patterns
        if self._is_ignored_by_default_patterns(file_path):
            return True

        # Check gitignore patterns
        if self._is_ignored_by_gitignore(file_path):
            return True

        return False

    def scan_directory(self, max_files: Optional[int] = None) -> List[Path]:
        """
        Scan the project directory for trackable files.

        Args:
            max_files: Maximum number of files to return (None for no limit)

        Returns:
            List of file paths that should be tracked
        """
        files = []

        try:
            for file_path in self._walk_directory():
                if not self.should_ignore_file(file_path):
                    files.append(file_path)

                    if max_files and len(files) >= max_files:
                        logger.info(f"Reached max_files limit of {max_files}")
                        break

        except Exception as e:
            logger.error(f"Error scanning directory {self.project_root}: {e}")

        # Sort files for consistent ordering
        files.sort()

        logger.info(f"Found {len(files)} trackable files in {self.project_root}")
        return files

    def _walk_directory(self) -> Generator[Path, None, None]:
        """Walk through all files in the project directory."""
        try:
            for item in self.project_root.rglob("*"):
                if item.is_file():
                    yield item
        except PermissionError as e:
            logger.warning(f"Permission denied accessing {e.filename}")
        except Exception as e:
            logger.error(f"Error walking directory: {e}")

    def get_relative_path(self, file_path: Path) -> str:
        """
        Get relative path from project root.

        Args:
            file_path: Absolute path to file

        Returns:
            Relative path string from project root
        """
        try:
            # Resolve both paths to handle symlinks and .. properly
            resolved_file = file_path.resolve()
            resolved_root = self.project_root.resolve()
            return str(resolved_file.relative_to(resolved_root))
        except ValueError:
            # File is outside project root, return absolute path
            return str(file_path)

    def find_missing_files(self, existing_paths: Set[str]) -> List[Path]:
        """
        Find files that exist on disk but aren't in the existing paths set.

        Args:
            existing_paths: Set of relative file paths that already have descriptions

        Returns:
            List of file paths that are missing descriptions
        """
        all_files = self.scan_directory()
        missing_files = []

        for file_path in all_files:
            rel_path = self.get_relative_path(file_path)
            if rel_path not in existing_paths:
                missing_files.append(file_path)

        logger.info(f"Found {len(missing_files)} files missing descriptions")
        return missing_files

    def is_valid_project_directory(self) -> bool:
        """
        Check if the project root is a valid directory for scanning.

        Returns:
            True if the directory exists and is accessible
        """
        try:
            return (
                self.project_root.exists()
                and self.project_root.is_dir()
                and self.project_root.stat().st_mode & 0o444  # Readable
            )
        except (OSError, PermissionError):
            return False

    def get_project_stats(self) -> dict:
        """
        Get statistics about the project directory.

        Returns:
            Dictionary with project statistics for trackable files only
        """
        stats = {
            "total_files": 0,
            "trackable_files": 0,
            "ignored_files": 0,
            "largest_file_size": 0,
            "file_extensions": {},
        }

        try:
            all_files_count = 0
            for file_path in self._walk_directory():
                all_files_count += 1

                # Check if trackable first
                if self.should_ignore_file(file_path):
                    stats["ignored_files"] += 1
                    continue

                # Only process trackable files for detailed stats
                stats["trackable_files"] += 1

                # Track file size
                try:
                    file_size = file_path.stat().st_size
                    stats["largest_file_size"] = max(
                        stats["largest_file_size"], file_size
                    )
                except OSError:
                    pass

                # Track extensions for trackable files only
                ext = file_path.suffix.lower()
                stats["file_extensions"][ext] = stats["file_extensions"].get(ext, 0) + 1

            # Total files is just trackable files
            stats["total_files"] = stats["trackable_files"]

        except Exception as e:
            logger.error(f"Error getting project stats: {e}")

        return stats
