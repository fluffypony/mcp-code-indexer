"""
Utility functions for vector mode operations.

Common utilities shared across vector mode components for file processing,
path handling, and pattern matching operations.
"""

import re
import fnmatch
from pathlib import Path
from typing import List


def should_ignore_path(
    path: Path, project_root: Path, ignore_patterns: List[str]
) -> bool:
    """
    Check if a path should be ignored based on ignore patterns.

    Args:
        path: Path to check for ignoring
        project_root: Root path of the project
        ignore_patterns: List of glob patterns to match against

    Returns:
        True if the path should be ignored, False otherwise

    Raises:
        ValueError: If path is not relative to project_root
    """
    try:
        relative_path = path.relative_to(project_root)
        path_str = str(relative_path)

        # Compile ignore patterns for matching
        compiled_patterns = [fnmatch.translate(pattern) for pattern in ignore_patterns]

        # Check if path matches any ignore pattern
        for pattern in compiled_patterns:
            if re.match(pattern, path_str):
                return True

        return False

    except ValueError:
        # Path is not relative to project root - should be ignored
        return True



