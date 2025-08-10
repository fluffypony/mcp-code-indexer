"""
Change detection utilities for file system monitoring.

Provides high-level change detection and classification for the vector mode
file monitoring system.
"""

import logging
from enum import Enum
from typing import List, Dict, Set, Optional, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

class ChangeType(str, Enum):
    """Types of file system changes."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"

class FileChange(NamedTuple):
    """Represents a file system change."""
    path: str
    change_type: ChangeType
    timestamp: datetime
    old_path: Optional[str] = None  # For moves
    size: Optional[int] = None
    hash: Optional[str] = None

@dataclass
class ChangeStats:
    """Statistics about detected changes."""
    total_changes: int = 0
    creates: int = 0
    modifications: int = 0
    deletions: int = 0
    moves: int = 0
    start_time: Optional[datetime] = None
    last_change: Optional[datetime] = None

class ChangeDetector:
    """
    High-level change detection and classification.
    
    Processes raw file system events and provides structured change information
    for the vector indexing pipeline.
    """
    
    def __init__(
        self,
        project_root: Path,
        ignore_patterns: Optional[List[str]] = None,
        debounce_interval: float = 0.1,
    ):
        """
        Initialize change detector.
        
        Args:
            project_root: Root directory to monitor
            ignore_patterns: Patterns to ignore (glob-style)
            debounce_interval: Minimum time between processing same file
        """
        self.project_root = Path(project_root).resolve()
        self.ignore_patterns = ignore_patterns or [
            "*.log", "*.tmp", "*~", ".git/*", "__pycache__/*", 
            "node_modules/*", "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db"
        ]
        self.debounce_interval = debounce_interval
        
        # Change tracking
        self.recent_changes: List[FileChange] = []
        self.pending_changes: Dict[str, FileChange] = {}
        self.last_change_time: Dict[str, datetime] = {}
        
        # Statistics
        self.stats = ChangeStats(start_time=datetime.utcnow())
        
        # Compile ignore patterns for performance
        import fnmatch
        self._compiled_patterns = [
            fnmatch.translate(pattern) for pattern in self.ignore_patterns
        ]
    
    def should_ignore_path(self, path: Path) -> bool:
        """Check if a path should be ignored based on patterns."""
        try:
            relative_path = path.relative_to(self.project_root)
            path_str = str(relative_path)
            
            import re
            for pattern in self._compiled_patterns:
                if re.match(pattern, path_str):
                    return True
            
            return False
            
        except ValueError:
            # Path is not relative to project root
            return True
    
    def _should_debounce(self, file_path: str) -> bool:
        """Check if change should be debounced."""
        now = datetime.utcnow()
        
        if file_path in self.last_change_time:
            elapsed = (now - self.last_change_time[file_path]).total_seconds()
            if elapsed < self.debounce_interval:
                return True
        
        self.last_change_time[file_path] = now
        return False
    
    def _get_file_info(self, path: Path) -> Dict[str, Optional[int]]:
        """Get file information (size, etc.)."""
        try:
            if path.exists() and path.is_file():
                stat = path.stat()
                return {"size": stat.st_size}
            else:
                return {"size": None}
        except (OSError, PermissionError):
            return {"size": None}
    
    def _classify_change(
        self,
        path: Path,
        event_type: str,
        old_path: Optional[Path] = None
    ) -> Optional[FileChange]:
        """Classify a file system event into a structured change."""
        
        # Convert to relative path
        try:
            relative_path = str(path.relative_to(self.project_root))
        except ValueError:
            # Path outside project root
            return None
        
        # Check if should be ignored
        if self.should_ignore_path(path):
            logger.debug(f"Ignoring change to {relative_path} (matches ignore pattern)")
            return None
        
        # Check debouncing
        if self._should_debounce(relative_path):
            logger.debug(f"Debouncing change to {relative_path}")
            return None
        
        # Get file info
        file_info = self._get_file_info(path)
        
        # Map event types to change types
        if event_type in ["created", "added"]:
            change_type = ChangeType.CREATED
        elif event_type in ["modified", "changed"]:
            change_type = ChangeType.MODIFIED
        elif event_type in ["deleted", "removed"]:
            change_type = ChangeType.DELETED
        elif event_type in ["moved", "renamed"]:
            change_type = ChangeType.MOVED
        else:
            logger.warning(f"Unknown event type: {event_type}")
            return None
        
        # Create change object
        old_relative_path = None
        if old_path:
            try:
                old_relative_path = str(old_path.relative_to(self.project_root))
            except ValueError:
                pass
        
        change = FileChange(
            path=relative_path,
            change_type=change_type,
            timestamp=datetime.utcnow(),
            old_path=old_relative_path,
            size=file_info.get("size"),
            hash=None  # Will be computed later if needed
        )
        
        return change
    
    def process_fs_event(
        self,
        event_type: str,
        path: Path,
        old_path: Optional[Path] = None
    ) -> Optional[FileChange]:
        """
        Process a file system event and return structured change.
        
        Args:
            event_type: Type of event (created, modified, deleted, moved)
            path: Path that changed
            old_path: Old path (for moves)
            
        Returns:
            FileChange object or None if ignored
        """
        change = self._classify_change(path, event_type, old_path)
        
        if change:
            self.recent_changes.append(change)
            
            # Update statistics
            self.stats.total_changes += 1
            self.stats.last_change = change.timestamp
            
            if change.change_type == ChangeType.CREATED:
                self.stats.creates += 1
            elif change.change_type == ChangeType.MODIFIED:
                self.stats.modifications += 1
            elif change.change_type == ChangeType.DELETED:
                self.stats.deletions += 1
            elif change.change_type == ChangeType.MOVED:
                self.stats.moves += 1
            
            logger.info(f"Detected change: {change.change_type.value} {change.path}")
        
        return change
    
    def get_recent_changes(
        self,
        limit: Optional[int] = None,
        change_types: Optional[List[ChangeType]] = None
    ) -> List[FileChange]:
        """
        Get recent changes with optional filtering.
        
        Args:
            limit: Maximum number of changes to return
            change_types: Filter by change types
            
        Returns:
            List of recent changes
        """
        changes = self.recent_changes
        
        # Filter by change types
        if change_types:
            changes = [c for c in changes if c.change_type in change_types]
        
        # Sort by timestamp (most recent first)
        changes = sorted(changes, key=lambda c: c.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            changes = changes[:limit]
        
        return changes
    
    def clear_recent_changes(self) -> int:
        """Clear recent changes and return count cleared."""
        count = len(self.recent_changes)
        self.recent_changes.clear()
        return count
    
    def get_changes_since(self, since: datetime) -> List[FileChange]:
        """Get all changes since a specific timestamp."""
        return [
            change for change in self.recent_changes
            if change.timestamp >= since
        ]
    
    def get_stats(self) -> ChangeStats:
        """Get change detection statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset change detection statistics."""
        self.stats = ChangeStats(start_time=datetime.utcnow())
    
    def get_changed_files(self, since: Optional[datetime] = None) -> Set[str]:
        """Get set of file paths that have changed."""
        changes = self.recent_changes
        
        if since:
            changes = [c for c in changes if c.timestamp >= since]
        
        # Collect unique file paths
        changed_files = set()
        for change in changes:
            changed_files.add(change.path)
            if change.old_path:  # For moves
                changed_files.add(change.old_path)
        
        return changed_files
    
    def is_code_file(self, path: str) -> bool:
        """Check if a file is likely a code file."""
        code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
            '.clj', '.cljs', '.hs', '.ml', '.fs', '.ex', '.exs', '.cr',
            '.dart', '.lua', '.pl', '.sh', '.bash', '.zsh', '.fish',
            '.sql', '.r', '.m', '.mm', '.vim', '.el', '.lisp', '.scm'
        }
        
        return Path(path).suffix.lower() in code_extensions
    
    def get_code_changes(self, since: Optional[datetime] = None) -> List[FileChange]:
        """Get changes to code files only."""
        changes = self.get_recent_changes()
        
        if since:
            changes = [c for c in changes if c.timestamp >= since]
        
        return [c for c in changes if self.is_code_file(c.path)]
