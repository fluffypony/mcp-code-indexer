"""
Real-time file system monitoring using watchdog.

Provides efficient file change detection with debouncing and pattern filtering
for the vector mode indexing pipeline.
"""

import asyncio
import logging
from .utils import _write_debug_log
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

try:
    from watchdog.observers import Observer, ObserverType
    from watchdog.events import FileSystemEventHandler, FileSystemEvent

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None
    FileSystemEvent = None

from .change_detector import ChangeDetector, FileChange, ChangeType
from .merkle_tree import MerkleTree

logger = logging.getLogger(__name__)


class BaseFileWatcher(ABC):
    """Abstract base class for file watchers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the file watcher."""
        pass

    @abstractmethod
    def add_change_callback(self, callback: Callable[[FileChange], None]) -> None:
        """Add a callback to be called when files change."""
        pass

    @abstractmethod
    def start_watching(self) -> None:
        """Start watching for file changes."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get watcher statistics."""
        pass


class VectorModeEventHandler(FileSystemEventHandler):
    """Event handler for file system changes in vector mode."""

    def __init__(
        self,
        change_detector: ChangeDetector,
        merkle_tree: Optional[MerkleTree] = None,
        callback: Optional[Callable[[FileChange], None]] = None,
    ):
        """
        Initialize event handler.

        Args:
            change_detector: Change detection processor
            merkle_tree: Optional Merkle tree for efficient change tracking
            callback: Optional callback for change notifications
        """
        super().__init__()
        self.change_detector = change_detector
        self.merkle_tree = merkle_tree
        self.callback = callback

        # Debouncing state
        self.pending_events: Dict[str, FileSystemEvent] = {}
        self.debounce_tasks: Dict[str, asyncio.Task] = {}

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Handle any file system event."""
        try:
            if event.is_directory:
                return  # Skip directory events for now

            try:
                asyncio.create_task(self._handle_event_async(event))
            except RuntimeError:
                # No event loop running, handle synchronously
                self._handle_event_sync(event)

        except Exception as e:
            # Critical: Catch ALL exceptions to prevent observer thread crash
            logger.error(
                f"Unhandled exception in event handler for {event.src_path}: {e}",
                exc_info=True,
            )

    def _handle_event_sync(self, event: FileSystemEvent) -> None:
        """Handle event synchronously."""
        path = Path(event.src_path)

        # Process the change
        change = self.change_detector.process_fs_event(
            event_type=event.event_type,
            path=path,
            old_path=Path(event.dest_path) if hasattr(event, "dest_path") else None,
        )
        if change:
            # Update Merkle tree if available
            if self.merkle_tree:
                try:
                    self.merkle_tree.update_file(change.path)
                except Exception as e:
                    logger.warning(
                        f"Failed to update Merkle tree for {change.path}: {e}"
                    )
            # Call callback if provided
            if self.callback:
                try:
                    self.callback(change)
                except Exception as e:
                    logger.error(f"Callback failed for change {change.path}: {e}")

    async def _handle_event_async(self, event: FileSystemEvent) -> None:
        """Handle event asynchronously with debouncing."""
        file_path = event.src_path

        # Cancel existing debounce task for this file
        if file_path in self.debounce_tasks:
            self.debounce_tasks[file_path].cancel()

        # Store pending event
        self.pending_events[file_path] = event

        # Create new debounce task
        self.debounce_tasks[file_path] = asyncio.create_task(
            self._process_after_debounce(file_path)
        )

    async def _process_after_debounce(self, file_path: str) -> None:
        """Process event after debounce delay."""
        # Wait for debounce interval
        await asyncio.sleep(0.1)  # 100ms debounce

        # Get pending event
        event = self.pending_events.pop(file_path, None)
        if event:
            self._handle_event_sync(event)

        # Clean up task reference
        self.debounce_tasks.pop(file_path, None)


class FileWatcher(BaseFileWatcher):
    """
    Real-time file system watcher for vector mode.

    Monitors file changes and integrates with change detection and Merkle tree
    systems for efficient vector index updates.
    """

    def __init__(
        self,
        project_root: Path,
        project_id: str,
        ignore_patterns: Optional[List[str]] = None,
        debounce_interval: float = 0.1,
        enable_merkle_tree: bool = True,
    ):
        """
        Initialize file watcher.

        Args:
            project_root: Root directory to watch
            project_id: Project identifier
            ignore_patterns: Patterns to ignore
            debounce_interval: Debounce interval in seconds
            enable_merkle_tree: Whether to use Merkle tree for change tracking
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError("watchdog library is required for file monitoring")

        self.project_root = Path(project_root).resolve()
        self.project_id = project_id
        self.ignore_patterns = ignore_patterns
        self.debounce_interval = debounce_interval

        # Initialize components
        self.change_detector = ChangeDetector(
            project_root=self.project_root,
            ignore_patterns=ignore_patterns,
            debounce_interval=debounce_interval,
        )

        self.merkle_tree: Optional[MerkleTree] = None
        if enable_merkle_tree:
            self.merkle_tree = MerkleTree(self.project_root, project_id)

        # Watchdog components
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[VectorModeEventHandler] = None

        # State
        self.is_watching = False
        self.change_callbacks: List[Callable[[FileChange], None]] = []

        # Thread pool for intensive operations
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="file_watcher"
        )

    def add_change_callback(self, callback: Callable[[FileChange], None]) -> None:
        """Add a callback to be called when files change."""
        self.change_callbacks.append(callback)

    def remove_change_callback(self, callback: Callable[[FileChange], None]) -> None:
        """Remove a change callback."""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)

    def _on_change(self, change: FileChange) -> None:
        """Handle a file change by notifying all callbacks."""
        for callback in self.change_callbacks:
            try:
                _write_debug_log(
                    f"File change detected: {change.path} ({change.change_type.value})"
                )
                callback(change)
            except Exception as e:
                logger.error(f"Change callback failed: {e}")

    async def initialize(self) -> None:
        """Initialize the file watcher (build Merkle tree, etc.)."""
        logger.info(f"Initializing file watcher for {self.project_root}")

        # Build Merkle tree in thread pool to avoid blocking
        if self.merkle_tree:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, self.merkle_tree.build_tree, self.ignore_patterns
            )
            logger.info("Merkle tree built successfully")

    def start_watching(self) -> None:
        """Start watching for file changes."""
        if self.is_watching:
            logger.warning("File watcher is already running")
            return

        if not WATCHDOG_AVAILABLE:
            logger.error("Cannot start file watching: watchdog not available")
            return

        logger.info(f"Starting file watcher for {self.project_root}")

        # Create event handler
        self.event_handler = VectorModeEventHandler(
            change_detector=self.change_detector,
            merkle_tree=self.merkle_tree,
            callback=self._on_change,
        )

        # Create and start observer
        try:
            self.observer = Observer()
            self.observer.schedule(
                self.event_handler, str(self.project_root), recursive=True
            )
            self.observer.start()
        except Exception as e:
            logger.error(f"Failed to start file observer: {e}", exc_info=True)
            # Clean up on failure
            if self.observer:
                try:
                    self.observer.stop()
                    self.observer = None
                except Exception:
                    pass
            self.event_handler = None
            self.is_watching = False
            raise

        self.is_watching = True
        logger.info("File watcher started successfully")

    def stop_watching(self) -> None:
        """Stop watching for file changes."""
        if not self.is_watching:
            return

        logger.info("Stopping file watcher")

        if self.observer:
            try:
                self.observer.stop()
                self.observer.join()
            except Exception as e:
                logger.error(f"Error stopping file observer: {e}", exc_info=True)
            finally:
                self.observer = None

        self.event_handler = None
        self.is_watching = False

        logger.info("File watcher stopped")

    def get_recent_changes(
        self,
        limit: Optional[int] = None,
        change_types: Optional[List[ChangeType]] = None,
    ) -> List[FileChange]:
        """Get recent file changes."""
        return self.change_detector.get_recent_changes(limit, change_types)

    def get_changed_files(self, since: Optional[str] = None) -> List[str]:
        """Get list of files that have changed."""
        from datetime import datetime

        since_dt = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError:
                logger.warning(f"Invalid timestamp format: {since}")

        # Get changes from detector
        changed_files = list(self.change_detector.get_changed_files(since_dt))

        # Add changes from Merkle tree if available
        if self.merkle_tree:
            merkle_changes = self.merkle_tree.get_changed_files(since_dt)
            changed_files.extend(merkle_changes)

        return list(set(changed_files))  # Remove duplicates

    def force_scan(self) -> int:
        """Force a full scan and return number of changes detected."""
        logger.info("Forcing full file system scan")

        if self.merkle_tree:
            # Rebuild Merkle tree
            self.merkle_tree.build_tree(self.ignore_patterns)

            # Get changed files
            changed_files = self.merkle_tree.get_changed_files()

            # Process changes through detector
            for file_path in changed_files:
                full_path = self.project_root / file_path
                change = self.change_detector.process_fs_event(
                    event_type="modified", path=full_path
                )

                if change and self.change_callbacks:
                    self._on_change(change)

            return len(changed_files)

        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get watcher statistics."""
        stats = {
            "is_watching": self.is_watching,
            "project_root": str(self.project_root),
            "project_id": self.project_id,
            "change_detector_stats": self.change_detector.get_stats().__dict__,
            "callbacks_registered": len(self.change_callbacks),
        }

        if self.merkle_tree:
            stats["merkle_tree"] = self.merkle_tree.get_tree_summary()

        return stats

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_watching()

        if self.executor:
            self.executor.shutdown(wait=True)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Fallback implementation for when watchdog is not available
class PollingFileWatcher(BaseFileWatcher):
    """
    Fallback file watcher using polling instead of OS events.

    Used when watchdog is not available or on systems that don't support
    efficient file system monitoring.
    """

    def __init__(
        self, project_root: Path, project_id: str, poll_interval: float = 5.0, **kwargs
    ):
        """Initialize polling file watcher."""
        self.project_root = Path(project_root).resolve()
        self.project_id = project_id
        self.poll_interval = poll_interval

        self.change_detector = ChangeDetector(project_root=self.project_root, **kwargs)
        self.merkle_tree = MerkleTree(self.project_root, project_id)

        self.is_watching = False
        self.poll_task: Optional[asyncio.Task] = None
        self.change_callbacks: List[Callable[[FileChange], None]] = []

    def add_change_callback(self, callback: Callable[[FileChange], None]) -> None:
        """Add a callback to be called when files change."""
        self.change_callbacks.append(callback)

    async def initialize(self) -> None:
        """Initialize the polling watcher."""
        self.merkle_tree.build_tree()

    def start_watching(self) -> None:
        """Start polling for changes."""
        if self.is_watching:
            return

        self.is_watching = True
        self.poll_task = asyncio.create_task(self._poll_loop())

    def stop_watching(self) -> None:
        """Stop polling for changes."""
        self.is_watching = False
        if self.poll_task:
            self.poll_task.cancel()

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self.is_watching:
            try:
                # Force scan for changes
                changed_files = self.merkle_tree.get_changed_files()

                for file_path in changed_files:
                    full_path = self.project_root / file_path
                    change = self.change_detector.process_fs_event(
                        event_type="modified", path=full_path
                    )

                    if change:
                        for callback in self.change_callbacks:
                            callback(change)

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(self.poll_interval)

    def get_stats(self) -> Dict[str, Any]:
        """Get watcher statistics."""
        return {
            "project_root": str(self.project_root),
            "project_id": self.project_id,
            "is_watching": self.is_watching,
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_watching()


def create_file_watcher(
    project_root: Path, project_id: str, use_polling: bool = False, **kwargs
) -> BaseFileWatcher:
    """
    Create appropriate file watcher based on availability.

    Args:
        project_root: Root directory to watch
        project_id: Project identifier
        use_polling: Force use of polling watcher
        **kwargs: Additional arguments for watcher

    Returns:
        FileWatcher or PollingFileWatcher instance
    """
    if use_polling or not WATCHDOG_AVAILABLE:
        logger.info("Using polling file watcher")
        watcher = PollingFileWatcher(project_root, project_id, **kwargs)
    else:
        logger.info("Using real-time file watcher")
        watcher = FileWatcher(project_root, project_id, **kwargs)

    return watcher
