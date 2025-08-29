"""
Vector Mode Daemon.

Runs as a background process to monitor file changes and maintain vector indexes.
Handles embedding generation, change detection, and vector database synchronization.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import time

from ..database.database import DatabaseManager
from ..database.models import Project
from .config import VectorConfig, load_vector_config
from .monitoring.file_watcher import create_file_watcher, FileWatcher
from .providers.voyage_client import VoyageClient, create_voyage_client
from .providers.turbopuffer_client import create_turbopuffer_client
from .services.embedding_service import EmbeddingService
from .services.vector_storage_service import VectorStorageService

from .monitoring.change_detector import FileChange, ChangeType
from .chunking.ast_chunker import ASTChunker, CodeChunk
from .types import (
    ScanProjectTask,
    VectorDaemonTaskType,
    ProcessFileChangeTask,
)

logger = logging.getLogger(__name__)


class VectorDaemon:
    """
    Background daemon for vector mode operations.

    Monitors file changes, generates embeddings, and maintains vector indexes
    for all projects with vector mode enabled.
    """

    def __init__(
        self,
        config: VectorConfig,
        db_manager: DatabaseManager,
        cache_dir: Path,
    ):
        """Initialize vector daemon."""
        self.config = config
        self.db_manager = db_manager
        self.cache_dir = cache_dir
        self.is_running = False

        # Process tracking
        self.monitored_projects: Set[str] = set()
        self.processing_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.max_queue_size
        )
        self.workers: list[asyncio.Task] = []
        self.monitor_tasks: list[asyncio.Task] = []

        # File watcher management
        self.file_watchers: Dict[str, FileWatcher] = {}
        self.watcher_locks: Dict[str, asyncio.Lock] = {}

        # Statistics
        self.stats = {
            "start_time": time.time(),
            "files_processed": 0,
            "embeddings_generated": 0,
            "errors_count": 0,
            "last_activity": time.time(),
        }

        # Initialize VoyageClient and EmbeddingService for embedding generation  
        self._voyage_client = create_voyage_client(self.config)
        self._embedding_service = EmbeddingService(self._voyage_client, self.config)

        # Initialize TurbopufferClient and VectorStorageService for vector storage
        self._turbopuffer_client = create_turbopuffer_client(self.config)
        self._vector_storage_service = VectorStorageService(self._turbopuffer_client, self.config)

        # Signal handling is delegated to the parent process

    def _on_file_change(self, project_name: str) -> callable:
        """Create a non-blocking change callback for a specific project."""

        def callback(change: FileChange) -> None:
            """Non-blocking callback that queues file change processing."""
            try:
                # Create file change processing task
                task_item: ProcessFileChangeTask = {
                    "type": VectorDaemonTaskType.PROCESS_FILE_CHANGE,
                    "project_name": project_name,
                    "change": change,
                    "timestamp": time.time(),
                }

                # Put task in processing queue (non-blocking)
                try:
                    self.processing_queue.put_nowait(task_item)
                except asyncio.QueueFull:
                    logger.warning(
                        f"Processing queue full, dropping file change event for {change.path}"
                    )
            except Exception as e:
                logger.error(f"Error queueing file change task: {e}")

        return callback

    async def start(self) -> None:
        """Start the vector daemon."""
        if self.is_running:
            logger.warning("Daemon is already running")
            return

        self.is_running = True

        logger.info(
            "Starting vector daemon",
            extra={
                "structured_data": {
                    "config": {
                        "worker_count": self.config.worker_count,
                        "batch_size": self.config.batch_size,
                        "poll_interval": self.config.daemon_poll_interval,
                    }
                }
            },
        )

        try:
            # Start worker tasks
            for i in range(self.config.worker_count):
                worker = asyncio.create_task(self._worker(f"worker-{i}"))
                self.workers.append(worker)

            # Start monitoring tasks
            monitor_task = asyncio.create_task(self._monitor_projects())
            stats_task = asyncio.create_task(self._stats_reporter())
            self.monitor_tasks.extend([monitor_task, stats_task])

            # Wait for shutdown signal
            await self._run_until_shutdown()

        except Exception as e:
            logger.error(f"Daemon error: {e}", exc_info=True)
            self.stats["errors_count"] += 1
        finally:
            await self._cleanup()

    async def _run_until_shutdown(self) -> None:
        """Run daemon until shutdown is requested."""
        # Wait indefinitely until task is cancelled by parent process
        try:
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("Vector daemon shutdown requested")
            raise

    async def _get_project_monitoring_status(self) -> Dict[str, List[Project]]:
        """
        Get projects categorized by monitoring status.

        Returns:
            Dict with 'monitored' and 'unmonitored' keys containing project lists
        """
        # Get all projects with vector mode enabled
        vector_enabled_projects = await self.db_manager.get_vector_enabled_projects()

        # Filter projects that should be monitored (have valid aliases)
        monitorable_projects = [
            project
            for project in vector_enabled_projects
            if project.aliases and project.vector_mode
        ]

        # Determine which projects to monitor (not currently monitored)
        projects_to_monitor = [
            project
            for project in monitorable_projects
            if project.name not in self.monitored_projects
        ]

        # Determine which projects to unmonitor (currently monitored but no longer should be)
        monitorable_names = {project.name for project in monitorable_projects}
        projects_to_unmonitor = []

        # Get full project data for unmonitoring
        all_projects = await self.db_manager.get_all_projects()
        for project in all_projects:
            if (
                project.name in self.monitored_projects
                and project.name not in monitorable_names
            ):
                projects_to_unmonitor.append(project)

        return {
            "monitored": projects_to_monitor,
            "unmonitored": projects_to_unmonitor,
        }

    async def _monitor_projects(self) -> None:
        """Monitor projects for vector indexing requirements."""
        logger.info("Starting project monitoring")

        while self.is_running:
            try:
                # Get project monitoring status
                monitoring_status = await self._get_project_monitoring_status()
                # Add new projects to monitoring
                for project in monitoring_status["monitored"]:
                    logger.info(f"Adding project to monitoring: {project.name}")
                    self.monitored_projects.add(project.name)

                    # Use first alias as folder path
                    folder_path = project.aliases[0]

                    # Queue initial indexing task
                    await self._queue_project_scan(project.name, folder_path)

                # Remove projects from monitoring
                for project in monitoring_status["unmonitored"]:
                    logger.info(f"Removing project from monitoring: {project.name}")
                    self.monitored_projects.discard(project.name)

                await asyncio.sleep(self.config.daemon_poll_interval)

            except asyncio.CancelledError:
                logger.info("Project monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in project monitoring: {e}")
                self.stats["errors_count"] += 1
                await asyncio.sleep(5.0)  # Back off on error

    async def _queue_project_scan(self, project_name: str, folder_path: str) -> None:
        """Queue a project for scanning and indexing."""
        task: ScanProjectTask = {
            "type": VectorDaemonTaskType.SCAN_PROJECT,
            "project_name": project_name,
            "folder_path": folder_path,
            "timestamp": time.time(),
        }

        try:
            await self.processing_queue.put(task)
            logger.debug(f"Queued project scan: {project_name}")
        except asyncio.QueueFull:
            logger.warning(
                f"Processing queue full, dropping scan task for {project_name}"
            )

    async def _worker(self, worker_id: str) -> None:
        """Worker task to process queued items."""
        logger.info(f"Starting worker: {worker_id}")

        while self.is_running:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(
                        self.processing_queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the task
                await self._process_task(task, worker_id)
                self.stats["last_activity"] = time.time()

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self.stats["errors_count"] += 1
                await asyncio.sleep(1.0)  # Brief pause on error

    async def _process_task(self, task: dict, worker_id: str) -> None:
        """Process a queued task."""
        task_type = task.get("type")

        if task_type == VectorDaemonTaskType.SCAN_PROJECT:
            await self._process_project_scan(task, worker_id)
        elif task_type == VectorDaemonTaskType.PROCESS_FILE_CHANGE:
            await self._process_file_change_task(task, worker_id)
        else:
            logger.warning(f"Unknown task type: {task_type}")

    async def _process_file_change_task(
        self, task: ProcessFileChangeTask, worker_id: str
    ) -> None:
        """Process a file change task."""
        project_name: str = task["project_name"]
        change: FileChange = task["change"]
        logger.info(
            f"Worker {worker_id}: File change detected for project {project_name}: {change.path} ({change.change_type.value})",
            extra={
                "structured_data": {
                    "worker_id": worker_id,
                    "project_name": project_name,
                    "file_path": str(change.path),
                    "change_type": change.change_type.value,
                    "timestamp": change.timestamp.isoformat(),
                    "size": change.size,
                }
            },
        )

        try:
            # Skip deleted files - only process created/modified files
            if change.change_type == ChangeType.DELETED:
                # TODO: handle deletions in vector DB
                logger.debug(f"Worker {worker_id}: Skipping deleted file {change.path}")
                return

            # Initialize ASTChunker with default settings
            chunker = ASTChunker(
                max_chunk_size=1500,
                min_chunk_size=50,
                enable_redaction=True,
                enable_optimization=True,
            )

            # Read and chunk the file
            try:
                chunks = chunker.chunk_file(str(change.path))
                chunk_count = len(chunks)

                # Only process files that actually produced chunks
                if chunk_count == 0:
                    logger.debug(
                        f"Worker {worker_id}: No chunks produced for {change.path}"
                    )
                    return

                # Log chunking results (to be replaced with embedding generation later)
                chunk_types = {}
                redacted_count = 0

                for chunk in chunks:
                    chunk_type = chunk.chunk_type.value
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    if chunk.redacted:
                        redacted_count += 1

                logger.info(
                    f"Worker {worker_id}: Chunked {change.path} into {chunk_count} chunks",
                    extra={
                        "structured_data": {
                            "worker_id": worker_id,
                            "project_name": project_name,
                            "file_path": str(change.path),
                            "chunk_count": chunk_count,
                            "chunk_types": chunk_types,
                            "redacted_chunks": redacted_count,
                            "file_size": change.size,
                        }
                    },
                )

                # Log detailed debug information
                sample_chunks = [
                    f"    [{i}] {chunk.chunk_type.value} - {chunk.name or 'unnamed'} "
                    f"(lines {chunk.start_line}-{chunk.end_line}, "
                    f"{len(chunk.content)} chars, redacted: {chunk.redacted})"
                    for i, chunk in enumerate(chunks[:3])  # Show first 3 chunks
                ]
                logger.debug(
                    f"Worker {worker_id}: File {change.path} chunking details: "
                    f"Total chunks: {chunk_count}, "
                    f"Chunk types: {chunk_types}, "
                    f"Redacted chunks: {redacted_count}, "
                    f"Sample chunks: {', '.join(sample_chunks)}"
                )

                # Generate and store embeddings for chunks
                embeddings = await self._generate_embeddings(
                    chunks, project_name, change.path
                )
                await self._store_embeddings(embeddings, chunks, project_name, change.path)

                # Only increment stats for successfully chunked files
                self.stats["files_processed"] += 1
                self.stats["last_activity"] = time.time()

            except Exception as read_error:
                logger.error(
                    f"Worker {worker_id}: Failed to read/chunk file {change.path}: {read_error}"
                )
                self.stats["errors_count"] += 1
                return

        except Exception as e:
            logger.error(
                f"Worker {worker_id}: Error processing file change {change.path}: {e}"
            )
            self.stats["errors_count"] += 1

    async def _process_project_scan(self, task: dict, worker_id: str) -> None:
        """Process a project scan task."""
        project_name = task["project_name"]
        folder_path = task["folder_path"]

        logger.debug(f"Worker {worker_id} processing project: {project_name}")

        try:
            # Ensure we have a lock for this project
            if project_name not in self.watcher_locks:
                self.watcher_locks[project_name] = asyncio.Lock()

            # Use project-specific lock to prevent race conditions
            async with self.watcher_locks[project_name]:
                # Check if file watcher already exists for this project
                if project_name not in self.file_watchers:
                    logger.info(
                        f"Initializing file watcher for project {project_name}",
                        extra={
                            "structured_data": {
                                "project_name": project_name,
                                "folder_path": folder_path,
                                "worker_id": worker_id,
                            }
                        },
                    )

                    # Validate folder path exists
                    project_path = Path(folder_path)
                    if not project_path.exists():
                        logger.warning(f"Project folder does not exist: {folder_path}")
                        return

                    # Create file watcher with appropriate configuration
                    watcher = create_file_watcher(
                        project_root=project_path,
                        project_id=project_name,
                        ignore_patterns=self.config.ignore_patterns,
                        debounce_interval=self.config.watch_debounce_ms / 1000.0,
                    )
                    logger.debug(f"VectorDaemon: Created watcher for {project_name}")
                    # Initialize the watcher
                    await watcher.initialize()
                    logger.debug(
                        f"VectorDaemon: Initialized watcher for {project_name}"
                    )
                    # Add change callback
                    watcher.add_change_callback(self._on_file_change(project_name))
                    logger.debug(
                        f"VectorDaemon: Added change callback for {project_name}"
                    )

                    # Start watching
                    watcher.start_watching()
                    logger.debug(f"VectorDaemon: Started watching for {project_name}")

                    # Store watcher for later cleanup
                    self.file_watchers[project_name] = watcher
                    logger.debug(f"VectorDaemon: Stored watcher for {project_name}")

                    logger.info(
                        f"File watcher started for project {project_name}",
                        extra={
                            "structured_data": {
                                "project_name": project_name,
                                "folder_path": folder_path,
                                "watcher_stats": watcher.get_stats(),
                            }
                        },
                    )
                else:
                    logger.debug(
                        f"File watcher already exists for project {project_name}"
                    )

            self.stats["files_processed"] += 1

            # TODO: Implement remaining vector processing:
            # 1. Use watcher to get changed files since last scan
            # 2. Chunk modified files using AST
            # 3. Apply secret redaction
            # 4. Generate embeddings via Voyage
            # 5. Store in Turbopuffer
            # 6. Update database metadata

        except Exception as e:
            logger.error(f"Error processing project {project_name}: {e}", exc_info=True)
            self.stats["errors_count"] += 1

    async def _stats_reporter(self) -> None:
        """Periodically report daemon statistics."""
        while self.is_running:
            try:
                uptime = time.time() - self.stats["start_time"]

                logger.info(
                    "Daemon statistics",
                    extra={
                        "structured_data": {
                            "uptime_seconds": uptime,
                            "monitored_projects": len(self.monitored_projects),
                            "queue_size": self.processing_queue.qsize(),
                            "files_processed": self.stats["files_processed"],
                            "embeddings_generated": self.stats["embeddings_generated"],
                            "errors_count": self.stats["errors_count"],
                        }
                    },
                )

                await asyncio.sleep(60.0)  # Report every minute

            except asyncio.CancelledError:
                logger.info("Stats reporting cancelled")
                break
            except Exception as e:
                logger.error(f"Error in stats reporting: {e}")
                await asyncio.sleep(10.0)

    async def _cleanup(self) -> None:
        """Clean up resources and shut down workers."""
        logger.info("Starting daemon cleanup")
        self.is_running = False

        # Stop and cleanup all file watchers first
        if self.file_watchers:
            logger.info(f"Cleaning up {len(self.file_watchers)} file watchers")
            for project_name, watcher in self.file_watchers.items():
                try:
                    logger.debug(f"Stopping file watcher for project: {project_name}")
                    watcher.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up watcher for {project_name}: {e}")
            self.file_watchers.clear()
            self.watcher_locks.clear()

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Cancel monitor tasks
        for task in self.monitor_tasks:
            task.cancel()

        # Wait for all tasks to finish
        all_tasks = self.workers + self.monitor_tasks
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        logger.info("Vector daemon shutdown complete")

    def get_status(self) -> dict:
        """Get current daemon status."""
        watcher_stats = {}
        for project_name, watcher in self.file_watchers.items():
            try:
                watcher_stats[project_name] = watcher.get_stats()
            except Exception as e:
                watcher_stats[project_name] = {"error": str(e)}

        return {
            "is_running": self.is_running,
            "uptime": time.time() - self.stats["start_time"] if self.is_running else 0,
            "monitored_projects": len(self.monitored_projects),
            "active_file_watchers": len(self.file_watchers),
            "queue_size": self.processing_queue.qsize(),
            "stats": self.stats.copy(),
            "file_watcher_stats": watcher_stats,
        }

    async def _generate_embeddings(
        self, chunks: list[CodeChunk], project_name: str, file_path: Path
    ) -> list[list[float]]:
        """Generate embeddings for file chunks using EmbeddingService."""
        try:
            embeddings = await self._embedding_service.generate_embeddings_for_chunks(
                chunks, project_name, file_path
            )
            
            # Update daemon statistics
            self.stats["embeddings_generated"] += len(embeddings)
            self.stats["last_activity"] = time.time()
            
            return embeddings
            
        except Exception as e:
            # Update error statistics
            self.stats["errors_count"] += 1
            raise

    async def _store_embeddings(
        self, embeddings: list[list[float]], chunks: list[CodeChunk], project_name: str, file_path: str
    ) -> None:
        """Store embeddings in vector database."""
        try:
            await self._vector_storage_service.store_embeddings(
                embeddings, chunks, project_name, file_path
            )
        except Exception as e:
            # Update error statistics
            self.stats["errors_count"] += 1
            raise


async def start_vector_daemon(
    config_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> None:
    """Start the vector daemon process."""

    # Load configuration
    config = load_vector_config(config_path)

    # Setup database
    if db_path is None:
        db_path = Path.home() / ".mcp-code-index" / "tracker.db"
    if cache_dir is None:
        cache_dir = Path.home() / ".mcp-code-index" / "cache"

    db_manager = DatabaseManager(db_path)
    await db_manager.initialize()

    # Create and start daemon
    daemon = VectorDaemon(config, db_manager, cache_dir)

    try:
        await daemon.start()
    finally:
        # Clean up database connections
        await db_manager.close_pool()


def main() -> None:
    """CLI entry point for vector daemon."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Code Indexer Vector Daemon")
    parser.add_argument("--config", type=Path, help="Path to config file")
    parser.add_argument("--db-path", type=Path, help="Path to database")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        asyncio.run(start_vector_daemon(args.config, args.db_path, args.cache_dir))
    except KeyboardInterrupt:
        logger.info("Daemon interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Daemon failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
