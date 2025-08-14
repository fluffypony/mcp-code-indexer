"""
Vector Mode Daemon.

Runs as a background process to monitor file changes and maintain vector indexes.
Handles embedding generation, change detection, and vector database synchronization.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import time

from ..database.database import DatabaseManager
from ..database.models import Project
from .config import VectorConfig, load_vector_config

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

        # Statistics
        self.stats = {
            "start_time": time.time(),
            "files_processed": 0,
            "embeddings_generated": 0,
            "errors_count": 0,
            "last_activity": time.time(),
        }

        # Signal handling is delegated to the parent process

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
            project for project in vector_enabled_projects 
            if project.aliases and project.vector_mode
        ]
        
        # Determine which projects to monitor (not currently monitored)
        projects_to_monitor = [
            project for project in monitorable_projects
            if project.name not in self.monitored_projects
        ]
        
        # Determine which projects to unmonitor (currently monitored but no longer should be)
        monitorable_names = {project.name for project in monitorable_projects}
        projects_to_unmonitor = []
        
        # Get full project data for unmonitoring
        all_projects = await self.db_manager.get_all_projects()
        for project in all_projects:
            if (project.name in self.monitored_projects and 
                project.name not in monitorable_names):
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
        task = {
            "type": "scan_project",
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

        if task_type == "scan_project":
            await self._process_project_scan(task, worker_id)
        else:
            logger.warning(f"Unknown task type: {task_type}")

    async def _process_project_scan(self, task: dict, worker_id: str) -> None:
        """Process a project scan task."""
        project_name = task["project_name"]
        folder_path = task["folder_path"]

        logger.debug(f"Worker {worker_id} processing project: {project_name}")

        try:
            # Check if vector mode components are available
            # For now, just log that we would process this project
            logger.info(
                f"Vector processing for project {project_name}",
                extra={
                    "structured_data": {
                        "project_name": project_name,
                        "folder_path": folder_path,
                        "worker_id": worker_id,
                    }
                },
            )

            self.stats["files_processed"] += 1

            # TODO: Implement actual vector processing:
            # 1. Scan for file changes using Merkle tree
            # 2. Chunk modified files using AST
            # 3. Apply secret redaction
            # 4. Generate embeddings via Voyage
            # 5. Store in Turbopuffer
            # 6. Update database metadata

        except Exception as e:
            logger.error(f"Error processing project {project_name}: {e}")
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
        return {
            "is_running": self.is_running,
            "uptime": time.time() - self.stats["start_time"] if self.is_running else 0,
            "monitored_projects": len(self.monitored_projects),
            "queue_size": self.processing_queue.qsize(),
            "stats": self.stats.copy(),
        }


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
