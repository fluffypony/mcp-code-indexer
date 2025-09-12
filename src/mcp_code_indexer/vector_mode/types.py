"""Type definitions for vector mode daemon tasks."""

from enum import Enum
from typing import TypedDict, Literal

from mcp_code_indexer.vector_mode.monitoring.change_detector import FileChange


class VectorDaemonTaskType(Enum):
    """Task types for daemon processing queue."""

    SCAN_PROJECT = "scan_project"
    PROCESS_FILE_CHANGE = "process_file_change"
    INITIAL_PROJECT_EMBEDDING = "initial_project_embedding"


class BaseTask(TypedDict):
    """Base task with common fields."""

    project_name: str
    timestamp: float


class ScanProjectTask(BaseTask):
    """Task for scanning and indexing a project."""

    type: Literal[VectorDaemonTaskType.SCAN_PROJECT]
    folder_path: str


class ProcessFileChangeTask(BaseTask):
    """Task for processing file changes."""

    type: Literal[VectorDaemonTaskType.PROCESS_FILE_CHANGE]
    change: "FileChange"  # Forward reference to avoid circular import


class InitialProjectEmbeddingTask(BaseTask):
    """Task for performing initial project embedding."""

    type: Literal[VectorDaemonTaskType.INITIAL_PROJECT_EMBEDDING]
    folder_path: str


# Union type for all task types
TaskItem = ScanProjectTask | ProcessFileChangeTask | InitialProjectEmbeddingTask
