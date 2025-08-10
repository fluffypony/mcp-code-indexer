"""
Data models for the MCP Code Indexer.

This module defines Pydantic models for project tracking, file descriptions,
and merge conflicts. These models provide validation and serialization for
the database operations.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


class Project(BaseModel):
    """
    Represents a tracked project/repository.

    Projects are identified by project name and folder paths,
    allowing tracking across different local copies without git coupling.
    """

    id: str = Field(..., description="Generated unique identifier")
    name: str = Field(..., description="User-provided project name")
    aliases: List[str] = Field(
        default_factory=list, description="Alternative identifiers"
    )
    created: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    last_accessed: datetime = Field(
        default_factory=datetime.utcnow, description="Last access timestamp"
    )


class FileDescription(BaseModel):
    """
    Represents a file description within a project.

    Stores detailed summaries of file contents including purpose, components,
    and relationships to enable efficient codebase navigation.
    """

    id: Optional[int] = Field(None, description="Database ID")
    project_id: str = Field(..., description="Reference to project")
    file_path: str = Field(..., description="Relative path from project root")
    description: str = Field(..., description="Detailed content description")
    file_hash: Optional[str] = Field(None, description="SHA-256 of file contents")
    last_modified: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    version: int = Field(default=1, description="For optimistic concurrency control")
    source_project_id: Optional[str] = Field(
        None, description="Source project if copied from upstream"
    )
    to_be_cleaned: Optional[int] = Field(
        None, description="UNIX timestamp for cleanup, NULL = active"
    )


class MergeConflict(BaseModel):
    """
    Represents a merge conflict between file descriptions.

    Used during branch merging when the same file has different descriptions
    in source and target branches.
    """

    id: Optional[int] = Field(None, description="Database ID")
    project_id: str = Field(..., description="Project identifier")
    file_path: str = Field(..., description="Path to conflicted file")
    source_branch: str = Field(..., description="Branch being merged from")
    target_branch: str = Field(..., description="Branch being merged into")
    source_description: str = Field(..., description="Description from source branch")
    target_description: str = Field(..., description="Description from target branch")
    resolution: Optional[str] = Field(None, description="AI-provided resolution")
    created: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )


class ProjectOverview(BaseModel):
    """
    Represents a condensed, interpretive overview of an entire codebase.

    Stores a comprehensive narrative that captures architecture, components,
    relationships, and design patterns in a single document rather than
    individual file descriptions.
    """

    project_id: str = Field(..., description="Reference to project")
    overview: str = Field(..., description="Comprehensive codebase narrative")
    last_modified: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    total_files: int = Field(..., description="Number of files in codebase")
    total_tokens: int = Field(
        ..., description="Total tokens in individual descriptions"
    )


class CodebaseOverview(BaseModel):
    """
    Represents a complete codebase structure with file descriptions.

    Provides hierarchical view of project files with token count information
    to help determine whether to use full overview or search-based approach.
    """

    project_name: str = Field(..., description="Project name")
    total_files: int = Field(..., description="Total number of tracked files")
    total_tokens: int = Field(..., description="Total token count for all descriptions")
    is_large: bool = Field(..., description="True if exceeds configured token limit")
    token_limit: int = Field(..., description="Current token limit setting")
    structure: "FolderNode" = Field(..., description="Hierarchical folder structure")


class FolderNode(BaseModel):
    """
    Represents a folder in the codebase hierarchy.
    """

    name: str = Field(..., description="Folder name")
    path: str = Field(..., description="Full path from project root")
    files: List["FileNode"] = Field(
        default_factory=list, description="Files in this folder"
    )
    folders: List["FolderNode"] = Field(default_factory=list, description="Subfolders")


class FileNode(BaseModel):
    """
    Represents a file in the codebase hierarchy.
    """

    name: str = Field(..., description="File name")
    path: str = Field(..., description="Full path from project root")
    description: str = Field(..., description="File description")


class SearchResult(BaseModel):
    """
    Represents a search result with relevance scoring.
    """

    file_path: str = Field(..., description="Path to the matching file")
    description: str = Field(..., description="File description")
    relevance_score: float = Field(..., description="Search relevance score")
    project_id: str = Field(..., description="Project identifier")


class CodebaseSizeInfo(BaseModel):
    """
    Information about codebase size and token usage.
    """

    total_tokens: int = Field(..., description="Total token count")
    is_large: bool = Field(..., description="Whether codebase exceeds token limit")
    recommendation: str = Field(
        ..., description="Recommended approach (use_search or use_overview)"
    )
    token_limit: int = Field(..., description="Configured token limit")
    cleaned_up_files: List[str] = Field(
        default_factory=list, description="Files removed during cleanup"
    )
    cleaned_up_count: int = Field(default=0, description="Number of files cleaned up")


class WordFrequencyTerm(BaseModel):
    """
    Represents a term and its frequency from word analysis.
    """

    term: str = Field(..., description="The word/term")
    frequency: int = Field(..., description="Number of occurrences")


class WordFrequencyResult(BaseModel):
    """
    Results from word frequency analysis of file descriptions.
    """

    top_terms: List[WordFrequencyTerm] = Field(..., description="Top frequent terms")
    total_terms_analyzed: int = Field(..., description="Total terms processed")
    total_unique_terms: int = Field(..., description="Number of unique terms found")


# Vector Mode Models

class ChunkType(str, Enum):
    """Types of code chunks for semantic analysis."""
    FUNCTION = "function"
    CLASS = "class"  
    METHOD = "method"
    IMPORT = "import"
    DOCSTRING = "docstring"
    COMMENT = "comment"
    VARIABLE = "variable"
    INTERFACE = "interface"
    TYPE_DEFINITION = "type_definition"
    MODULE = "module"
    NAMESPACE = "namespace"
    GENERIC = "generic"

class NodeType(str, Enum):
    """Types of nodes in Merkle tree."""
    FILE = "file"
    DIRECTORY = "directory"
    PROJECT = "project"

class SyncStatus(str, Enum):
    """Vector index synchronization status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class CodeChunk(BaseModel):
    """
    Represents a semantic chunk of code extracted from a file.
    
    Used for embedding generation and vector search operations.
    """
    
    id: Optional[int] = Field(None, description="Database ID")
    file_id: int = Field(..., description="Reference to FileDescription")
    project_id: str = Field(..., description="Reference to project")
    chunk_type: ChunkType = Field(..., description="Type of code chunk")
    name: Optional[str] = Field(None, description="Name of function/class/etc")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    content_hash: str = Field(..., description="SHA-256 hash of chunk content")
    embedding_id: Optional[str] = Field(None, description="Vector database ID")
    redacted: bool = Field(default=False, description="Whether content was redacted")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_modified: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

class MerkleNode(BaseModel):
    """
    Represents a node in the Merkle tree for change detection.
    
    Used to efficiently detect file system changes without scanning entire directory trees.
    """
    
    id: Optional[int] = Field(None, description="Database ID")
    project_id: str = Field(..., description="Reference to project")
    path: str = Field(..., description="File/directory path relative to project root")
    hash: str = Field(..., description="SHA-256 hash of content or children")
    node_type: NodeType = Field(..., description="Type of filesystem node")
    parent_path: Optional[str] = Field(None, description="Path to parent directory")
    children_hash: Optional[str] = Field(None, description="Combined hash of children")
    last_modified: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

class IndexMeta(BaseModel):
    """
    Metadata about vector indexing progress and status for a project.
    
    Tracks indexing state, statistics, and synchronization status.
    """
    
    id: Optional[int] = Field(None, description="Database ID")
    project_id: str = Field(..., description="Reference to project", unique=True)
    total_chunks: int = Field(default=0, description="Total number of chunks")
    indexed_chunks: int = Field(default=0, description="Number of chunks with embeddings")
    total_files: int = Field(default=0, description="Total number of files")
    indexed_files: int = Field(default=0, description="Number of files processed")
    last_sync: Optional[datetime] = Field(None, description="Last successful sync timestamp")
    sync_status: SyncStatus = Field(default=SyncStatus.PENDING, description="Current sync status")
    error_message: Optional[str] = Field(None, description="Last error message")
    queue_depth: int = Field(default=0, description="Number of pending tasks")
    processing_rate: float = Field(default=0.0, description="Files per second processing rate")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_modified: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

class VectorSearchResult(BaseModel):
    """
    Represents a vector search result with similarity scoring.
    """
    
    file_path: str = Field(..., description="Path to the matching file")
    chunk_name: Optional[str] = Field(None, description="Name of the code chunk")
    chunk_type: ChunkType = Field(..., description="Type of code chunk")
    code_snippet: str = Field(..., description="Original code content")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    similarity_score: float = Field(..., description="Cosine similarity score")
    project_id: str = Field(..., description="Project identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class VectorIndexStatus(BaseModel):
    """
    Current status of vector indexing for a project.
    """
    
    is_indexing: bool = Field(..., description="Whether indexing is currently active")
    indexed_files: int = Field(..., description="Number of files indexed")
    total_files: int = Field(..., description="Total number of files")
    indexed_chunks: int = Field(..., description="Number of chunks indexed")
    total_chunks: int = Field(..., description="Total number of chunks")
    last_sync: Optional[datetime] = Field(None, description="Last sync timestamp")
    sync_status: SyncStatus = Field(..., description="Current sync status")
    queue_depth: int = Field(..., description="Number of pending tasks")
    processing_rate: float = Field(..., description="Processing rate")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Last error message")

# Enable forward references for recursive models
FolderNode.model_rebuild()
CodebaseOverview.model_rebuild()
