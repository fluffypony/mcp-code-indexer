"""
Data models for the MCP Code Indexer.

This module defines Pydantic models for project tracking, file descriptions,
and merge conflicts. These models provide validation and serialization for
the database operations.
"""

from datetime import datetime
from typing import List, Optional
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


# Enable forward references for recursive models
FolderNode.model_rebuild()
CodebaseOverview.model_rebuild()
