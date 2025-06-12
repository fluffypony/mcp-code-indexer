#!/usr/bin/env python3
"""
Git Hook Handler for MCP Code Indexer

Handles automated analysis of git changes and updates file descriptions
and project overview using OpenRouter API integration.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse

import aiohttp
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from .database.database import DatabaseManager
from .database.models import Project, FileDescription
from .error_handler import ValidationError
from .token_counter import TokenCounter


class GitHookError(Exception):
    """Custom exception for git hook operations."""
    pass


class ThrottlingError(Exception):
    """Exception for rate limiting scenarios."""
    pass


class GitHookHandler:
    """
    Handles git hook integration for automated code indexing.
    
    This class provides functionality to:
    - Analyze git diffs to identify changed files
    - Use OpenRouter API to update file descriptions
    - Update project overview when structural changes occur
    """
    
    # OpenRouter configuration
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL = "anthropic/claude-sonnet-4"
    
    def __init__(self, db_manager: DatabaseManager, cache_dir: Path):
        """
        Initialize GitHookHandler.
        
        Args:
            db_manager: Database manager instance
            cache_dir: Cache directory for temporary files
        """
        self.db_manager = db_manager
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self.token_counter = TokenCounter()
        
        # Git hook specific settings
        self.config = {
            "model": os.getenv("MCP_GITHOOK_MODEL", self.OPENROUTER_MODEL),
            "max_diff_tokens": 136000,  # Skip if diff larger than this (in tokens)
            "timeout": 30,
            "temperature": 0.3,  # Lower temperature for consistent updates
        }
        
        # Validate OpenRouter API key
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise GitHookError("OPENROUTER_API_KEY environment variable is required for git hook mode")
    
    async def run_githook_mode(
        self, 
        commit_hash: Optional[str] = None, 
        commit_range: Optional[Tuple[str, str]] = None
    ) -> None:
        """
        Run in git hook mode - analyze changes and update descriptions.
        
        Args:
            commit_hash: Process a specific commit by hash
            commit_range: Process commits in range (start_hash, end_hash)
        
        This is the main entry point for git hook functionality.
        """
        try:
            # Get git info from current directory
            project_info = await self._identify_project_from_git()
            
            # Get git diff and commit message based on mode
            if commit_hash:
                git_diff = await self._get_git_diff_for_commit(commit_hash)
                commit_message = await self._get_commit_message_for_commit(commit_hash)
            elif commit_range:
                git_diff = await self._get_git_diff_for_range(commit_range[0], commit_range[1])
                commit_message = await self._get_commit_messages_for_range(commit_range[0], commit_range[1])
            else:
                git_diff = await self._get_git_diff()
                commit_message = await self._get_commit_message()
            
            if not git_diff:
                self.logger.info(f"Skipping git hook update - no git diff")
                return
            
            # Fetch current state
            current_overview = await self._get_project_overview(project_info)
            current_descriptions = await self._get_all_descriptions(project_info)
            changed_files = self._extract_changed_files(git_diff)
            
            if not changed_files:
                self.logger.info("No changed files detected in git diff")
                return
            
            # Build prompt for OpenRouter
            prompt = self._build_githook_prompt(
                git_diff, 
                commit_message,
                current_overview, 
                current_descriptions,
                changed_files
            )
            
            # Check total prompt token count
            prompt_tokens = self.token_counter.count_tokens(prompt)
            if prompt_tokens > self.config["max_diff_tokens"]:
                self.logger.info(f"Skipping git hook update - prompt too large ({prompt_tokens} tokens > {self.config['max_diff_tokens']} limit)")
                return
            
            # Call OpenRouter API
            updates = await self._call_openrouter(prompt)
            
            # Apply updates to database
            await self._apply_updates(project_info, updates)
            
            self.logger.info(f"Git hook update completed for {len(changed_files)} files")
            
        except Exception as e:
            self.logger.error(f"Git hook mode failed: {e}")
            # Don't fail the git operation - just log the error
            raise GitHookError(f"Git hook processing failed: {e}")
    
    async def _identify_project_from_git(self) -> Dict[str, Any]:
        """
        Identify project information from git repository.
        
        Returns:
            Dict containing project identification info
        """
        try:
            # Get current working directory as project root
            project_root = Path.cwd()
            
            # Get git remote info
            remote_result = await self._run_git_command(["remote", "get-url", "origin"])
            remote_origin = remote_result.strip() if remote_result else None
            
            # Try to get upstream origin
            upstream_origin = None
            try:
                upstream_result = await self._run_git_command(["remote", "get-url", "upstream"])
                upstream_origin = upstream_result.strip() if upstream_result else None
            except subprocess.CalledProcessError:
                pass  # No upstream remote
            
            # Get current branch
            branch_result = await self._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
            branch = branch_result.strip() if branch_result else "main"
            
            # Extract project name from remote URL or use directory name
            project_name = self._extract_project_name(remote_origin, project_root)
            
            return {
                "projectName": project_name,
                "folderPath": str(project_root),
                "branch": branch,
                "remoteOrigin": remote_origin,
                "upstreamOrigin": upstream_origin
            }
            
        except Exception as e:
            raise GitHookError(f"Failed to identify project from git: {e}")
    
    def _extract_project_name(self, remote_origin: Optional[str], project_root: Path) -> str:
        """Extract project name from remote URL or directory name."""
        if remote_origin:
            # Parse GitHub/GitLab URL
            if remote_origin.startswith("git@"):
                # SSH format: git@github.com:user/repo.git
                parts = remote_origin.split(":")
                if len(parts) >= 2:
                    repo_path = parts[-1].replace(".git", "")
                    return repo_path.split("/")[-1]
            else:
                # HTTPS format
                parsed = urlparse(remote_origin)
                if parsed.path:
                    repo_path = parsed.path.strip("/").replace(".git", "")
                    return repo_path.split("/")[-1]
        
        # Fallback to directory name
        return project_root.name
    
    async def _get_git_diff(self) -> str:
        """
        Get git diff for recent changes.
        
        Returns:
            Git diff content as string
        """
        try:
            # Get diff from last commit
            diff_result = await self._run_git_command([
                "diff", "--no-color", "--no-ext-diff", "HEAD~1..HEAD"
            ])
            return diff_result
            
        except subprocess.CalledProcessError:
            # If HEAD~1 doesn't exist (first commit), get diff against empty tree
            try:
                diff_result = await self._run_git_command([
                    "diff", "--no-color", "--no-ext-diff", "--cached"
                ])
                return diff_result
            except subprocess.CalledProcessError as e:
                raise GitHookError(f"Failed to get git diff: {e}")

    async def _get_commit_message(self) -> str:
        """
        Get the commit message for context about what was changed.
        
        Returns:
            Commit message as string
        """
        try:
            # Get the commit message from the latest commit
            message_result = await self._run_git_command([
                "log", "-1", "--pretty=%B"
            ])
            return message_result.strip()
            
        except subprocess.CalledProcessError:
            # If no commits exist yet, return empty string
            return ""

    async def _get_git_diff_for_commit(self, commit_hash: str) -> str:
        """
        Get git diff for a specific commit.
        
        Args:
            commit_hash: The commit hash to analyze
            
        Returns:
            Git diff content as string
        """
        try:
            # Get diff for the specific commit compared to its parent
            diff_result = await self._run_git_command([
                "diff", "--no-color", "--no-ext-diff", f"{commit_hash}~1..{commit_hash}"
            ])
            return diff_result
            
        except subprocess.CalledProcessError:
            # If parent doesn't exist (first commit), diff against empty tree
            try:
                diff_result = await self._run_git_command([
                    "diff", "--no-color", "--no-ext-diff", "4b825dc642cb6eb9a060e54bf8d69288fbee4904", commit_hash
                ])
                return diff_result
            except subprocess.CalledProcessError as e:
                raise GitHookError(f"Failed to get git diff for commit {commit_hash}: {e}")

    async def _get_git_diff_for_range(self, start_hash: str, end_hash: str) -> str:
        """
        Get git diff for a range of commits.
        
        Args:
            start_hash: Starting commit hash (exclusive)
            end_hash: Ending commit hash (inclusive)
            
        Returns:
            Git diff content as string
        """
        try:
            diff_result = await self._run_git_command([
                "diff", "--no-color", "--no-ext-diff", f"{start_hash}..{end_hash}"
            ])
            return diff_result
        except subprocess.CalledProcessError as e:
            raise GitHookError(f"Failed to get git diff for range {start_hash}..{end_hash}: {e}")

    async def _get_commit_message_for_commit(self, commit_hash: str) -> str:
        """
        Get the commit message for a specific commit.
        
        Args:
            commit_hash: The commit hash
            
        Returns:
            Commit message as string
        """
        try:
            message_result = await self._run_git_command([
                "log", "-1", "--pretty=%B", commit_hash
            ])
            return message_result.strip()
        except subprocess.CalledProcessError as e:
            raise GitHookError(f"Failed to get commit message for {commit_hash}: {e}")

    async def _get_commit_messages_for_range(self, start_hash: str, end_hash: str) -> str:
        """
        Get commit messages for a range of commits.
        
        Args:
            start_hash: Starting commit hash (exclusive)
            end_hash: Ending commit hash (inclusive)
            
        Returns:
            Combined commit messages as string
        """
        try:
            # Get all commit messages in the range
            message_result = await self._run_git_command([
                "log", "--pretty=%B", f"{start_hash}..{end_hash}"
            ])
            
            # Clean up and format the messages
            messages = message_result.strip()
            if messages:
                return f"Combined commit messages for range {start_hash}..{end_hash}:\n\n{messages}"
            else:
                return f"No commits found in range {start_hash}..{end_hash}"
                
        except subprocess.CalledProcessError as e:
            raise GitHookError(f"Failed to get commit messages for range {start_hash}..{end_hash}: {e}")
    
    def _extract_changed_files(self, git_diff: str) -> List[str]:
        """
        Extract list of changed files from git diff.
        
        Args:
            git_diff: Git diff content
            
        Returns:
            List of file paths that changed
        """
        changed_files = []
        lines = git_diff.split('\n')
        
        for line in lines:
            if line.startswith('diff --git a/'):
                # Parse file path from diff header
                # Format: diff --git a/path/to/file b/path/to/file
                parts = line.split(' ')
                if len(parts) >= 4:
                    file_path = parts[2][2:]  # Remove 'a/' prefix
                    changed_files.append(file_path)
        
        return changed_files
    
    async def _get_project_overview(self, project_info: Dict[str, Any]) -> str:
        """Get current project overview from database."""
        try:
            # Try to find existing project
            project = await self.db_manager.find_matching_project(
                project_info["projectName"],
                project_info.get("remoteOrigin"),
                project_info.get("upstreamOrigin"),
                project_info["folderPath"]
            )
            
            if project:
                overview = await self.db_manager.get_project_overview(
                    project.id, project_info["branch"]
                )
                return overview.overview if overview else ""
            
            return ""
            
        except Exception as e:
            self.logger.warning(f"Failed to get project overview: {e}")
            return ""
    
    async def _get_all_descriptions(self, project_info: Dict[str, Any]) -> Dict[str, str]:
        """Get all current file descriptions from database."""
        try:
            # Try to find existing project
            project = await self.db_manager.find_matching_project(
                project_info["projectName"],
                project_info.get("remoteOrigin"),
                project_info.get("upstreamOrigin"),
                project_info["folderPath"]
            )
            
            if project:
                descriptions = await self.db_manager.get_all_file_descriptions(
                    project.id, project_info["branch"]
                )
                return {desc.file_path: desc.description for desc in descriptions}
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"Failed to get file descriptions: {e}")
            return {}
    
    def _build_githook_prompt(
        self, 
        git_diff: str, 
        commit_message: str,
        overview: str, 
        descriptions: Dict[str, str], 
        changed_files: List[str]
    ) -> str:
        """
        Build prompt for OpenRouter API to analyze git changes.
        
        Args:
            git_diff: Git diff content
            commit_message: Commit message explaining the changes
            overview: Current project overview
            descriptions: Current file descriptions
            changed_files: List of changed file paths
            
        Returns:
            Formatted prompt for the API
        """
        return f"""Analyze this git commit and update the file descriptions and project overview as needed.

COMMIT MESSAGE:
{commit_message or "No commit message available"}

CURRENT PROJECT OVERVIEW:
{overview or "No overview available"}

CURRENT FILE DESCRIPTIONS:
{json.dumps(descriptions, indent=2)}

GIT DIFF:
{git_diff}

CHANGED FILES:
{', '.join(changed_files)}

INSTRUCTIONS:

Use the COMMIT MESSAGE to understand the intent and context of the changes. The commit message explains what the developer was trying to accomplish.

1. **File Descriptions**: Update descriptions for any files that have changed significantly. Consider both the diff content and the commit message context. Only include files that need actual description updates.

2. **Project Overview**: Update ONLY if there are major structural changes like:
   - New major features or components (which may be indicated by commit message)
   - Architectural changes (new patterns, frameworks, or approaches)
   - Significant dependency additions
   - New API endpoints or workflows
   - Changes to build/deployment processes
   
   Do NOT update overview for minor changes like bug fixes, small refactors, or documentation updates.

3. **Overview Format**: If updating the overview, follow this structure with comprehensive narrative (10-20 pages of text):

````
## Directory Structure
```
src/
├── api/          # REST API endpoints and middleware
├── models/       # Database models and business logic  
├── services/     # External service integrations
├── utils/        # Shared utilities and helpers
└── tests/        # Test suites
```

## Architecture Overview
[Describe how components interact, data flow, key design decisions]

## Core Components
### API Layer
[Details about API structure, authentication, routing]

### Data Model
[Key entities, relationships, database design]

## Key Workflows
1. User Authentication Flow
   [Step-by-step description]
2. Data Processing Pipeline
   [How data moves through the system]

[Continue with other sections...]
````

Return ONLY a JSON object in this exact format:
{{
  "file_updates": {{
    "path/to/file1.py": "Updated description for file1",
    "path/to/file2.js": "Updated description for file2"
  }},
  "overview_update": "Updated project overview text (or null if no update needed)"
}}

Return ONLY the JSON, no other text."""
    
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ThrottlingError)
    )
    async def _call_openrouter(self, prompt: str) -> Dict[str, Any]:
        """
        Call OpenRouter API to analyze changes.
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            Parsed response with file updates and overview update
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/fluffypony/mcp-code-indexer",
            "X-Title": "MCP Code Indexer Git Hook",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config["model"],
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a technical assistant that analyzes code changes and updates file descriptions accurately and concisely."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": self.config["temperature"],
            "max_tokens": 24000,
        }
        

        
        timeout = aiohttp.ClientTimeout(total=self.config["timeout"])
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.OPENROUTER_API_URL, 
                    headers=headers, 
                    json=payload
                ) as response:
                    
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        raise ThrottlingError(f"Rate limited. Retry after {retry_after}s")
                    
                    response.raise_for_status()
                    
                    response_data = await response.json()
                    
                    if "choices" not in response_data:
                        raise GitHookError(f"Invalid API response format: {response_data}")
                    
                    content = response_data["choices"][0]["message"]["content"]
                    return self._validate_githook_response(content)
                    
        except aiohttp.ClientError as e:
            raise GitHookError(f"OpenRouter API request failed: {e}")
        except asyncio.TimeoutError:
            raise GitHookError("OpenRouter API request timed out")
    
    def _validate_githook_response(self, response_text: str) -> Dict[str, Any]:
        """
        Validate and parse JSON response from OpenRouter.
        
        Args:
            response_text: Raw response content
            
        Returns:
            Validated response data
        """
        try:
            data = json.loads(response_text.strip())
            
            # Validate structure
            if "file_updates" not in data:
                raise ValueError("Missing 'file_updates' field")
            if "overview_update" not in data:
                raise ValueError("Missing 'overview_update' field")
            
            if not isinstance(data["file_updates"], dict):
                raise ValueError("'file_updates' must be a dictionary")
            
            # Validate descriptions
            for path, desc in data["file_updates"].items():
                if not isinstance(desc, str) or not desc.strip():
                    raise ValueError(f"Invalid description for {path}")
            
            return data
            
        except json.JSONDecodeError as e:
            raise GitHookError(f"Invalid JSON response from API: {e}")
        except ValueError as e:
            raise GitHookError(f"Invalid response structure: {e}")
    
    async def _apply_updates(self, project_info: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Apply updates to database.
        
        Args:
            project_info: Project identification info
            updates: Updates from OpenRouter API
        """
        try:
            # Get or create project
            project = await self.db_manager.get_or_create_project(
                project_info["projectName"],
                project_info["folderPath"],
                project_info.get("remoteOrigin"),
                project_info.get("upstreamOrigin")
            )
            
            # Update file descriptions
            file_updates = updates.get("file_updates", {})
            for file_path, description in file_updates.items():
                from mcp_code_indexer.database.models import FileDescription
                from datetime import datetime
                
                file_desc = FileDescription(
                    project_id=project.id,
                    branch=project_info["branch"],
                    file_path=file_path,
                    description=description,
                    file_hash=None,
                    last_modified=datetime.utcnow(),
                    version=1
                )
                await self.db_manager.create_file_description(file_desc)
                self.logger.info(f"Updated description for {file_path}")
            
            # Update project overview if provided
            overview_update = updates.get("overview_update")
            if overview_update and overview_update.strip():
                from mcp_code_indexer.database.models import ProjectOverview
                from datetime import datetime
                
                overview = ProjectOverview(
                    project_id=project.id,
                    branch=project_info["branch"],
                    overview=overview_update,
                    last_modified=datetime.utcnow(),
                    total_files=len(file_updates),
                    total_tokens=len(overview_update.split())
                )
                await self.db_manager.create_project_overview(overview)
                self.logger.info("Updated project overview")
            
        except Exception as e:
            raise GitHookError(f"Failed to apply updates to database: {e}")
    
    async def _run_git_command(self, cmd: List[str]) -> str:
        """
        Run a git command and return output.
        
        Args:
            cmd: Git command arguments
            
        Returns:
            Command output as string
        """
        full_cmd = ["git"] + cmd
        
        try:
            process = await asyncio.create_subprocess_exec(
                *full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, 
                    full_cmd, 
                    stdout, 
                    stderr
                )
            
            return stdout.decode('utf-8')
            
        except FileNotFoundError:
            raise GitHookError("Git command not found - ensure git is installed and in PATH")
