"""
Two-phase merge functionality for branch descriptions.

This module implements conflict detection and resolution for merging
file descriptions between branches with AI-assisted conflict resolution.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from mcp_code_indexer.database.database import DatabaseManager
from mcp_code_indexer.database.models import FileDescription
from mcp_code_indexer.error_handler import ValidationError, DatabaseError
from mcp_code_indexer.logging_config import get_logger

logger = get_logger(__name__)


class MergeConflict:
    """Represents a merge conflict between file descriptions."""
    
    def __init__(
        self,
        file_path: str,
        source_branch: str,
        target_branch: str,
        source_description: str,
        target_description: str,
        conflict_id: Optional[str] = None
    ):
        """
        Initialize merge conflict.
        
        Args:
            file_path: Path to conflicted file
            source_branch: Branch being merged from
            target_branch: Branch being merged into
            source_description: Description from source branch
            target_description: Description from target branch
            conflict_id: Optional conflict identifier
        """
        self.file_path = file_path
        self.source_branch = source_branch
        self.target_branch = target_branch
        self.source_description = source_description
        self.target_description = target_description
        self.conflict_id = conflict_id or str(uuid4())
        self.resolution: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert conflict to dictionary representation."""
        return {
            "conflictId": self.conflict_id,
            "filePath": self.file_path,
            "sourceBranch": self.source_branch,
            "targetBranch": self.target_branch,
            "sourceDescription": self.source_description,
            "targetDescription": self.target_description,
            "resolution": self.resolution
        }


class MergeSession:
    """Manages a merge session with conflicts and resolutions."""
    
    def __init__(self, project_id: str, source_branch: str, target_branch: str):
        """
        Initialize merge session.
        
        Args:
            project_id: Project identifier
            source_branch: Branch being merged from
            target_branch: Branch being merged into
        """
        self.session_id = str(uuid4())
        self.project_id = project_id
        self.source_branch = source_branch
        self.target_branch = target_branch
        self.conflicts: List[MergeConflict] = []
        self.created = datetime.utcnow()
        self.status = "pending"  # pending, resolved, aborted
    
    def add_conflict(self, conflict: MergeConflict) -> None:
        """Add a conflict to the session."""
        self.conflicts.append(conflict)
    
    def get_conflict_count(self) -> int:
        """Get total number of conflicts."""
        return len(self.conflicts)
    
    def get_resolved_count(self) -> int:
        """Get number of resolved conflicts."""
        return len([c for c in self.conflicts if c.resolution is not None])
    
    def is_fully_resolved(self) -> bool:
        """Check if all conflicts are resolved."""
        return self.get_resolved_count() == self.get_conflict_count()
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary representation."""
        return {
            "sessionId": self.session_id,
            "projectId": self.project_id,
            "sourceBranch": self.source_branch,
            "targetBranch": self.target_branch,
            "totalConflicts": self.get_conflict_count(),
            "resolvedConflicts": self.get_resolved_count(),
            "isFullyResolved": self.is_fully_resolved(),
            "created": self.created.isoformat(),
            "status": self.status,
            "conflicts": [conflict.to_dict() for conflict in self.conflicts]
        }


class MergeHandler:
    """
    Handles two-phase merge operations for file descriptions.
    
    Phase 1: Detect conflicts between source and target branches
    Phase 2: Apply resolutions and complete merge
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize merge handler.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self._active_sessions: Dict[str, MergeSession] = {}
    
    async def start_merge_phase1(
        self,
        project_id: str,
        source_branch: str,
        target_branch: str
    ) -> MergeSession:
        """
        Phase 1: Detect merge conflicts.
        
        Args:
            project_id: Project identifier
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            
        Returns:
            MergeSession with detected conflicts
            
        Raises:
            ValidationError: If branches are invalid
            DatabaseError: If database operation fails
        """
        if source_branch == target_branch:
            raise ValidationError("Source and target branches cannot be the same")
        
        logger.info(f"Starting merge phase 1: {source_branch} -> {target_branch}")
        
        try:
            # Get file descriptions from both branches
            source_descriptions = await self.db_manager.get_all_file_descriptions(
                project_id, source_branch
            )
            target_descriptions = await self.db_manager.get_all_file_descriptions(
                project_id, target_branch
            )
            
            # Create session
            session = MergeSession(project_id, source_branch, target_branch)
            
            # Build lookup dictionaries
            source_lookup = {desc.file_path: desc for desc in source_descriptions}
            target_lookup = {desc.file_path: desc for desc in target_descriptions}
            
            # Detect conflicts
            conflicts_found = 0
            all_files = set(source_lookup.keys()) | set(target_lookup.keys())
            
            for file_path in all_files:
                source_desc = source_lookup.get(file_path)
                target_desc = target_lookup.get(file_path)
                
                # Conflict occurs when:
                # 1. File exists in both branches with different descriptions
                # 2. File has been modified in source but also exists in target
                if source_desc and target_desc:
                    if source_desc.description != target_desc.description:
                        conflict = MergeConflict(
                            file_path=file_path,
                            source_branch=source_branch,
                            target_branch=target_branch,
                            source_description=source_desc.description,
                            target_description=target_desc.description
                        )
                        session.add_conflict(conflict)
                        conflicts_found += 1
            
            # Store session
            self._active_sessions[session.session_id] = session
            
            logger.info(f"Merge phase 1 completed: {conflicts_found} conflicts found")
            
            return session
            
        except Exception as e:
            logger.error(f"Error in merge phase 1: {e}")
            raise DatabaseError(f"Failed to detect merge conflicts: {e}") from e
    
    async def complete_merge_phase2(
        self,
        session_id: str,
        conflict_resolutions: List[Dict[str, str]]
    ) -> Dict:
        """
        Phase 2: Apply resolutions and complete merge.
        
        Args:
            session_id: Merge session identifier
            conflict_resolutions: List of {conflictId, resolvedDescription}
            
        Returns:
            Merge result summary
            
        Raises:
            ValidationError: If session not found or resolutions invalid
            DatabaseError: If database operation fails
        """
        session = self._active_sessions.get(session_id)
        if not session:
            raise ValidationError(f"Merge session not found: {session_id}")
        
        logger.info(f"Starting merge phase 2 for session {session_id}")
        
        try:
            # Validate and apply resolutions
            resolution_lookup = {res["conflictId"]: res["resolvedDescription"] 
                               for res in conflict_resolutions}
            
            resolved_count = 0
            for conflict in session.conflicts:
                if conflict.conflict_id in resolution_lookup:
                    conflict.resolution = resolution_lookup[conflict.conflict_id]
                    resolved_count += 1
            
            # Check if all conflicts are resolved
            if not session.is_fully_resolved():
                unresolved = session.get_conflict_count() - session.get_resolved_count()
                raise ValidationError(
                    f"Not all conflicts resolved: {unresolved} remaining",
                    details={
                        "total_conflicts": session.get_conflict_count(),
                        "resolved_conflicts": session.get_resolved_count(),
                        "unresolved_conflicts": unresolved
                    }
                )
            
            # Apply merge
            merged_descriptions = []
            
            # Get all descriptions from source branch
            source_descriptions = await self.db_manager.get_all_file_descriptions(
                session.project_id, session.source_branch
            )
            
            # Get existing target descriptions
            target_descriptions = await self.db_manager.get_all_file_descriptions(
                session.project_id, session.target_branch
            )
            
            target_lookup = {desc.file_path: desc for desc in target_descriptions}
            
            # Apply resolved descriptions
            for source_desc in source_descriptions:
                resolved_conflict = next(
                    (c for c in session.conflicts if c.file_path == source_desc.file_path),
                    None
                )
                
                if resolved_conflict:
                    # Use resolved description
                    new_desc = FileDescription(
                        project_id=session.project_id,
                        branch=session.target_branch,
                        file_path=source_desc.file_path,
                        description=resolved_conflict.resolution,
                        file_hash=source_desc.file_hash,
                        last_modified=datetime.utcnow(),
                        version=1,
                        source_project_id=source_desc.source_project_id
                    )
                else:
                    # No conflict, copy from source
                    new_desc = FileDescription(
                        project_id=session.project_id,
                        branch=session.target_branch,
                        file_path=source_desc.file_path,
                        description=source_desc.description,
                        file_hash=source_desc.file_hash,
                        last_modified=datetime.utcnow(),
                        version=1,
                        source_project_id=source_desc.source_project_id
                    )
                
                merged_descriptions.append(new_desc)
            
            # Batch update target branch
            await self.db_manager.batch_create_file_descriptions(merged_descriptions)
            
            # Mark session as completed
            session.status = "resolved"
            
            result = {
                "success": True,
                "sessionId": session_id,
                "sourceBranch": session.source_branch,
                "targetBranch": session.target_branch,
                "totalConflicts": session.get_conflict_count(),
                "resolvedConflicts": session.get_resolved_count(),
                "mergedFiles": len(merged_descriptions),
                "message": f"Successfully merged {len(merged_descriptions)} files from {session.source_branch} to {session.target_branch}"
            }
            
            logger.info(f"Merge phase 2 completed successfully: {len(merged_descriptions)} files merged")
            
            # Clean up session
            del self._active_sessions[session_id]
            
            return result
            
        except Exception as e:
            if session:
                session.status = "aborted"
            logger.error(f"Error in merge phase 2: {e}")
            raise DatabaseError(f"Failed to complete merge: {e}") from e
    
    def get_session(self, session_id: str) -> Optional[MergeSession]:
        """Get merge session by ID."""
        return self._active_sessions.get(session_id)
    
    def get_active_sessions(self) -> List[MergeSession]:
        """Get all active merge sessions."""
        return list(self._active_sessions.values())
    
    def abort_session(self, session_id: str) -> bool:
        """
        Abort a merge session.
        
        Args:
            session_id: Session to abort
            
        Returns:
            True if session was aborted
        """
        session = self._active_sessions.get(session_id)
        if session:
            session.status = "aborted"
            del self._active_sessions[session_id]
            logger.info(f"Merge session {session_id} aborted")
            return True
        return False
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old merge sessions.
        
        Args:
            max_age_hours: Maximum age of sessions to keep
            
        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.utcnow() - datetime.timedelta(hours=max_age_hours)
        old_sessions = [
            session_id for session_id, session in self._active_sessions.items()
            if session.created < cutoff_time
        ]
        
        for session_id in old_sessions:
            del self._active_sessions[session_id]
        
        if old_sessions:
            logger.info(f"Cleaned up {len(old_sessions)} old merge sessions")
        
        return len(old_sessions)
