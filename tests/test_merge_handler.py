"""
Tests for merge handler functionality.

This module tests the two-phase merge process including conflict detection,
resolution validation, and successful merge completion.
"""

import pytest
import pytest_asyncio
from datetime import datetime

from src.merge_handler import MergeHandler, MergeConflict, MergeSession
from src.database.models import FileDescription
from src.error_handler import ValidationError, DatabaseError
from tests.conftest import assert_file_description_equal, create_test_file_description


class TestMergeConflict:
    """Test the MergeConflict class."""
    
    def test_conflict_creation(self):
        """Test creating a merge conflict."""
        conflict = MergeConflict(
            file_path="src/test.py",
            source_branch="feature",
            target_branch="main",
            source_description="Feature implementation",
            target_description="Original implementation"
        )
        
        assert conflict.file_path == "src/test.py"
        assert conflict.source_branch == "feature"
        assert conflict.target_branch == "main"
        assert conflict.source_description == "Feature implementation"
        assert conflict.target_description == "Original implementation"
        assert conflict.conflict_id is not None
        assert conflict.resolution is None
    
    def test_conflict_to_dict(self):
        """Test converting conflict to dictionary."""
        conflict = MergeConflict(
            file_path="src/test.py",
            source_branch="feature", 
            target_branch="main",
            source_description="Feature description",
            target_description="Main description"
        )
        
        result = conflict.to_dict()
        
        assert result["filePath"] == "src/test.py"
        assert result["sourceBranch"] == "feature"
        assert result["targetBranch"] == "main"
        assert result["sourceDescription"] == "Feature description"
        assert result["targetDescription"] == "Main description"
        assert "conflictId" in result
        assert result["resolution"] is None


class TestMergeSession:
    """Test the MergeSession class."""
    
    def test_session_creation(self):
        """Test creating a merge session."""
        session = MergeSession("project123", "feature", "main")
        
        assert session.project_id == "project123"
        assert session.source_branch == "feature"
        assert session.target_branch == "main"
        assert session.session_id is not None
        assert session.status == "pending"
        assert len(session.conflicts) == 0
        assert isinstance(session.created, datetime)
    
    def test_conflict_management(self):
        """Test adding and managing conflicts."""
        session = MergeSession("project123", "feature", "main")
        
        conflict1 = MergeConflict("file1.py", "feature", "main", "desc1", "desc2")
        conflict2 = MergeConflict("file2.py", "feature", "main", "desc3", "desc4")
        
        session.add_conflict(conflict1)
        session.add_conflict(conflict2)
        
        assert session.get_conflict_count() == 2
        assert session.get_resolved_count() == 0
        assert not session.is_fully_resolved()
        
        # Resolve one conflict
        conflict1.resolution = "resolved description"
        assert session.get_resolved_count() == 1
        assert not session.is_fully_resolved()
        
        # Resolve second conflict
        conflict2.resolution = "another resolved description"
        assert session.get_resolved_count() == 2
        assert session.is_fully_resolved()
    
    def test_session_to_dict(self):
        """Test converting session to dictionary."""
        session = MergeSession("project123", "feature", "main")
        conflict = MergeConflict("file1.py", "feature", "main", "desc1", "desc2")
        session.add_conflict(conflict)
        
        result = session.to_dict()
        
        assert result["projectId"] == "project123"
        assert result["sourceBranch"] == "feature"
        assert result["targetBranch"] == "main"
        assert result["totalConflicts"] == 1
        assert result["resolvedConflicts"] == 0
        assert result["isFullyResolved"] is False
        assert result["status"] == "pending"
        assert len(result["conflicts"]) == 1


class TestMergeHandler:
    """Test the MergeHandler class."""
    
    @pytest_asyncio.fixture
    async def merge_handler_with_data(self, merge_handler, sample_file_descriptions):
        """Get merge handler with sample data loaded."""
        return merge_handler
    
    async def test_merge_phase1_no_conflicts(self, merge_handler_with_data, sample_project):
        """Test merge phase 1 when there are no conflicts."""
        # Create identical descriptions in both branches
        descriptions = [
            create_test_file_description(
                project_id=sample_project.id,
                branch="branch1",
                file_path="common.py",
                description="Same description"
            ),
            create_test_file_description(
                project_id=sample_project.id,
                branch="branch2", 
                file_path="common.py",
                description="Same description"
            )
        ]
        
        await merge_handler_with_data.db_manager.batch_create_file_descriptions(descriptions)
        
        session = await merge_handler_with_data.start_merge_phase1(
            sample_project.id, "branch1", "branch2"
        )
        
        assert session.get_conflict_count() == 0
        assert session.source_branch == "branch1"
        assert session.target_branch == "branch2"
    
    async def test_merge_phase1_with_conflicts(self, merge_handler_with_data, sample_project):
        """Test merge phase 1 when there are conflicts."""
        # The sample data already contains a conflict between main and feature/new-ui for src/main.py
        session = await merge_handler_with_data.start_merge_phase1(
            sample_project.id, "feature/new-ui", "main"
        )
        
        assert session.get_conflict_count() == 1
        conflict = session.conflicts[0]
        assert conflict.file_path == "src/main.py"
        assert conflict.source_branch == "feature/new-ui"
        assert conflict.target_branch == "main"
        assert "enhanced CLI interface" in conflict.source_description
        assert "CLI argument parsing" in conflict.target_description
    
    async def test_merge_phase1_same_branch_error(self, merge_handler_with_data, sample_project):
        """Test that merging the same branch raises an error."""
        with pytest.raises(ValidationError, match="Source and target branches cannot be the same"):
            await merge_handler_with_data.start_merge_phase1(
                sample_project.id, "main", "main"
            )
    
    async def test_merge_phase2_successful(self, merge_handler_with_data, sample_project):
        """Test successful completion of merge phase 2."""
        # Start phase 1
        session = await merge_handler_with_data.start_merge_phase1(
            sample_project.id, "feature/new-ui", "main"
        )
        
        assert session.get_conflict_count() == 1
        conflict = session.conflicts[0]
        
        # Resolve conflicts
        resolutions = [{
            "conflictId": conflict.conflict_id,
            "resolvedDescription": "Main entry point with merged CLI and UI features."
        }]
        
        result = await merge_handler_with_data.complete_merge_phase2(
            session.session_id, resolutions
        )
        
        assert result["success"] is True
        assert result["totalConflicts"] == 1
        assert result["resolvedConflicts"] == 1
        assert result["mergedFiles"] >= 1
        assert "main" in result["targetBranch"]
        
        # Verify the resolution was applied
        merged_desc = await merge_handler_with_data.db_manager.get_file_description(
            sample_project.id, "main", "src/main.py"
        )
        assert merged_desc is not None
        assert "merged CLI and UI features" in merged_desc.description
    
    async def test_merge_phase2_invalid_session(self, merge_handler_with_data):
        """Test merge phase 2 with invalid session ID."""
        with pytest.raises(ValidationError, match="Merge session not found"):
            await merge_handler_with_data.complete_merge_phase2(
                "nonexistent_session", []
            )
    
    async def test_merge_phase2_incomplete_resolutions(self, merge_handler_with_data, sample_project):
        """Test merge phase 2 with incomplete conflict resolutions."""
        # Start phase 1
        session = await merge_handler_with_data.start_merge_phase1(
            sample_project.id, "feature/new-ui", "main"
        )
        
        # Provide no resolutions
        with pytest.raises(ValidationError, match="Not all conflicts resolved"):
            await merge_handler_with_data.complete_merge_phase2(
                session.session_id, []
            )
    
    async def test_session_management(self, merge_handler, sample_project):
        """Test session management functionality."""
        # Create a session
        session = await merge_handler.start_merge_phase1(
            sample_project.id, "main", "develop"
        )
        
        # Test getting session
        retrieved = merge_handler.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id
        
        # Test active sessions
        active = merge_handler.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_id == session.session_id
        
        # Test aborting session
        success = merge_handler.abort_session(session.session_id)
        assert success is True
        
        # Session should be gone
        retrieved = merge_handler.get_session(session.session_id)
        assert retrieved is None
        
        active = merge_handler.get_active_sessions()
        assert len(active) == 0
    
    def test_cleanup_old_sessions(self, merge_handler):
        """Test cleanup of old sessions."""
        # This is a unit test that doesn't require async
        # Create some mock sessions and test cleanup logic
        old_session = MergeSession("project1", "branch1", "branch2")
        old_session.created = datetime.utcnow() - datetime.timedelta(days=2)
        
        new_session = MergeSession("project2", "branch3", "branch4")
        
        merge_handler._active_sessions[old_session.session_id] = old_session
        merge_handler._active_sessions[new_session.session_id] = new_session
        
        # Clean up sessions older than 1 day
        cleaned = merge_handler.cleanup_old_sessions(max_age_hours=24)
        
        assert cleaned == 1
        assert old_session.session_id not in merge_handler._active_sessions
        assert new_session.session_id in merge_handler._active_sessions
    
    async def test_merge_new_files_only(self, merge_handler, sample_project):
        """Test merging when source has new files not in target."""
        # Create a file only in source branch
        new_file_desc = create_test_file_description(
            project_id=sample_project.id,
            branch="feature/new-feature",
            file_path="src/new_feature.py",
            description="Brand new feature implementation"
        )
        
        await merge_handler.db_manager.create_file_description(new_file_desc)
        
        # Start merge
        session = await merge_handler.start_merge_phase1(
            sample_project.id, "feature/new-feature", "main"
        )
        
        # Should have no conflicts since file doesn't exist in target
        assert session.get_conflict_count() == 0
    
    async def test_merge_with_upstream_source(self, merge_handler, sample_project):
        """Test merging files that have upstream source tracking."""
        # Create file with upstream source
        upstream_file = create_test_file_description(
            project_id=sample_project.id,
            branch="source-branch",
            file_path="upstream_file.py",
            description="File from upstream",
        )
        upstream_file.source_project_id = "upstream_project_id"
        
        await merge_handler.db_manager.create_file_description(upstream_file)
        
        session = await merge_handler.start_merge_phase1(
            sample_project.id, "source-branch", "target-branch"
        )
        
        # Complete merge with no conflicts
        assert session.get_conflict_count() == 0


@pytest.mark.integration
class TestMergeIntegration:
    """Integration tests for merge functionality."""
    
    async def test_full_merge_workflow(self, merge_handler, sample_project):
        """Test the complete merge workflow from start to finish."""
        # Setup: Create conflicting descriptions
        descriptions = [
            create_test_file_description(
                project_id=sample_project.id,
                branch="feature",
                file_path="shared.py",
                description="Feature version of shared functionality"
            ),
            create_test_file_description(
                project_id=sample_project.id,
                branch="main",
                file_path="shared.py", 
                description="Main version of shared functionality"
            ),
            create_test_file_description(
                project_id=sample_project.id,
                branch="feature",
                file_path="feature_only.py",
                description="Feature-specific file"
            )
        ]
        
        await merge_handler.db_manager.batch_create_file_descriptions(descriptions)
        
        # Phase 1: Detect conflicts
        session = await merge_handler.start_merge_phase1(
            sample_project.id, "feature", "main"
        )
        
        assert session.get_conflict_count() == 1
        conflict = session.conflicts[0]
        assert conflict.file_path == "shared.py"
        
        # Phase 2: Resolve and merge
        resolutions = [{
            "conflictId": conflict.conflict_id,
            "resolvedDescription": "Merged shared functionality with feature enhancements"
        }]
        
        result = await merge_handler.complete_merge_phase2(
            session.session_id, resolutions
        )
        
        assert result["success"] is True
        assert result["totalConflicts"] == 1
        assert result["mergedFiles"] == 2  # shared.py + feature_only.py
        
        # Verify final state
        merged_shared = await merge_handler.db_manager.get_file_description(
            sample_project.id, "main", "shared.py"
        )
        assert "Merged shared functionality" in merged_shared.description
        
        merged_feature_only = await merge_handler.db_manager.get_file_description(
            sample_project.id, "main", "feature_only.py"
        )
        assert merged_feature_only is not None
        assert "Feature-specific file" in merged_feature_only.description
        
        # Verify session was cleaned up
        assert merge_handler.get_session(session.session_id) is None


@pytest.mark.performance
class TestMergePerformance:
    """Performance tests for merge operations."""
    
    async def test_large_merge_performance(self, merge_handler, sample_project):
        """Test merge performance with large number of files."""
        import time
        
        # Create many files in both branches
        descriptions = []
        for i in range(100):
            descriptions.extend([
                create_test_file_description(
                    project_id=sample_project.id,
                    branch="source",
                    file_path=f"file_{i:03d}.py",
                    description=f"Source description for file {i}"
                ),
                create_test_file_description(
                    project_id=sample_project.id,
                    branch="target",
                    file_path=f"file_{i:03d}.py",
                    description=f"Target description for file {i}"
                )
            ])
        
        await merge_handler.db_manager.batch_create_file_descriptions(descriptions)
        
        # Time the merge detection
        start_time = time.time()
        session = await merge_handler.start_merge_phase1(
            sample_project.id, "source", "target"
        )
        detection_time = time.time() - start_time
        
        assert session.get_conflict_count() == 100
        # Should complete conflict detection in reasonable time
        assert detection_time < 5.0  # 5 seconds max
        
        # Create resolutions for all conflicts
        resolutions = []
        for conflict in session.conflicts:
            resolutions.append({
                "conflictId": conflict.conflict_id,
                "resolvedDescription": f"Resolved: {conflict.file_path}"
            })
        
        # Time the merge completion
        start_time = time.time()
        result = await merge_handler.complete_merge_phase2(
            session.session_id, resolutions
        )
        completion_time = time.time() - start_time
        
        assert result["success"] is True
        # Should complete merge in reasonable time
        assert completion_time < 10.0  # 10 seconds max
