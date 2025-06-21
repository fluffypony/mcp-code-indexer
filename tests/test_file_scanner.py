"""
Tests for file scanner functionality.

Tests file discovery, gitignore integration, and pattern-based filtering
to ensure correct identification of trackable files.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.mcp_code_indexer.file_scanner import FileScanner, DEFAULT_IGNORE_PATTERNS, IGNORED_EXTENSIONS


class TestFileScanner:
    """Test cases for FileScanner class."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        self.scanner = FileScanner(self.project_root)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_file(self, relative_path: str, content: str = "test content") -> Path:
        """Create a test file with given content."""
        file_path = self.project_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    def test_initialization(self):
        """Test FileScanner initialization."""
        # Compare resolved paths since FileScanner resolves the path
        assert self.scanner.project_root.resolve() == self.project_root.resolve()
        assert isinstance(self.scanner._gitignore_cache, dict)
    
    def test_default_ignore_patterns(self):
        """Test that default ignore patterns work correctly."""
        # Create files that should be ignored
        ignored_files = [
            "node_modules/package.json",
            ".git/config",
            "__pycache__/module.pyc",
            "build/output.bin",
            ".vscode/settings.json",
            "test.pyc",
            "temp.log",
        ]
        
        # Create files that should not be ignored
        tracked_files = [
            "src/main.py",
            "README.md",
            "package.json",
            "config.yaml",
        ]
        
        # Create all test files
        for file_path in ignored_files + tracked_files:
            self.create_test_file(file_path)
        
        # Test scanning
        found_files = self.scanner.scan_directory()
        found_relative = [self.scanner.get_relative_path(f) for f in found_files]
        
        # Check that tracked files are found
        for tracked_file in tracked_files:
            assert tracked_file in found_relative, f"Expected {tracked_file} to be tracked"
        
        # Check that ignored files are not found
        for ignored_file in ignored_files:
            assert ignored_file not in found_relative, f"Expected {ignored_file} to be ignored"
    
    def test_ignored_extensions(self):
        """Test that files with ignored extensions are filtered out."""
        # Create files with ignored extensions
        ignored_files = [
            "image.png",
            "document.pdf",
            "archive.zip",
            "binary.exe",
            "media.mp4",
        ]
        
        # Create files with trackable extensions
        tracked_files = [
            "script.py",
            "config.json",
            "readme.txt",
            "style.css",
            "code.js",
        ]
        
        for file_path in ignored_files + tracked_files:
            self.create_test_file(file_path)
        
        found_files = self.scanner.scan_directory()
        found_relative = [self.scanner.get_relative_path(f) for f in found_files]
        
        # Check results
        for tracked_file in tracked_files:
            assert tracked_file in found_relative
        
        for ignored_file in ignored_files:
            assert ignored_file not in found_relative
    
    def test_gitignore_integration(self):
        """Test .gitignore file integration."""
        # Create .gitignore file
        gitignore_content = """
# Ignore specific files
secret.txt
temp_*

# Ignore directory
private/
*.backup
"""
        self.create_test_file(".gitignore", gitignore_content)
        
        # Create files that should be ignored by .gitignore
        ignored_files = [
            "secret.txt",
            "temp_file.py",
            "temp_data.json",
            "private/config.py",
            "private/data/info.txt",
            "backup.backup",
        ]
        
        # Create files that should not be ignored
        tracked_files = [
            "public.txt",
            "main.py",
            "config.json",
            "public/readme.md",
        ]
        
        for file_path in ignored_files + tracked_files:
            self.create_test_file(file_path)
        
        # Reload gitignore patterns
        self.scanner._load_gitignore_patterns()
        
        found_files = self.scanner.scan_directory()
        found_relative = [self.scanner.get_relative_path(f) for f in found_files]
        
        # .gitignore itself should be tracked
        assert ".gitignore" in found_relative
        
        # Check tracked files are found
        for tracked_file in tracked_files:
            assert tracked_file in found_relative, f"Expected {tracked_file} to be tracked"
        
        # Check ignored files are not found
        for ignored_file in ignored_files:
            assert ignored_file not in found_relative, f"Expected {ignored_file} to be ignored"
    
    def test_relative_path_conversion(self):
        """Test relative path conversion."""
        test_file = self.create_test_file("src/utils/helper.py")
        
        relative_path = self.scanner.get_relative_path(test_file)
        assert relative_path == "src/utils/helper.py"
        
        # Test file outside project root
        outside_file = Path("/tmp/outside.py")
        relative_path = self.scanner.get_relative_path(outside_file)
        assert relative_path == str(outside_file)
    
    def test_find_missing_files(self):
        """Test finding files missing descriptions."""
        # Create test files
        all_files = [
            "src/main.py",
            "src/utils.py",
            "tests/test_main.py",
            "README.md",
            "config.json",
        ]
        
        for file_path in all_files:
            self.create_test_file(file_path)
        
        # Simulate some files already having descriptions
        existing_paths = {
            "src/main.py",
            "README.md",
        }
        
        missing_files = self.scanner.find_missing_files(existing_paths)
        missing_relative = [self.scanner.get_relative_path(f) for f in missing_files]
        
        expected_missing = [
            "src/utils.py",
            "tests/test_main.py", 
            "config.json",
        ]
        
        for expected in expected_missing:
            assert expected in missing_relative
        
        # Files with descriptions should not be in missing
        for existing in existing_paths:
            assert existing not in missing_relative
    
    def test_max_files_limit(self):
        """Test max_files parameter in scan_directory."""
        # Create many test files
        for i in range(20):
            self.create_test_file(f"file_{i:02d}.py")
        
        # Test with limit
        found_files = self.scanner.scan_directory(max_files=10)
        assert len(found_files) == 10
        
        # Test without limit
        all_files = self.scanner.scan_directory()
        assert len(all_files) == 20
    
    def test_should_ignore_file(self):
        """Test individual file ignore checking."""
        # Create test files
        tracked_file = self.create_test_file("main.py")
        ignored_file = self.create_test_file("cache.pyc")
        
        assert not self.scanner.should_ignore_file(tracked_file)
        assert self.scanner.should_ignore_file(ignored_file)
        
        # Test directory (should be ignored)
        test_dir = self.project_root / "test_dir"
        test_dir.mkdir()
        assert self.scanner.should_ignore_file(test_dir)
    
    def test_project_validation(self):
        """Test project directory validation."""
        # Valid directory
        assert self.scanner.is_valid_project_directory()
        
        # Invalid directory
        invalid_scanner = FileScanner(Path("/nonexistent/path"))
        assert not invalid_scanner.is_valid_project_directory()
    
    def test_project_stats(self):
        """Test project statistics gathering."""
        # Create diverse test files
        files_to_create = [
            ("src/main.py", "python code"),
            ("README.md", "documentation"),
            ("config.json", "{}"),
            ("image.png", "binary data"),  # Will be ignored
            ("node_modules/lib.js", "library"),  # Will be ignored
        ]
        
        for file_path, content in files_to_create:
            self.create_test_file(file_path, content)
        
        stats = self.scanner.get_project_stats()
        
        assert stats['total_files'] == 5
        assert stats['trackable_files'] == 3  # main.py, README.md, config.json
        assert stats['ignored_files'] == 2   # image.png, node_modules/lib.js
        assert '.py' in stats['file_extensions']
        assert '.md' in stats['file_extensions']
        assert '.json' in stats['file_extensions']
        assert stats['largest_file_size'] > 0
    
    @patch('src.file_scanner.parse_gitignore', None)
    def test_without_gitignore_parser(self):
        """Test functionality when gitignore_parser is not available."""
        scanner = FileScanner(self.project_root)
        
        # Should work with default patterns only
        test_file = self.create_test_file("main.py")
        ignored_file = self.create_test_file("cache.pyc")
        
        found_files = scanner.scan_directory()
        found_relative = [scanner.get_relative_path(f) for f in found_files]
        
        assert "main.py" in found_relative
        assert "cache.pyc" not in found_relative
    
    def test_nested_gitignore(self):
        """Test that gitignore from project root works correctly."""
        # Create root .gitignore with pattern that applies to subdirectories
        self.create_test_file(".gitignore", "*.tmp\n*.local\n")
        
        # Create test files
        files = [
            "root.tmp",      # Ignored by root .gitignore
            "root.txt",      # Not ignored
            "subdir/file.local",  # Ignored by root .gitignore pattern
            "subdir/file.txt",    # Not ignored
        ]
        
        for file_path in files:
            self.create_test_file(file_path)
        
        # Reload patterns
        self.scanner._load_gitignore_patterns()
        
        found_files = self.scanner.scan_directory()
        found_relative = [self.scanner.get_relative_path(f) for f in found_files]
        
        # Check results
        assert "root.txt" in found_relative
        assert "subdir/file.txt" in found_relative
        assert "root.tmp" not in found_relative
        assert "subdir/file.local" not in found_relative


if __name__ == "__main__":
    pytest.main([__file__])
