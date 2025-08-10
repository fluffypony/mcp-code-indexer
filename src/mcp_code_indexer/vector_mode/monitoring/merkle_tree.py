"""
Merkle tree implementation for efficient change detection.

Provides a hierarchical hash tree for detecting file system changes
without scanning entire directory structures.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os

from ...database.models import MerkleNode as MerkleNodeModel, NodeType

logger = logging.getLogger(__name__)

@dataclass
class TreeStats:
    """Statistics about the Merkle tree."""
    total_nodes: int = 0
    file_nodes: int = 0
    directory_nodes: int = 0
    max_depth: int = 0
    total_size: int = 0
    last_updated: Optional[datetime] = None

class MerkleNode:
    """
    Node in the Merkle tree representing a file or directory.
    """
    
    def __init__(
        self,
        path: str,
        node_type: NodeType,
        hash_value: Optional[str] = None,
        parent: Optional["MerkleNode"] = None,
    ):
        """
        Initialize a Merkle tree node.
        
        Args:
            path: Relative path from project root
            node_type: Type of node (file or directory)
            hash_value: Hash value for the node
            parent: Parent node
        """
        self.path = path
        self.node_type = node_type
        self.hash_value = hash_value
        self.parent = parent
        self.children: Dict[str, "MerkleNode"] = {}
        self.last_modified = datetime.utcnow()
        self.size: Optional[int] = None
        self.metadata: Dict[str, any] = {}
    
    def add_child(self, name: str, child: "MerkleNode") -> None:
        """Add a child node."""
        child.parent = self
        self.children[name] = child
    
    def remove_child(self, name: str) -> Optional["MerkleNode"]:
        """Remove and return a child node."""
        return self.children.pop(name, None)
    
    def get_child(self, name: str) -> Optional["MerkleNode"]:
        """Get a child node by name."""
        return self.children.get(name)
    
    def compute_hash(self, project_root: Path) -> str:
        """Compute hash for this node."""
        if self.node_type == NodeType.FILE:
            return self._compute_file_hash(project_root)
        else:
            return self._compute_directory_hash()
    
    def _compute_file_hash(self, project_root: Path) -> str:
        """Compute hash for a file node."""
        file_path = project_root / self.path
        
        try:
            if not file_path.exists():
                return "deleted"
            
            # Use file modification time and size for quick comparison
            stat = file_path.stat()
            self.size = stat.st_size
            
            # For small files, use content hash
            if stat.st_size < 1024 * 1024:  # 1MB
                with open(file_path, 'rb') as f:
                    content = f.read()
                return hashlib.sha256(content).hexdigest()
            else:
                # For large files, use metadata hash
                metadata = f"{stat.st_size}:{stat.st_mtime}"
                return hashlib.sha256(metadata.encode()).hexdigest()
                
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return "error"
    
    def _compute_directory_hash(self) -> str:
        """Compute hash for a directory node based on children."""
        if not self.children:
            return hashlib.sha256(b"empty").hexdigest()
        
        # Sort children by name for consistent hashing
        child_hashes = []
        for name in sorted(self.children.keys()):
            child = self.children[name]
            child_hash = child.hash_value or ""
            combined = f"{name}:{child_hash}"
            child_hashes.append(combined)
        
        combined_hash = "|".join(child_hashes)
        return hashlib.sha256(combined_hash.encode()).hexdigest()
    
    def update_hash(self, project_root: Path) -> bool:
        """Update hash and return True if it changed."""
        old_hash = self.hash_value
        new_hash = self.compute_hash(project_root)
        
        if old_hash != new_hash:
            self.hash_value = new_hash
            self.last_modified = datetime.utcnow()
            return True
        
        return False
    
    def get_depth(self) -> int:
        """Get depth of this node in the tree."""
        if self.parent is None:
            return 0
        return self.parent.get_depth() + 1
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary representation."""
        return {
            "path": self.path,
            "node_type": self.node_type.value,
            "hash_value": self.hash_value,
            "last_modified": self.last_modified.isoformat(),
            "size": self.size,
            "children": list(self.children.keys()),
            "metadata": self.metadata,
        }

class MerkleTree:
    """
    Merkle tree for efficient file system change detection.
    
    Maintains a hierarchical hash tree of the project structure to quickly
    identify changes without scanning all files.
    """
    
    def __init__(self, project_root: Path, project_id: str):
        """
        Initialize Merkle tree.
        
        Args:
            project_root: Root directory path
            project_id: Project identifier
        """
        self.project_root = Path(project_root).resolve()
        self.project_id = project_id
        self.root: Optional[MerkleNode] = None
        self.node_map: Dict[str, MerkleNode] = {}  # path -> node mapping
        
        # Statistics
        self.stats = TreeStats()
        
        # Change tracking
        self.changed_nodes: Set[str] = set()
        self.last_scan_time: Optional[datetime] = None
    
    def build_tree(self, ignore_patterns: Optional[List[str]] = None) -> None:
        """Build the complete Merkle tree by scanning the file system."""
        logger.info(f"Building Merkle tree for {self.project_root}")
        
        ignore_patterns = ignore_patterns or [
            "*.log", "*.tmp", "*~", ".git", "__pycache__", 
            "node_modules", "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db"
        ]
        
        # Create root node
        self.root = MerkleNode("", NodeType.DIRECTORY)
        self.node_map[""] = self.root
        
        # Recursively build tree
        self._build_tree_recursive(self.project_root, self.root, ignore_patterns)
        
        # Compute hashes bottom-up
        self._compute_hashes_recursive(self.root)
        
        # Update statistics
        self._update_stats()
        self.last_scan_time = datetime.utcnow()
        
        logger.info(
            f"Built Merkle tree: {self.stats.total_nodes} nodes "
            f"({self.stats.file_nodes} files, {self.stats.directory_nodes} directories)"
        )
    
    def _build_tree_recursive(
        self,
        current_path: Path,
        current_node: MerkleNode,
        ignore_patterns: List[str]
    ) -> None:
        """Recursively build tree structure."""
        try:
            if not current_path.is_dir():
                return
            
            for item in current_path.iterdir():
                # Check if should ignore
                if self._should_ignore(item, ignore_patterns):
                    continue
                
                # Get relative path
                try:
                    relative_path = str(item.relative_to(self.project_root))
                except ValueError:
                    continue
                
                # Create node
                if item.is_file():
                    node = MerkleNode(relative_path, NodeType.FILE)
                    current_node.add_child(item.name, node)
                    self.node_map[relative_path] = node
                    
                elif item.is_dir():
                    node = MerkleNode(relative_path, NodeType.DIRECTORY)
                    current_node.add_child(item.name, node)
                    self.node_map[relative_path] = node
                    
                    # Recurse into directory
                    self._build_tree_recursive(item, node, ignore_patterns)
                    
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not scan directory {current_path}: {e}")
    
    def _should_ignore(self, path: Path, ignore_patterns: List[str]) -> bool:
        """Check if path should be ignored."""
        import fnmatch
        
        path_str = path.name
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        
        return False
    
    def _compute_hashes_recursive(self, node: MerkleNode) -> None:
        """Compute hashes recursively (bottom-up)."""
        # First compute hashes for all children
        for child in node.children.values():
            self._compute_hashes_recursive(child)
        
        # Then compute hash for this node
        node.update_hash(self.project_root)
    
    def _update_stats(self) -> None:
        """Update tree statistics."""
        self.stats = TreeStats()
        self.stats.last_updated = datetime.utcnow()
        
        if self.root:
            self._update_stats_recursive(self.root, 0)
    
    def _update_stats_recursive(self, node: MerkleNode, depth: int) -> None:
        """Recursively update statistics."""
        self.stats.total_nodes += 1
        self.stats.max_depth = max(self.stats.max_depth, depth)
        
        if node.node_type == NodeType.FILE:
            self.stats.file_nodes += 1
            if node.size:
                self.stats.total_size += node.size
        else:
            self.stats.directory_nodes += 1
        
        for child in node.children.values():
            self._update_stats_recursive(child, depth + 1)
    
    def update_file(self, relative_path: str) -> bool:
        """
        Update a file in the tree and return True if hash changed.
        
        Args:
            relative_path: Path relative to project root
            
        Returns:
            True if the file's hash changed
        """
        node = self.node_map.get(relative_path)
        if not node:
            # File might be new, rebuild tree
            logger.info(f"File {relative_path} not in tree, triggering rebuild")
            self.build_tree()
            return True
        
        # Update file hash
        changed = node.update_hash(self.project_root)
        
        if changed:
            self.changed_nodes.add(relative_path)
            
            # Propagate hash changes up the tree
            self._propagate_hash_changes(node.parent)
        
        return changed
    
    def _propagate_hash_changes(self, node: Optional[MerkleNode]) -> None:
        """Propagate hash changes up the tree."""
        if not node:
            return
        
        old_hash = node.hash_value
        node.update_hash(self.project_root)
        
        if old_hash != node.hash_value:
            self.changed_nodes.add(node.path)
            self._propagate_hash_changes(node.parent)
    
    def get_changed_files(self, since: Optional[datetime] = None) -> List[str]:
        """Get list of files that changed since timestamp."""
        if since is None:
            return list(self.changed_nodes)
        
        changed_files = []
        for path in self.changed_nodes:
            node = self.node_map.get(path)
            if node and node.last_modified >= since:
                changed_files.append(path)
        
        return changed_files
    
    def verify_tree(self) -> Tuple[bool, List[str]]:
        """
        Verify tree integrity by recomputing hashes.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not self.root:
            return False, ["No root node"]
        
        # Recompute all hashes and compare
        for path, node in self.node_map.items():
            expected_hash = node.compute_hash(self.project_root)
            if node.hash_value != expected_hash:
                errors.append(f"Hash mismatch for {path}: {node.hash_value} != {expected_hash}")
        
        return len(errors) == 0, errors
    
    def get_subtree_hash(self, relative_path: str) -> Optional[str]:
        """Get hash for a subtree rooted at the given path."""
        node = self.node_map.get(relative_path)
        return node.hash_value if node else None
    
    def export_to_database_models(self) -> List[MerkleNodeModel]:
        """Export tree to database models for persistence."""
        models = []
        
        for path, node in self.node_map.items():
            # Determine parent path
            parent_path = None
            if node.parent and node.parent.path:
                parent_path = node.parent.path
            
            model = MerkleNodeModel(
                project_id=self.project_id,
                path=path,
                hash=node.hash_value or "",
                node_type=node.node_type,
                parent_path=parent_path,
                children_hash=node._compute_directory_hash() if node.node_type == NodeType.DIRECTORY else None,
                last_modified=node.last_modified,
            )
            models.append(model)
        
        return models
    
    def clear_changed_nodes(self) -> int:
        """Clear the changed nodes set and return count."""
        count = len(self.changed_nodes)
        self.changed_nodes.clear()
        return count
    
    def get_tree_summary(self) -> Dict:
        """Get a summary of the tree structure."""
        return {
            "project_id": self.project_id,
            "project_root": str(self.project_root),
            "stats": {
                "total_nodes": self.stats.total_nodes,
                "file_nodes": self.stats.file_nodes,
                "directory_nodes": self.stats.directory_nodes,
                "max_depth": self.stats.max_depth,
                "total_size": self.stats.total_size,
                "last_updated": self.stats.last_updated.isoformat() if self.stats.last_updated else None,
            },
            "root_hash": self.root.hash_value if self.root else None,
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "changed_nodes": len(self.changed_nodes),
        }
