"""
File system monitoring for vector mode.

Provides real-time file change detection using watchdog and efficient
change tracking using Merkle trees.
"""

from .file_watcher import FileWatcher
from .merkle_tree import MerkleTree, MerkleNode
from .change_detector import ChangeDetector, FileChange, ChangeType

__all__ = [
    "FileWatcher",
    "MerkleTree", 
    "MerkleNode",
    "ChangeDetector",
    "FileChange",
    "ChangeType",
]
