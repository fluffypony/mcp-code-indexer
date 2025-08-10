"""
AST-based code chunking for vector mode.

Provides semantic code chunking using Tree-sitter parsers to extract
meaningful code units for embedding generation.
"""

from .ast_chunker import ASTChunker, CodeChunk
from .language_handlers import LanguageHandler, get_language_handler
from .chunk_optimizer import ChunkOptimizer, OptimizedChunk

__all__ = [
    "ASTChunker",
    "CodeChunk", 
    "LanguageHandler",
    "get_language_handler",
    "ChunkOptimizer",
    "OptimizedChunk",
]
