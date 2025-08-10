"""
Main AST-based code chunker for vector mode.

Coordinates language-specific parsing and produces optimized code chunks
for embedding generation while preserving semantic meaning.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from .language_handlers import get_language_handler, ParsedChunk
from .chunk_optimizer import ChunkOptimizer, OptimizedChunk
from ..security.redactor import SecretRedactor, RedactionResult
from ...database.models import ChunkType

logger = logging.getLogger(__name__)

@dataclass
class CodeChunk:
    """
    Represents a code chunk ready for embedding generation.
    
    This is the final output of the chunking process, optimized and
    ready for vector indexing.
    """
    content: str
    chunk_type: ChunkType
    name: Optional[str]
    file_path: str
    start_line: int
    end_line: int
    content_hash: str
    language: str
    redacted: bool = False
    metadata: Dict[str, Any] = None
    imports: List[str] = None
    parent_context: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.imports is None:
            self.imports = []

@dataclass
class ChunkingStats:
    """Statistics about the chunking process."""
    files_processed: int = 0
    total_chunks: int = 0
    chunks_by_type: Dict[ChunkType, int] = None
    chunks_by_language: Dict[str, int] = None
    redacted_chunks: int = 0
    fallback_chunks: int = 0
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.chunks_by_type is None:
            self.chunks_by_type = {}
        if self.chunks_by_language is None:
            self.chunks_by_language = {}

class ASTChunker:
    """
    Main AST-based code chunker.
    
    Orchestrates the entire chunking process from file content to
    optimized code chunks ready for embedding generation.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 50,
        enable_redaction: bool = True,
        enable_optimization: bool = True,
        redaction_confidence: float = 0.5,
    ):
        """
        Initialize AST chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            enable_redaction: Whether to redact secrets
            enable_optimization: Whether to optimize chunks
            redaction_confidence: Confidence threshold for redaction
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.enable_redaction = enable_redaction
        self.enable_optimization = enable_optimization
        
        # Initialize components
        self.redactor: Optional[SecretRedactor] = None
        if enable_redaction:
            self.redactor = SecretRedactor(
                min_confidence=redaction_confidence,
                preserve_structure=True,
            )
        
        self.optimizer: Optional[ChunkOptimizer] = None
        if enable_optimization:
            self.optimizer = ChunkOptimizer(
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
            )
        
        # Statistics
        self.stats = ChunkingStats()
        
        # Cache for performance
        self.handler_cache: Dict[str, Any] = {}
    
    def chunk_file(self, file_path: str, content: Optional[str] = None) -> List[CodeChunk]:
        """
        Chunk a single file into semantic code chunks.
        
        Args:
            file_path: Path to the file to chunk
            content: Optional file content (if not provided, will read from file)
            
        Returns:
            List of code chunks
        """
        start_time = datetime.utcnow()
        
        try:
            # Read content if not provided
            if content is None:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            # Skip empty files
            if not content.strip():
                logger.debug(f"Skipping empty file: {file_path}")
                return []
            
            # Get language handler
            handler = self._get_language_handler(file_path)
            if not handler:
                logger.warning(f"No handler available for {file_path}")
                return []
            
            # Parse into semantic chunks
            logger.debug(f"Parsing {file_path} with {handler.language_name} handler")
            parsed_chunks = handler.parse_code(content, file_path)
            
            # Convert to code chunks
            code_chunks = []
            for parsed_chunk in parsed_chunks:
                code_chunk = self._convert_parsed_chunk(parsed_chunk, file_path)
                if code_chunk:
                    code_chunks.append(code_chunk)
            
            # Apply redaction if enabled
            if self.enable_redaction and self.redactor:
                code_chunks = self._apply_redaction(code_chunks, file_path)
            
            # Apply optimization if enabled
            if self.enable_optimization and self.optimizer:
                code_chunks = self._apply_optimization(code_chunks)
            
            # Update statistics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_stats(code_chunks, handler.language_name, processing_time)
            
            logger.info(f"Chunked {file_path}: {len(code_chunks)} chunks")
            return code_chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk file {file_path}: {e}")
            return []
    
    def chunk_content(
        self,
        content: str,
        file_path: str,
        language: Optional[str] = None
    ) -> List[CodeChunk]:
        """
        Chunk content directly without reading from file.
        
        Args:
            content: Source code content
            file_path: Virtual file path for language detection
            language: Optional language override
            
        Returns:
            List of code chunks
        """
        return self.chunk_file(file_path, content)
    
    def chunk_multiple_files(self, file_paths: List[str]) -> Dict[str, List[CodeChunk]]:
        """
        Chunk multiple files and return results grouped by file.
        
        Args:
            file_paths: List of file paths to chunk
            
        Returns:
            Dictionary mapping file paths to their chunks
        """
        results = {}
        
        for file_path in file_paths:
            try:
                chunks = self.chunk_file(file_path)
                results[file_path] = chunks
            except Exception as e:
                logger.error(f"Failed to chunk {file_path}: {e}")
                results[file_path] = []
        
        return results
    
    def _get_language_handler(self, file_path: str) -> Optional[Any]:
        """Get language handler for file, with caching."""
        extension = Path(file_path).suffix.lower()
        
        if extension in self.handler_cache:
            return self.handler_cache[extension]
        
        handler = get_language_handler(file_path)
        self.handler_cache[extension] = handler
        return handler
    
    def _convert_parsed_chunk(self, parsed_chunk: ParsedChunk, file_path: str) -> Optional[CodeChunk]:
        """Convert a parsed chunk to a code chunk."""
        if not parsed_chunk.content.strip():
            return None
        
        # Generate content hash
        content_hash = hashlib.sha256(parsed_chunk.content.encode('utf-8')).hexdigest()
        
        # Create code chunk
        code_chunk = CodeChunk(
            content=parsed_chunk.content,
            chunk_type=parsed_chunk.chunk_type,
            name=parsed_chunk.name,
            file_path=file_path,
            start_line=parsed_chunk.start_line,
            end_line=parsed_chunk.end_line,
            content_hash=content_hash,
            language=parsed_chunk.language,
            metadata=parsed_chunk.metadata.copy(),
            imports=parsed_chunk.imports.copy() if parsed_chunk.imports else [],
            parent_context=parsed_chunk.parent_context,
        )
        
        return code_chunk
    
    def _apply_redaction(self, chunks: List[CodeChunk], file_path: str) -> List[CodeChunk]:
        """Apply secret redaction to chunks."""
        redacted_chunks = []
        
        for chunk in chunks:
            try:
                redaction_result = self.redactor.redact_content(
                    content=chunk.content,
                    file_path=file_path,
                )
                
                if redaction_result.was_redacted:
                    # Update chunk with redacted content
                    chunk.content = redaction_result.redacted_content
                    chunk.redacted = True
                    chunk.metadata["redaction_count"] = redaction_result.redaction_count
                    chunk.metadata["redacted_patterns"] = redaction_result.patterns_matched
                    
                    # Recompute hash for redacted content
                    chunk.content_hash = hashlib.sha256(
                        chunk.content.encode('utf-8')
                    ).hexdigest()
                    
                    logger.debug(f"Redacted {redaction_result.redaction_count} secrets from chunk {chunk.name}")
                
                redacted_chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"Failed to redact chunk {chunk.name}: {e}")
                redacted_chunks.append(chunk)
        
        return redacted_chunks
    
    def _apply_optimization(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Apply chunk optimization."""
        try:
            # Convert to optimized chunks
            optimized_chunks = []
            for chunk in chunks:
                opt_chunk = OptimizedChunk(
                    content=chunk.content,
                    chunk_type=chunk.chunk_type,
                    name=chunk.name,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    metadata=chunk.metadata,
                    language=chunk.language,
                    imports=chunk.imports,
                    parent_context=chunk.parent_context,
                )
                optimized_chunks.append(opt_chunk)
            
            # Apply optimization
            optimized_chunks = self.optimizer.optimize_chunks(optimized_chunks)
            
            # Convert back to code chunks
            result_chunks = []
            for opt_chunk in optimized_chunks:
                code_chunk = CodeChunk(
                    content=opt_chunk.content,
                    chunk_type=opt_chunk.chunk_type,
                    name=opt_chunk.name,
                    file_path=chunks[0].file_path if chunks else "",
                    start_line=opt_chunk.start_line,
                    end_line=opt_chunk.end_line,
                    content_hash=hashlib.sha256(opt_chunk.content.encode('utf-8')).hexdigest(),
                    language=opt_chunk.language,
                    metadata=opt_chunk.metadata,
                    imports=opt_chunk.imports,
                    parent_context=opt_chunk.parent_context,
                )
                result_chunks.append(code_chunk)
            
            return result_chunks
            
        except Exception as e:
            logger.warning(f"Chunk optimization failed: {e}")
            return chunks
    
    def _update_stats(self, chunks: List[CodeChunk], language: str, processing_time: float) -> None:
        """Update chunking statistics."""
        self.stats.files_processed += 1
        self.stats.total_chunks += len(chunks)
        self.stats.processing_time += processing_time
        
        # Count by type
        for chunk in chunks:
            self.stats.chunks_by_type[chunk.chunk_type] = (
                self.stats.chunks_by_type.get(chunk.chunk_type, 0) + 1
            )
            
            if chunk.redacted:
                self.stats.redacted_chunks += 1
            
            if chunk.metadata.get("fallback", False):
                self.stats.fallback_chunks += 1
        
        # Count by language
        self.stats.chunks_by_language[language] = (
            self.stats.chunks_by_language.get(language, 0) + len(chunks)
        )
    
    def get_stats(self) -> ChunkingStats:
        """Get chunking statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset chunking statistics."""
        self.stats = ChunkingStats()
    
    def get_supported_extensions(self) -> Set[str]:
        """Get list of supported file extensions."""
        from .language_handlers import LANGUAGE_HANDLERS
        return set(LANGUAGE_HANDLERS.keys())
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if a file is supported for chunking."""
        extension = Path(file_path).suffix.lower()
        return extension in self.get_supported_extensions()
    
    def estimate_chunks(self, file_path: str) -> Dict[str, Any]:
        """Estimate number of chunks for a file without full processing."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Simple estimation based on content length and average chunk size
            content_length = len(content)
            lines = content.count('\n') + 1
            
            # Rough estimates
            estimated_chunks = max(1, content_length // self.max_chunk_size)
            
            return {
                "file_path": file_path,
                "content_length": content_length,
                "line_count": lines,
                "estimated_chunks": estimated_chunks,
                "is_supported": self.is_supported_file(file_path),
            }
            
        except Exception as e:
            logger.warning(f"Failed to estimate chunks for {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "estimated_chunks": 0,
                "is_supported": False,
            }
