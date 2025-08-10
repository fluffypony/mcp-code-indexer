"""
Chunk optimization for vector mode.

Optimizes code chunks for embedding generation by combining small chunks,
splitting large chunks, and ensuring optimal token distribution.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ...database.models import ChunkType

logger = logging.getLogger(__name__)

@dataclass
class OptimizedChunk:
    """Represents an optimized code chunk."""
    content: str
    chunk_type: ChunkType
    name: Optional[str]
    start_line: int
    end_line: int
    metadata: Dict[str, Any]
    language: str
    imports: List[str] = None
    parent_context: Optional[str] = None
    optimization_applied: str = "none"
    
    def __post_init__(self):
        if self.imports is None:
            self.imports = []

class ChunkOptimizer:
    """
    Optimizes code chunks for better embedding quality.
    
    Applies various optimization strategies including:
    - Combining small related chunks
    - Splitting oversized chunks
    - Adding context from imports/parent scopes
    - Balancing chunk sizes
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 50,
        target_chunk_size: int = 800,
        context_window: int = 200,
        enable_context_enrichment: bool = True,
    ):
        """
        Initialize chunk optimizer.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk  
            target_chunk_size: Target size for optimal chunks
            context_window: Characters of context to add
            enable_context_enrichment: Whether to add context from imports/parent
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.target_chunk_size = target_chunk_size
        self.context_window = context_window
        self.enable_context_enrichment = enable_context_enrichment
        
        # Optimization statistics
        self.stats = {
            "chunks_processed": 0,
            "chunks_combined": 0,
            "chunks_split": 0,
            "chunks_enriched": 0,
            "optimization_time": 0.0,
        }
    
    def optimize_chunks(self, chunks: List[OptimizedChunk]) -> List[OptimizedChunk]:
        """
        Optimize a list of code chunks.
        
        Args:
            chunks: List of chunks to optimize
            
        Returns:
            Optimized list of chunks
        """
        start_time = datetime.utcnow()
        
        if not chunks:
            return chunks
        
        logger.debug(f"Optimizing {len(chunks)} chunks")
        
        # Step 1: Add context enrichment
        if self.enable_context_enrichment:
            chunks = self._enrich_with_context(chunks)
        
        # Step 2: Combine small chunks
        chunks = self._combine_small_chunks(chunks)
        
        # Step 3: Split oversized chunks
        chunks = self._split_large_chunks(chunks)
        
        # Step 4: Balance chunk sizes
        chunks = self._balance_chunks(chunks)
        
        # Update statistics
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        self.stats["chunks_processed"] += len(chunks)
        self.stats["optimization_time"] += processing_time
        
        logger.debug(f"Optimization complete: {len(chunks)} chunks")
        return chunks
    
    def _enrich_with_context(self, chunks: List[OptimizedChunk]) -> List[OptimizedChunk]:
        """Add context information to chunks for better embeddings."""
        enriched_chunks = []
        
        # Group chunks by file to share context
        chunks_by_file = {}
        for chunk in chunks:
            # Use language as a proxy for file grouping
            file_key = f"{chunk.language}_{chunk.parent_context or 'global'}"
            if file_key not in chunks_by_file:
                chunks_by_file[file_key] = []
            chunks_by_file[file_key].append(chunk)
        
        for file_chunks in chunks_by_file.values():
            # Find import chunks to use as context
            import_chunks = [c for c in file_chunks if c.chunk_type == ChunkType.IMPORT]
            import_context = ""
            
            if import_chunks:
                import_lines = []
                for imp_chunk in import_chunks:
                    import_lines.extend(imp_chunk.imports)
                
                if import_lines:
                    import_context = "\n".join(import_lines[:10]) + "\n\n"  # Limit imports
            
            # Enrich non-import chunks with context
            for chunk in file_chunks:
                if chunk.chunk_type != ChunkType.IMPORT and import_context:
                    # Add import context at the beginning
                    original_content = chunk.content
                    enriched_content = import_context + original_content
                    
                    # Only enrich if it doesn't exceed max size
                    if len(enriched_content) <= self.max_chunk_size:
                        chunk.content = enriched_content
                        chunk.optimization_applied = "context_enriched"
                        chunk.metadata["context_added"] = True
                        chunk.metadata["import_lines_added"] = len(import_context.split('\n')) - 2
                        self.stats["chunks_enriched"] += 1
                
                enriched_chunks.append(chunk)
        
        return enriched_chunks
    
    def _combine_small_chunks(self, chunks: List[OptimizedChunk]) -> List[OptimizedChunk]:
        """Combine small chunks that are related."""
        combined_chunks = []
        pending_combination = []
        
        def should_combine(chunk1: OptimizedChunk, chunk2: OptimizedChunk) -> bool:
            """Check if two chunks should be combined."""
            # Don't combine different types unless they're generic
            if (chunk1.chunk_type != chunk2.chunk_type and 
                chunk1.chunk_type != ChunkType.GENERIC and 
                chunk2.chunk_type != ChunkType.GENERIC):
                return False
            
            # Don't combine if they're from different contexts
            if chunk1.parent_context != chunk2.parent_context:
                return False
            
            # Don't combine if result would be too large
            combined_size = len(chunk1.content) + len(chunk2.content) + 2  # +2 for separator
            if combined_size > self.max_chunk_size:
                return False
            
            # Don't combine imports with other types
            if (chunk1.chunk_type == ChunkType.IMPORT or 
                chunk2.chunk_type == ChunkType.IMPORT):
                return False
            
            return True
        
        def combine_chunks(chunk_list: List[OptimizedChunk]) -> OptimizedChunk:
            """Combine a list of chunks into one."""
            if len(chunk_list) == 1:
                return chunk_list[0]
            
            # Combine content
            combined_content = "\n\n".join(chunk.content for chunk in chunk_list)
            
            # Use properties from first chunk as base
            base_chunk = chunk_list[0]
            
            # Combine metadata
            combined_metadata = base_chunk.metadata.copy()
            combined_metadata["combined_from"] = len(chunk_list)
            combined_metadata["original_chunks"] = [c.name for c in chunk_list if c.name]
            
            # Combine imports
            all_imports = []
            for chunk in chunk_list:
                all_imports.extend(chunk.imports)
            unique_imports = list(dict.fromkeys(all_imports))  # Preserve order, remove dupes
            
            return OptimizedChunk(
                content=combined_content,
                chunk_type=base_chunk.chunk_type,
                name=f"combined_{len(chunk_list)}_chunks",
                start_line=min(c.start_line for c in chunk_list),
                end_line=max(c.end_line for c in chunk_list),
                metadata=combined_metadata,
                language=base_chunk.language,
                imports=unique_imports,
                parent_context=base_chunk.parent_context,
                optimization_applied="combined",
            )
        
        for chunk in chunks:
            # Check if chunk is small
            if len(chunk.content) < self.min_chunk_size:
                # Try to combine with pending chunks
                can_combine = False
                if pending_combination:
                    last_chunk = pending_combination[-1]
                    if should_combine(last_chunk, chunk):
                        pending_combination.append(chunk)
                        can_combine = True
                
                if not can_combine:
                    # Flush pending combination if any
                    if pending_combination:
                        combined = combine_chunks(pending_combination)
                        combined_chunks.append(combined)
                        self.stats["chunks_combined"] += len(pending_combination) - 1
                    
                    # Start new combination
                    pending_combination = [chunk]
            else:
                # Flush pending combination
                if pending_combination:
                    combined = combine_chunks(pending_combination)
                    combined_chunks.append(combined)
                    self.stats["chunks_combined"] += len(pending_combination) - 1
                    pending_combination = []
                
                # Add regular chunk
                combined_chunks.append(chunk)
        
        # Flush any remaining pending combination
        if pending_combination:
            combined = combine_chunks(pending_combination)
            combined_chunks.append(combined)
            self.stats["chunks_combined"] += len(pending_combination) - 1
        
        return combined_chunks
    
    def _split_large_chunks(self, chunks: List[OptimizedChunk]) -> List[OptimizedChunk]:
        """Split chunks that are too large."""
        split_chunks = []
        
        for chunk in chunks:
            if len(chunk.content) <= self.max_chunk_size:
                split_chunks.append(chunk)
                continue
            
            # Split the chunk
            logger.debug(f"Splitting large chunk: {len(chunk.content)} chars")
            
            # Try to split at logical boundaries
            sub_chunks = self._split_chunk_intelligently(chunk)
            split_chunks.extend(sub_chunks)
            
            self.stats["chunks_split"] += len(sub_chunks) - 1
        
        return split_chunks
    
    def _split_chunk_intelligently(self, chunk: OptimizedChunk) -> List[OptimizedChunk]:
        """Split a chunk at intelligent boundaries."""
        content = chunk.content
        max_size = self.max_chunk_size
        
        # Try to split at natural boundaries
        split_points = self._find_split_points(content)
        
        if not split_points:
            # Fallback to simple line-based splitting
            return self._split_chunk_by_lines(chunk)
        
        sub_chunks = []
        start_idx = 0
        current_line = chunk.start_line
        
        for split_point in split_points:
            if split_point - start_idx > max_size:
                # This segment is too large, split it further
                sub_content = content[start_idx:split_point]
                sub_sub_chunks = self._split_content_by_size(
                    sub_content, chunk, current_line, max_size
                )
                sub_chunks.extend(sub_sub_chunks)
            else:
                # Create sub-chunk
                sub_content = content[start_idx:split_point]
                if sub_content.strip():
                    lines_in_chunk = sub_content.count('\n')
                    sub_chunk = OptimizedChunk(
                        content=sub_content,
                        chunk_type=chunk.chunk_type,
                        name=f"{chunk.name}_part_{len(sub_chunks) + 1}" if chunk.name else None,
                        start_line=current_line,
                        end_line=current_line + lines_in_chunk,
                        metadata=chunk.metadata.copy(),
                        language=chunk.language,
                        imports=chunk.imports.copy(),
                        parent_context=chunk.parent_context,
                        optimization_applied="split",
                    )
                    sub_chunks.append(sub_chunk)
                    current_line += lines_in_chunk + 1
            
            start_idx = split_point
        
        # Handle remaining content
        if start_idx < len(content):
            remaining_content = content[start_idx:]
            if remaining_content.strip():
                lines_in_chunk = remaining_content.count('\n')
                sub_chunk = OptimizedChunk(
                    content=remaining_content,
                    chunk_type=chunk.chunk_type,
                    name=f"{chunk.name}_part_{len(sub_chunks) + 1}" if chunk.name else None,
                    start_line=current_line,
                    end_line=current_line + lines_in_chunk,
                    metadata=chunk.metadata.copy(),
                    language=chunk.language,
                    imports=chunk.imports.copy(),
                    parent_context=chunk.parent_context,
                    optimization_applied="split",
                )
                sub_chunks.append(sub_chunk)
        
        return sub_chunks or [chunk]  # Return original if splitting failed
    
    def _find_split_points(self, content: str) -> List[int]:
        """Find intelligent split points in content."""
        split_points = []
        lines = content.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_with_newline = line + '\n' if i < len(lines) - 1 else line
            
            # Look for natural boundaries
            stripped = line.strip()
            
            # End of function/class/block
            if (stripped.startswith(('def ', 'class ', 'function ', 'const ', 'let ', 'var ')) or
                stripped.endswith(('{', '}', ';')) or
                not stripped):  # Empty lines
                
                split_points.append(current_pos + len(line_with_newline))
            
            current_pos += len(line_with_newline)
        
        return split_points
    
    def _split_chunk_by_lines(self, chunk: OptimizedChunk) -> List[OptimizedChunk]:
        """Fallback: split chunk by lines when intelligent splitting fails."""
        lines = chunk.content.split('\n')
        sub_chunks = []
        current_lines = []
        current_size = 0
        current_line_num = chunk.start_line
        start_line_num = chunk.start_line
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > self.max_chunk_size and current_lines:
                # Create sub-chunk
                sub_content = '\n'.join(current_lines)
                sub_chunk = OptimizedChunk(
                    content=sub_content,
                    chunk_type=chunk.chunk_type,
                    name=f"{chunk.name}_part_{len(sub_chunks) + 1}" if chunk.name else None,
                    start_line=start_line_num,
                    end_line=current_line_num,
                    metadata=chunk.metadata.copy(),
                    language=chunk.language,
                    imports=chunk.imports.copy(),
                    parent_context=chunk.parent_context,
                    optimization_applied="split_lines",
                )
                sub_chunks.append(sub_chunk)
                
                # Reset for next chunk
                current_lines = [line]
                current_size = line_size
                start_line_num = current_line_num + 1
            else:
                current_lines.append(line)
                current_size += line_size
            
            current_line_num += 1
        
        # Add final chunk
        if current_lines:
            sub_content = '\n'.join(current_lines)
            sub_chunk = OptimizedChunk(
                content=sub_content,
                chunk_type=chunk.chunk_type,
                name=f"{chunk.name}_part_{len(sub_chunks) + 1}" if chunk.name else None,
                start_line=start_line_num,
                end_line=current_line_num,
                metadata=chunk.metadata.copy(),
                language=chunk.language,
                imports=chunk.imports.copy(),
                parent_context=chunk.parent_context,
                optimization_applied="split_lines",
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _split_content_by_size(
        self,
        content: str,
        original_chunk: OptimizedChunk,
        start_line: int,
        max_size: int
    ) -> List[OptimizedChunk]:
        """Split content by size when other methods fail."""
        sub_chunks = []
        current_pos = 0
        chunk_num = 1
        
        while current_pos < len(content):
            end_pos = min(current_pos + max_size, len(content))
            
            # Try to end at a word boundary
            if end_pos < len(content):
                while end_pos > current_pos and content[end_pos] not in ' \n\t':
                    end_pos -= 1
                
                if end_pos == current_pos:  # No word boundary found
                    end_pos = min(current_pos + max_size, len(content))
            
            sub_content = content[current_pos:end_pos]
            if sub_content.strip():
                lines_in_chunk = sub_content.count('\n')
                sub_chunk = OptimizedChunk(
                    content=sub_content,
                    chunk_type=original_chunk.chunk_type,
                    name=f"{original_chunk.name}_size_part_{chunk_num}" if original_chunk.name else None,
                    start_line=start_line,
                    end_line=start_line + lines_in_chunk,
                    metadata=original_chunk.metadata.copy(),
                    language=original_chunk.language,
                    imports=original_chunk.imports.copy(),
                    parent_context=original_chunk.parent_context,
                    optimization_applied="split_size",
                )
                sub_chunks.append(sub_chunk)
                start_line += lines_in_chunk + 1
                chunk_num += 1
            
            current_pos = end_pos
        
        return sub_chunks
    
    def _balance_chunks(self, chunks: List[OptimizedChunk]) -> List[OptimizedChunk]:
        """Apply final balancing to chunks."""
        # For now, just return as-is
        # Future enhancements could include:
        # - Redistributing content between chunks
        # - Merging very small chunks
        # - Further splitting of slightly oversized chunks
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset optimization statistics."""
        self.stats = {
            "chunks_processed": 0,
            "chunks_combined": 0,
            "chunks_split": 0,
            "chunks_enriched": 0,
            "optimization_time": 0.0,
        }
