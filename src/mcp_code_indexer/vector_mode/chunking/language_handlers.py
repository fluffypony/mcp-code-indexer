"""
Language-specific handlers for AST parsing and code chunking.

Provides specialized handling for different programming languages using
Tree-sitter parsers with language-specific semantic understanding.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import tree-sitter, fallback if not available
try:
    import tree_sitter
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Language = None
    Parser = None
    Node = None

from ...database.models import ChunkType

@dataclass
class ParsedChunk:
    """Represents a parsed code chunk with metadata."""
    content: str
    chunk_type: ChunkType
    name: Optional[str]
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    metadata: Dict[str, Any]
    language: str
    parent_context: Optional[str] = None
    imports: List[str] = None
    
    def __post_init__(self):
        if self.imports is None:
            self.imports = []

class LanguageHandler(ABC):
    """Base class for language-specific AST parsing."""
    
    def __init__(self, language_name: str):
        """Initialize language handler."""
        self.language_name = language_name
        self.parser: Optional[Parser] = None
        self.language: Optional[Language] = None
        
        # Language-specific configuration
        self.function_nodes = set()
        self.class_nodes = set()
        self.import_nodes = set()
        self.comment_nodes = set()
        self.docstring_nodes = set()
        
        self._setup_node_types()
    
    @abstractmethod
    def _setup_node_types(self) -> None:
        """Set up language-specific node types."""
        pass
    
    def initialize_parser(self) -> bool:
        """Initialize Tree-sitter parser for this language."""
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available, falling back to line-based chunking")
            return False
        
        try:
            # This would need actual Tree-sitter language binaries
            # For now, we'll simulate the interface
            logger.info(f"Parser for {self.language_name} would be initialized here")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize {self.language_name} parser: {e}")
            return False
    
    def parse_code(self, source_code: str, file_path: str) -> List[ParsedChunk]:
        """Parse source code into semantic chunks."""
        if not self.parser:
            # Fallback to simple line-based chunking
            return self._fallback_chunking(source_code, file_path)
        
        try:
            tree = self.parser.parse(bytes(source_code, "utf8"))
            return self._extract_chunks(tree.root_node, source_code, file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return self._fallback_chunking(source_code, file_path)
    
    def _fallback_chunking(self, source_code: str, file_path: str) -> List[ParsedChunk]:
        """Fallback to line-based chunking when AST parsing fails."""
        lines = source_code.split('\n')
        chunks = []
        
        # Simple heuristic-based chunking
        current_chunk_lines = []
        current_start_line = 1
        
        for i, line in enumerate(lines, 1):
            current_chunk_lines.append(line)
            
            # End chunk on empty lines or when chunk gets too large
            if (not line.strip() and len(current_chunk_lines) > 5) or len(current_chunk_lines) >= 50:
                if current_chunk_lines:
                    content = '\n'.join(current_chunk_lines)
                    if content.strip():  # Only add non-empty chunks
                        chunk = ParsedChunk(
                            content=content,
                            chunk_type=ChunkType.GENERIC,
                            name=None,
                            start_line=current_start_line,
                            end_line=i,
                            start_byte=0,
                            end_byte=len(content.encode('utf-8')),
                            metadata={"fallback": True},
                            language=self.language_name,
                        )
                        chunks.append(chunk)
                
                current_chunk_lines = []
                current_start_line = i + 1
        
        # Add final chunk if any
        if current_chunk_lines:
            content = '\n'.join(current_chunk_lines)
            if content.strip():
                chunk = ParsedChunk(
                    content=content,
                    chunk_type=ChunkType.GENERIC,
                    name=None,
                    start_line=current_start_line,
                    end_line=len(lines),
                    start_byte=0,
                    end_byte=len(content.encode('utf-8')),
                    metadata={"fallback": True},
                    language=self.language_name,
                )
                chunks.append(chunk)
        
        return chunks
    
    @abstractmethod
    def _extract_chunks(self, root_node: Any, source_code: str, file_path: str) -> List[ParsedChunk]:
        """Extract semantic chunks from AST."""
        pass
    
    def _get_node_text(self, node: Any, source_code: str) -> str:
        """Get text content of a node."""
        if hasattr(node, 'start_byte') and hasattr(node, 'end_byte'):
            return source_code[node.start_byte:node.end_byte]
        return ""
    
    def _get_line_numbers(self, node: Any) -> Tuple[int, int]:
        """Get start and end line numbers for a node."""
        if hasattr(node, 'start_point') and hasattr(node, 'end_point'):
            return node.start_point[0] + 1, node.end_point[0] + 1
        return 1, 1

class PythonHandler(LanguageHandler):
    """Handler for Python code."""
    
    def __init__(self):
        super().__init__("python")
    
    def _setup_node_types(self) -> None:
        """Set up Python-specific node types."""
        self.function_nodes = {"function_definition", "async_function_definition"}
        self.class_nodes = {"class_definition"}
        self.import_nodes = {"import_statement", "import_from_statement"}
        self.comment_nodes = {"comment"}
        self.docstring_nodes = {"expression_statement"}  # May contain docstrings
    
    def _extract_chunks(self, root_node: Any, source_code: str, file_path: str) -> List[ParsedChunk]:
        """Extract chunks from Python AST."""
        chunks = []
        
        # Extract imports first
        imports = self._extract_imports(root_node, source_code)
        
        # Extract top-level constructs
        for child in root_node.children:
            if child.type in self.function_nodes:
                chunk = self._extract_function(child, source_code, imports)
                if chunk:
                    chunks.append(chunk)
            
            elif child.type in self.class_nodes:
                class_chunks = self._extract_class(child, source_code, imports)
                chunks.extend(class_chunks)
        
        # Add import chunk if imports exist
        if imports:
            import_content = '\n'.join(imports)
            import_chunk = ParsedChunk(
                content=import_content,
                chunk_type=ChunkType.IMPORT,
                name="imports",
                start_line=1,
                end_line=len(imports),
                start_byte=0,
                end_byte=len(import_content.encode('utf-8')),
                metadata={"import_count": len(imports)},
                language=self.language_name,
                imports=imports,
            )
            chunks.insert(0, import_chunk)  # Imports go first
        
        return chunks
    
    def _extract_imports(self, root_node: Any, source_code: str) -> List[str]:
        """Extract import statements."""
        imports = []
        for child in root_node.children:
            if child.type in self.import_nodes:
                import_text = self._get_node_text(child, source_code)
                imports.append(import_text.strip())
        return imports
    
    def _extract_function(self, node: Any, source_code: str, imports: List[str]) -> Optional[ParsedChunk]:
        """Extract a function definition."""
        content = self._get_node_text(node, source_code)
        start_line, end_line = self._get_line_numbers(node)
        
        # Extract function name
        name = "unknown_function"
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
                break
        
        # Check for docstring
        docstring = self._extract_docstring(node, source_code)
        
        metadata = {
            "is_async": node.type == "async_function_definition",
            "has_docstring": docstring is not None,
            "line_count": end_line - start_line + 1,
        }
        
        if docstring:
            metadata["docstring"] = docstring
        
        return ParsedChunk(
            content=content,
            chunk_type=ChunkType.FUNCTION,
            name=name,
            start_line=start_line,
            end_line=end_line,
            start_byte=node.start_byte if hasattr(node, 'start_byte') else 0,
            end_byte=node.end_byte if hasattr(node, 'end_byte') else len(content.encode('utf-8')),
            metadata=metadata,
            language=self.language_name,
            imports=imports,
        )
    
    def _extract_class(self, node: Any, source_code: str, imports: List[str]) -> List[ParsedChunk]:
        """Extract a class definition and its methods."""
        chunks = []
        
        # Extract class name
        class_name = "unknown_class"
        for child in node.children:
            if child.type == "identifier":
                class_name = self._get_node_text(child, source_code)
                break
        
        # Extract class docstring
        class_docstring = self._extract_docstring(node, source_code)
        
        # Extract class-level chunk
        class_content = self._get_node_text(node, source_code)
        start_line, end_line = self._get_line_numbers(node)
        
        class_metadata = {
            "has_docstring": class_docstring is not None,
            "method_count": len([c for c in node.children if c.type in self.function_nodes]),
        }
        
        if class_docstring:
            class_metadata["docstring"] = class_docstring
        
        class_chunk = ParsedChunk(
            content=class_content,
            chunk_type=ChunkType.CLASS,
            name=class_name,
            start_line=start_line,
            end_line=end_line,
            start_byte=node.start_byte if hasattr(node, 'start_byte') else 0,
            end_byte=node.end_byte if hasattr(node, 'end_byte') else len(class_content.encode('utf-8')),
            metadata=class_metadata,
            language=self.language_name,
            parent_context=None,
            imports=imports,
        )
        chunks.append(class_chunk)
        
        # Extract methods separately for better granularity
        for child in node.children:
            if child.type in self.function_nodes:
                method_chunk = self._extract_function(child, source_code, imports)
                if method_chunk:
                    method_chunk.chunk_type = ChunkType.METHOD
                    method_chunk.parent_context = class_name
                    chunks.append(method_chunk)
        
        return chunks
    
    def _extract_docstring(self, node: Any, source_code: str) -> Optional[str]:
        """Extract docstring from a function or class."""
        # Look for string literal as first statement in body
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for expr in stmt.children:
                            if expr.type == "string":
                                return self._get_node_text(expr, source_code).strip('"\'')
        return None

class JavaScriptHandler(LanguageHandler):
    """Handler for JavaScript/TypeScript code."""
    
    def __init__(self, language_name: str = "javascript"):
        super().__init__(language_name)
    
    def _setup_node_types(self) -> None:
        """Set up JavaScript-specific node types."""
        self.function_nodes = {
            "function_declaration", "arrow_function", "function_expression",
            "method_definition", "generator_function_declaration"
        }
        self.class_nodes = {"class_declaration"}
        self.import_nodes = {"import_statement", "import_declaration"}
        self.comment_nodes = {"comment", "line_comment", "block_comment"}
    
    def _extract_chunks(self, root_node: Any, source_code: str, file_path: str) -> List[ParsedChunk]:
        """Extract chunks from JavaScript AST."""
        chunks = []
        
        # Extract imports
        imports = self._extract_imports(root_node, source_code)
        
        # Extract top-level constructs
        for child in root_node.children:
            if child.type in self.function_nodes:
                chunk = self._extract_function(child, source_code, imports)
                if chunk:
                    chunks.append(chunk)
            
            elif child.type in self.class_nodes:
                class_chunks = self._extract_class(child, source_code, imports)
                chunks.extend(class_chunks)
        
        return chunks
    
    def _extract_imports(self, root_node: Any, source_code: str) -> List[str]:
        """Extract import statements."""
        imports = []
        for child in root_node.children:
            if child.type in self.import_nodes:
                import_text = self._get_node_text(child, source_code)
                imports.append(import_text.strip())
        return imports
    
    def _extract_function(self, node: Any, source_code: str, imports: List[str]) -> Optional[ParsedChunk]:
        """Extract a function definition."""
        content = self._get_node_text(node, source_code)
        start_line, end_line = self._get_line_numbers(node)
        
        # Extract function name
        name = "anonymous_function"
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source_code)
                break
        
        metadata = {
            "is_arrow": node.type == "arrow_function",
            "is_async": "async" in content,
            "line_count": end_line - start_line + 1,
        }
        
        return ParsedChunk(
            content=content,
            chunk_type=ChunkType.FUNCTION,
            name=name,
            start_line=start_line,
            end_line=end_line,
            start_byte=node.start_byte if hasattr(node, 'start_byte') else 0,
            end_byte=node.end_byte if hasattr(node, 'end_byte') else len(content.encode('utf-8')),
            metadata=metadata,
            language=self.language_name,
            imports=imports,
        )
    
    def _extract_class(self, node: Any, source_code: str, imports: List[str]) -> List[ParsedChunk]:
        """Extract a class definition."""
        chunks = []
        
        # Extract class name
        class_name = "unknown_class"
        for child in node.children:
            if child.type == "identifier":
                class_name = self._get_node_text(child, source_code)
                break
        
        # Extract class chunk
        class_content = self._get_node_text(node, source_code)
        start_line, end_line = self._get_line_numbers(node)
        
        class_chunk = ParsedChunk(
            content=class_content,
            chunk_type=ChunkType.CLASS,
            name=class_name,
            start_line=start_line,
            end_line=end_line,
            start_byte=node.start_byte if hasattr(node, 'start_byte') else 0,
            end_byte=node.end_byte if hasattr(node, 'end_byte') else len(class_content.encode('utf-8')),
            metadata={"line_count": end_line - start_line + 1},
            language=self.language_name,
            imports=imports,
        )
        chunks.append(class_chunk)
        
        return chunks

# Language handler registry
LANGUAGE_HANDLERS = {
    ".py": PythonHandler,
    ".js": JavaScriptHandler,
    ".ts": lambda: JavaScriptHandler("typescript"),
    ".jsx": lambda: JavaScriptHandler("jsx"),
    ".tsx": lambda: JavaScriptHandler("tsx"),
}

def get_language_handler(file_path: str) -> Optional[LanguageHandler]:
    """Get appropriate language handler for a file."""
    path = Path(file_path)
    extension = path.suffix.lower()
    
    handler_class = LANGUAGE_HANDLERS.get(extension)
    if handler_class:
        try:
            handler = handler_class()
            if handler.initialize_parser():
                return handler
            else:
                # Return handler anyway for fallback chunking
                return handler
        except Exception as e:
            logger.warning(f"Failed to create handler for {extension}: {e}")
    
    # Return generic handler for unknown extensions
    return GenericHandler(extension)

class GenericHandler(LanguageHandler):
    """Generic handler for unsupported languages."""
    
    def __init__(self, extension: str):
        super().__init__(f"generic{extension}")
    
    def _setup_node_types(self) -> None:
        """No specific node types for generic handler."""
        pass
    
    def _extract_chunks(self, root_node: Any, source_code: str, file_path: str) -> List[ParsedChunk]:
        """Generic chunking always falls back to line-based."""
        return self._fallback_chunking(source_code, file_path)
