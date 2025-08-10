"""
Secret redaction engine for vector mode.

Detects and redacts sensitive information from code before sending
to external APIs for embedding generation.
"""

import hashlib
import logging
import re
from typing import List, Dict, Any, Optional, Set, NamedTuple
from dataclasses import dataclass
from pathlib import Path

from .patterns import SecurityPatterns, PatternMatch

logger = logging.getLogger(__name__)

class RedactionResult(NamedTuple):
    """Result of secret redaction process."""
    original_hash: str
    redacted_content: str
    redaction_count: int
    patterns_matched: List[str]
    confidence_scores: List[float]
    was_redacted: bool

@dataclass
class RedactionStats:
    """Statistics about redaction operations."""
    total_files_processed: int = 0
    total_redactions: int = 0
    redactions_by_type: Dict[str, int] = None
    redactions_by_pattern: Dict[str, int] = None
    high_confidence_redactions: int = 0
    
    def __post_init__(self):
        if self.redactions_by_type is None:
            self.redactions_by_type = {}
        if self.redactions_by_pattern is None:
            self.redactions_by_pattern = {}

class SecretRedactor:
    """
    Main secret redaction engine.
    
    Scans code content for secrets and replaces them with safe placeholders
    while preserving code structure and semantics.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        preserve_structure: bool = True,
        redaction_marker: str = "[REDACTED]",
        custom_patterns_file: Optional[Path] = None,
    ):
        """
        Initialize the secret redactor.
        
        Args:
            min_confidence: Minimum confidence threshold for redaction
            preserve_structure: Whether to preserve code structure
            redaction_marker: Marker to use for redacted content
            custom_patterns_file: Path to custom patterns file
        """
        self.min_confidence = min_confidence
        self.preserve_structure = preserve_structure
        self.redaction_marker = redaction_marker
        
        # Load security patterns
        self.patterns = SecurityPatterns()
        
        # Load custom patterns if provided
        if custom_patterns_file and custom_patterns_file.exists():
            self._load_custom_patterns(custom_patterns_file)
        
        # Statistics tracking
        self.stats = RedactionStats()
        
        # Common safe file extensions (no redaction needed)
        self.safe_extensions = {
            '.md', '.txt', '.rst', '.json', '.yaml', '.yml',
            '.xml', '.html', '.css', '.svg', '.license'
        }
        
        # Cache for performance
        self._pattern_cache: Dict[str, List[PatternMatch]] = {}
    
    def _load_custom_patterns(self, patterns_file: Path) -> None:
        """Load custom security patterns from file."""
        try:
            # TODO: Implement custom pattern loading
            logger.info(f"Custom patterns file specified but not yet implemented: {patterns_file}")
        except Exception as e:
            logger.warning(f"Failed to load custom patterns from {patterns_file}: {e}")
    
    def _should_process_file(self, file_path: str) -> bool:
        """Determine if a file should be processed for redaction."""
        path = Path(file_path)
        
        # Skip safe file types
        if path.suffix.lower() in self.safe_extensions:
            return False
        
        # Skip test files (often contain mock data)
        if any(test_marker in path.name.lower() for test_marker in ['test', 'spec', 'mock']):
            return False
        
        # Skip documentation directories
        if any(doc_dir in path.parts for doc_dir in ['docs', 'documentation', 'examples']):
            return False
        
        return True
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _create_redaction_marker(
        self,
        pattern_type: str,
        pattern_name: str,
        original_length: int
    ) -> str:
        """Create a redaction marker that preserves structure."""
        if not self.preserve_structure:
            return self.redaction_marker
        
        # Create a marker that maintains similar length
        base_marker = f"[REDACTED:{pattern_type.upper()}]"
        
        if original_length <= len(base_marker):
            return base_marker[:original_length]
        
        # Pad with safe characters to maintain length
        padding = 'X' * (original_length - len(base_marker))
        return base_marker + padding
    
    def _redact_matches(self, content: str, matches: List[PatternMatch]) -> str:
        """Apply redactions to content based on matches."""
        if not matches:
            return content
        
        # Sort matches by position (reverse order for safe replacement)
        sorted_matches = sorted(matches, key=lambda m: m.start_pos, reverse=True)
        
        redacted_content = content
        
        for match in sorted_matches:
            # Create appropriate redaction marker
            marker = self._create_redaction_marker(
                match.pattern_type,
                match.pattern_name,
                len(match.matched_text)
            )
            
            # Replace the matched text
            redacted_content = (
                redacted_content[:match.start_pos] +
                marker +
                redacted_content[match.end_pos:]
            )
            
            # Update statistics
            self.stats.total_redactions += 1
            self.stats.redactions_by_type[match.pattern_type] = (
                self.stats.redactions_by_type.get(match.pattern_type, 0) + 1
            )
            self.stats.redactions_by_pattern[match.pattern_name] = (
                self.stats.redactions_by_pattern.get(match.pattern_name, 0) + 1
            )
            
            if match.confidence >= 0.8:
                self.stats.high_confidence_redactions += 1
            
            logger.debug(
                f"Redacted {match.pattern_name} (confidence: {match.confidence:.2f}): "
                f"{match.matched_text[:20]}..."
            )
        
        return redacted_content
    
    def _filter_false_positives(
        self,
        matches: List[PatternMatch],
        content: str,
        file_path: Optional[str] = None
    ) -> List[PatternMatch]:
        """Filter out likely false positives based on context."""
        filtered_matches = []
        
        for match in matches:
            # Skip very short matches for low-confidence patterns
            if match.confidence < 0.7 and len(match.matched_text) < 16:
                continue
            
            # Skip matches that look like placeholders
            if self._looks_like_placeholder(match.matched_text):
                continue
            
            # Skip matches in comments (for code files)
            if file_path and self._is_in_comment(content, match.start_pos, file_path):
                continue
            
            # Skip matches that are likely examples or documentation
            if self._is_example_content(content, match.start_pos):
                continue
            
            filtered_matches.append(match)
        
        return filtered_matches
    
    def _looks_like_placeholder(self, text: str) -> bool:
        """Check if text looks like a placeholder rather than real secret."""
        placeholder_indicators = [
            'example', 'sample', 'test', 'demo', 'placeholder', 'your',
            'xxxxx', '11111', '00000', 'aaaaa', 'abcdef',
            'replace', 'insert', 'enter', 'put'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in placeholder_indicators)
    
    def _is_in_comment(self, content: str, position: int, file_path: str) -> bool:
        """Check if position is within a comment."""
        # Get file extension for comment style detection
        ext = Path(file_path).suffix.lower()
        
        # Find line containing the position
        lines = content[:position].split('\n')
        current_line = content.split('\n')[len(lines) - 1] if lines else ""
        
        # Check for common comment patterns
        comment_patterns = {
            '.py': [r'#.*', r'""".*?"""', r"'''.*?'''"],
            '.js': [r'//.*', r'/\*.*?\*/'],
            '.java': [r'//.*', r'/\*.*?\*/'],
            '.cpp': [r'//.*', r'/\*.*?\*/'],
            '.c': [r'//.*', r'/\*.*?\*/'],
            '.go': [r'//.*', r'/\*.*?\*/'],
            '.rs': [r'//.*', r'/\*.*?\*/'],
            '.sh': [r'#.*'],
            '.sql': [r'--.*', r'/\*.*?\*/'],
        }
        
        patterns = comment_patterns.get(ext, [])
        for pattern in patterns:
            if re.search(pattern, current_line, re.DOTALL):
                return True
        
        return False
    
    def _is_example_content(self, content: str, position: int, context_size: int = 100) -> bool:
        """Check if content around position suggests it's example/documentation."""
        start = max(0, position - context_size)
        end = min(len(content), position + context_size)
        context = content[start:end].lower()
        
        example_indicators = [
            'example', 'sample', 'demo', 'tutorial', 'documentation',
            'readme', 'how to', 'getting started', 'quickstart'
        ]
        
        return any(indicator in context for indicator in example_indicators)
    
    def redact_content(
        self,
        content: str,
        file_path: Optional[str] = None,
        cache_key: Optional[str] = None
    ) -> RedactionResult:
        """
        Redact secrets from content.
        
        Args:
            content: Content to redact
            file_path: Path to file (for context)
            cache_key: Optional cache key for performance
            
        Returns:
            RedactionResult with original hash and redacted content
        """
        # Generate hash of original content
        original_hash = self._generate_content_hash(content)
        
        # Check if file should be processed
        if file_path and not self._should_process_file(file_path):
            logger.debug(f"Skipping redaction for safe file: {file_path}")
            return RedactionResult(
                original_hash=original_hash,
                redacted_content=content,
                redaction_count=0,
                patterns_matched=[],
                confidence_scores=[],
                was_redacted=False
            )
        
        # Check cache if available
        if cache_key and cache_key in self._pattern_cache:
            matches = self._pattern_cache[cache_key]
        else:
            # Find all pattern matches
            matches = self.patterns.find_matches(content, self.min_confidence)
            
            # Filter false positives
            matches = self._filter_false_positives(matches, content, file_path)
            
            # Cache results
            if cache_key:
                self._pattern_cache[cache_key] = matches
        
        # Apply redactions
        redacted_content = self._redact_matches(content, matches)
        
        # Update statistics
        self.stats.total_files_processed += 1
        
        # Log redaction summary
        if matches:
            logger.info(
                f"Redacted {len(matches)} secrets from {file_path or 'content'} "
                f"(confidence range: {min(m.confidence for m in matches):.2f}-"
                f"{max(m.confidence for m in matches):.2f})"
            )
        
        return RedactionResult(
            original_hash=original_hash,
            redacted_content=redacted_content,
            redaction_count=len(matches),
            patterns_matched=[m.pattern_name for m in matches],
            confidence_scores=[m.confidence for m in matches],
            was_redacted=len(matches) > 0
        )
    
    def redact_file(self, file_path: Path) -> RedactionResult:
        """Redact secrets from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return self.redact_content(content, str(file_path))
            
        except Exception as e:
            logger.error(f"Failed to redact file {file_path}: {e}")
            return RedactionResult(
                original_hash="",
                redacted_content="",
                redaction_count=0,
                patterns_matched=[],
                confidence_scores=[],
                was_redacted=False
            )
    
    def get_redaction_stats(self) -> RedactionStats:
        """Get redaction statistics."""
        return self.stats
    
    def clear_cache(self) -> None:
        """Clear the pattern match cache."""
        self._pattern_cache.clear()
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the confidence threshold for redaction."""
        if 0.0 <= threshold <= 1.0:
            self.min_confidence = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
