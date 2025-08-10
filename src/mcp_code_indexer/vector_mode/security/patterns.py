"""
Security patterns for detecting secrets in code.

Comprehensive collection of regex patterns to identify API keys, tokens,
passwords, connection strings, and other sensitive information.
"""

import re
from typing import List, Dict, Pattern, NamedTuple
from dataclasses import dataclass

class PatternMatch(NamedTuple):
    """Represents a detected secret pattern match."""
    pattern_name: str
    pattern_type: str
    start_pos: int
    end_pos: int
    matched_text: str
    confidence: float

@dataclass
class SecurityPattern:
    """Represents a security pattern with metadata."""
    name: str
    pattern: Pattern[str]
    pattern_type: str
    description: str
    confidence: float = 1.0
    context_required: bool = False

class SecurityPatterns:
    """Collection of security patterns for secret detection."""
    
    def __init__(self):
        self.patterns = self._build_patterns()
    
    def _build_patterns(self) -> List[SecurityPattern]:
        """Build comprehensive list of security patterns."""
        patterns = []
        
        # API Keys and Tokens
        patterns.extend([
            SecurityPattern(
                name="aws_access_key",
                pattern=re.compile(r'AKIA[0-9A-Z]{16}', re.IGNORECASE),
                pattern_type="api_key",
                description="AWS Access Key ID",
                confidence=0.95
            ),
            SecurityPattern(
                name="aws_secret_key",
                pattern=re.compile(r'[A-Za-z0-9/+=]{40}', re.IGNORECASE),
                pattern_type="api_key",
                description="AWS Secret Access Key",
                confidence=0.7,
                context_required=True
            ),
            SecurityPattern(
                name="github_token",
                pattern=re.compile(r'gh[pousr]_[A-Za-z0-9_]{36,}', re.IGNORECASE),
                pattern_type="api_key",
                description="GitHub Token",
                confidence=0.95
            ),
            SecurityPattern(
                name="google_api_key",
                pattern=re.compile(r'AIza[0-9A-Za-z\-_]{35}', re.IGNORECASE),
                pattern_type="api_key",
                description="Google API Key",
                confidence=0.95
            ),
            SecurityPattern(
                name="slack_token",
                pattern=re.compile(r'xox[baprs]-([0-9a-zA-Z]{10,48})', re.IGNORECASE),
                pattern_type="api_key",
                description="Slack Token",
                confidence=0.95
            ),
            SecurityPattern(
                name="stripe_key",
                pattern=re.compile(r'[rs]k_(test|live)_[0-9a-zA-Z]{24}', re.IGNORECASE),
                pattern_type="api_key",
                description="Stripe API Key",
                confidence=0.95
            ),
            SecurityPattern(
                name="openai_api_key",
                pattern=re.compile(r'sk-[a-zA-Z0-9]{48}', re.IGNORECASE),
                pattern_type="api_key",
                description="OpenAI API Key",
                confidence=0.95
            ),
            SecurityPattern(
                name="anthropic_api_key",
                pattern=re.compile(r'sk-ant-api03-[a-zA-Z0-9\-_]{95}', re.IGNORECASE),
                pattern_type="api_key",
                description="Anthropic API Key",
                confidence=0.95
            ),
            SecurityPattern(
                name="voyage_api_key",
                pattern=re.compile(r'pa-[a-zA-Z0-9]{32}', re.IGNORECASE),
                pattern_type="api_key",
                description="Voyage AI API Key",
                confidence=0.95
            ),
        ])
        
        # JWT Tokens
        patterns.append(
            SecurityPattern(
                name="jwt_token",
                pattern=re.compile(r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*', re.IGNORECASE),
                pattern_type="token",
                description="JWT Token",
                confidence=0.9
            )
        )
        
        # Database Connection Strings
        patterns.extend([
            SecurityPattern(
                name="postgres_url",
                pattern=re.compile(r'postgres(?:ql)?://[^\s]+', re.IGNORECASE),
                pattern_type="connection_string",
                description="PostgreSQL Connection String",
                confidence=0.85
            ),
            SecurityPattern(
                name="mysql_url",
                pattern=re.compile(r'mysql://[^\s]+', re.IGNORECASE),
                pattern_type="connection_string",
                description="MySQL Connection String",
                confidence=0.85
            ),
            SecurityPattern(
                name="mongodb_url",
                pattern=re.compile(r'mongodb(?:\+srv)?://[^\s]+', re.IGNORECASE),
                pattern_type="connection_string",
                description="MongoDB Connection String",
                confidence=0.85
            ),
            SecurityPattern(
                name="redis_url",
                pattern=re.compile(r'redis://[^\s]+', re.IGNORECASE),
                pattern_type="connection_string",
                description="Redis Connection String",
                confidence=0.85
            ),
        ])
        
        # Private Keys
        patterns.extend([
            SecurityPattern(
                name="rsa_private_key",
                pattern=re.compile(r'-----BEGIN RSA PRIVATE KEY-----[^-]+-----END RSA PRIVATE KEY-----', re.MULTILINE | re.DOTALL),
                pattern_type="private_key",
                description="RSA Private Key",
                confidence=1.0
            ),
            SecurityPattern(
                name="ssh_private_key",
                pattern=re.compile(r'-----BEGIN OPENSSH PRIVATE KEY-----[^-]+-----END OPENSSH PRIVATE KEY-----', re.MULTILINE | re.DOTALL),
                pattern_type="private_key",
                description="SSH Private Key",
                confidence=1.0
            ),
            SecurityPattern(
                name="ec_private_key",
                pattern=re.compile(r'-----BEGIN EC PRIVATE KEY-----[^-]+-----END EC PRIVATE KEY-----', re.MULTILINE | re.DOTALL),
                pattern_type="private_key",
                description="EC Private Key",
                confidence=1.0
            ),
        ])
        
        # Environment Variable Patterns
        patterns.extend([
            SecurityPattern(
                name="env_password",
                pattern=re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?[^\s"\']+["\']?', re.IGNORECASE),
                pattern_type="password",
                description="Environment Variable Password",
                confidence=0.7,
                context_required=True
            ),
            SecurityPattern(
                name="env_secret",
                pattern=re.compile(r'(?i)(secret|token|key)\s*[=:]\s*["\']?[^\s"\']+["\']?', re.IGNORECASE),
                pattern_type="secret",
                description="Environment Variable Secret",
                confidence=0.6,
                context_required=True
            ),
        ])
        
        # Generic Patterns (lower confidence)
        patterns.extend([
            SecurityPattern(
                name="base64_encoded",
                pattern=re.compile(r'[A-Za-z0-9+/]{32,}={0,2}', re.IGNORECASE),
                pattern_type="encoded_data",
                description="Base64 Encoded Data",
                confidence=0.3,
                context_required=True
            ),
            SecurityPattern(
                name="hex_encoded",
                pattern=re.compile(r'[a-fA-F0-9]{32,}', re.IGNORECASE),
                pattern_type="encoded_data",
                description="Hex Encoded Data",
                confidence=0.3,
                context_required=True
            ),
            SecurityPattern(
                name="uuid",
                pattern=re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE),
                pattern_type="identifier",
                description="UUID",
                confidence=0.2,
                context_required=True
            ),
        ])
        
        # URLs with embedded credentials
        patterns.append(
            SecurityPattern(
                name="url_with_credentials",
                pattern=re.compile(r'https?://[^:/\s]+:[^@/\s]+@[^\s]+', re.IGNORECASE),
                pattern_type="credential_url",
                description="URL with embedded credentials",
                confidence=0.9
            )
        )
        
        return patterns
    
    def get_patterns_by_type(self, pattern_type: str) -> List[SecurityPattern]:
        """Get all patterns of a specific type."""
        return [p for p in self.patterns if p.pattern_type == pattern_type]
    
    def get_high_confidence_patterns(self, min_confidence: float = 0.8) -> List[SecurityPattern]:
        """Get patterns with confidence above threshold."""
        return [p for p in self.patterns if p.confidence >= min_confidence]
    
    def get_context_sensitive_patterns(self) -> List[SecurityPattern]:
        """Get patterns that require context for accurate detection."""
        return [p for p in self.patterns if p.context_required]
    
    def find_matches(self, text: str, min_confidence: float = 0.5) -> List[PatternMatch]:
        """Find all pattern matches in text above confidence threshold."""
        matches = []
        
        for pattern in self.patterns:
            if pattern.confidence < min_confidence:
                continue
            
            for match in pattern.pattern.finditer(text):
                # For context-sensitive patterns, check surrounding context
                if pattern.context_required:
                    if not self._has_suspicious_context(text, match.start(), match.end()):
                        continue
                
                matches.append(PatternMatch(
                    pattern_name=pattern.name,
                    pattern_type=pattern.pattern_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    matched_text=match.group(),
                    confidence=pattern.confidence
                ))
        
        # Sort by position for consistent output
        return sorted(matches, key=lambda m: m.start_pos)
    
    def _has_suspicious_context(self, text: str, start: int, end: int, context_size: int = 50) -> bool:
        """Check if match has suspicious context indicating it's likely a secret."""
        # Get surrounding context
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        context = text[context_start:context_end].lower()
        
        # Keywords that suggest secret/credential usage
        suspicious_keywords = [
            'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'api',
            'auth', 'credential', 'login', 'access', 'private', 'confidential',
            'env', 'config', 'setting', 'var', 'export', 'process.env'
        ]
        
        return any(keyword in context for keyword in suspicious_keywords)
    
    def get_pattern_summary(self) -> Dict[str, int]:
        """Get summary of patterns by type."""
        summary = {}
        for pattern in self.patterns:
            summary[pattern.pattern_type] = summary.get(pattern.pattern_type, 0) + 1
        return summary
