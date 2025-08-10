"""
Security module for vector mode.

Provides secret redaction capabilities to prevent sensitive information
from being sent to external APIs for embedding generation.
"""

from .redactor import SecretRedactor, RedactionResult
from .patterns import SecurityPatterns

__all__ = ["SecretRedactor", "RedactionResult", "SecurityPatterns"]
