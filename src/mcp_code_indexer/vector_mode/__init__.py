"""
Vector Mode for MCP Code Indexer.

This package provides semantic search capabilities using embeddings and vector databases.
Includes automated file monitoring, AST-based code chunking, and secure embedding generation.
"""

from typing import Optional
from pathlib import Path
import os

__version__ = "1.0.0"

def is_vector_mode_available() -> bool:
    """Check if vector mode dependencies are available."""
    try:
        import voyage
        import turbopuffer
        import tree_sitter
        import watchdog
        return True
    except ImportError:
        return False

def get_vector_config_path() -> Path:
    """Get path to vector mode configuration."""
    config_dir = Path.home() / ".mcp-code-index" / "vector"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.yaml"

def check_api_keys() -> dict[str, bool]:
    """Check availability of required API keys."""
    return {
        "voyage": os.getenv("VOYAGE_API_KEY") is not None,
        "turbopuffer": os.getenv("TURBOPUFFER_API_KEY") is not None,
    }
