"""
MCP Code Indexer - Intelligent codebase navigation for AI agents.

A production-ready Model Context Protocol (MCP) server that provides
intelligent codebase navigation through searchable file descriptions,
token-aware overviews, and advanced merge capabilities.
"""

def _get_version() -> str:
    """Read version from pyproject.toml."""
    try:
        from pathlib import Path
        import sys
        
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            try:
                import tomli as tomllib
            except ImportError:
                # Fallback if tomli not available
                return "1.6.3"
        
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except Exception:
        # Return dev version if reading fails - indicates something is wrong
        return "dev"

__version__ = _get_version()
__author__ = "MCP Code Indexer Contributors"
__email__ = ""
__license__ = "MIT"

from .server.mcp_server import MCPCodeIndexServer

__all__ = ["MCPCodeIndexServer", "__version__"]
