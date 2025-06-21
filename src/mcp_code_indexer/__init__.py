"""
MCP Code Indexer - Intelligent codebase navigation for AI agents.

A production-ready Model Context Protocol (MCP) server that provides
intelligent codebase navigation through searchable file descriptions,
token-aware overviews, and advanced merge capabilities.
"""

from .server.mcp_server import MCPCodeIndexServer


def _get_version() -> str:
    """Get version from package metadata or pyproject.toml."""
    # First try to get version from installed package metadata
    try:
        try:
            from importlib.metadata import version
        except ImportError:
            # Python < 3.8 fallback
            from importlib_metadata import version

        # Try different package name variations
        for pkg_name in ["mcp-code-indexer", "mcp_code_indexer"]:
            try:
                return version(pkg_name)
            except Exception:
                continue
    except Exception:
        pass

    # Fallback to reading from pyproject.toml (for development)
    try:
        from pathlib import Path
        import sys

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            try:
                import tomli as tomllib
            except ImportError:
                return "dev"

        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except Exception:
        return "dev"


__version__ = _get_version()
__author__ = "MCP Code Indexer Contributors"
__email__ = ""
__license__ = "MIT"

__all__ = ["MCPCodeIndexServer", "__version__"]
