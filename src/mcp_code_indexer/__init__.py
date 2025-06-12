"""
MCP Code Indexer - Intelligent codebase navigation for AI agents.

A production-ready Model Context Protocol (MCP) server that provides
intelligent codebase navigation through searchable file descriptions,
token-aware overviews, and advanced merge capabilities.
"""

__version__ = "1.6.0"
__author__ = "MCP Code Indexer Contributors"
__email__ = ""
__license__ = "MIT"

from .server.mcp_server import MCPCodeIndexServer

__all__ = ["MCPCodeIndexServer", "__version__"]
