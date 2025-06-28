#!/usr/bin/env python3
"""Simple test for local database functionality."""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_code_indexer.database.path_resolver import DatabasePathResolver


async def test_path_resolver():
    """Test database path resolution."""
    global_db = Path.home() / '.mcp-code-index' / 'tracker.db'
    resolver = DatabasePathResolver(global_db)
    
    print("Testing DatabasePathResolver...")
    
    # Test with no folder path
    result1 = resolver.resolve_database_path()
    print(f'No folder: {result1}')
    
    # Test with non-existent local db
    result2 = resolver.resolve_database_path('/tmp')
    print(f'No local db: {result2}')
    
    # Test with existing empty local db
    result3 = resolver.resolve_database_path('/tmp/test_project')
    print(f'With local db: {result3}')
    
    # Test empty database detection
    is_empty = resolver.is_empty_database_file(Path('/tmp/test_project/.code-index.db'))
    print(f'Is empty: {is_empty}')
    
    should_init = resolver.should_initialize_local_database('/tmp/test_project')
    print(f'Should initialize: {should_init}')


if __name__ == "__main__":
    asyncio.run(test_path_resolver())
