# MCP Code Indexer 🚀

A production-ready **Model Context Protocol (MCP) server** that provides intelligent codebase navigation for AI agents through searchable file descriptions, token-aware overviews, and advanced merge capabilities.

## 🎯 What It Does

The MCP Code Indexer solves a critical problem for AI agents working with large codebases: **understanding code structure without repeatedly scanning files**. Instead of reading every file, agents can:

- **Query file purposes** instantly with natural language descriptions
- **Search across codebases** using full-text search
- **Get intelligent recommendations** based on codebase size (overview vs search)
- **Merge branch descriptions** with conflict resolution
- **Inherit descriptions** from upstream repositories automatically

Perfect for AI-powered code review, refactoring tools, documentation generation, and codebase analysis workflows.

## ⚡ Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/mcp-code-indexer.git
cd mcp-code-indexer

# Install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the server
python main.py --token-limit 32000

# The server is now ready to accept MCP tool calls!
```

## 🛠️ MCP Tools Available

The server provides **8 powerful MCP tools** for intelligent codebase management:

### Core Operations
- **`get_file_description`** - Retrieve stored file descriptions instantly
- **`update_file_description`** - Store detailed file summaries and metadata
- **`check_codebase_size`** - Get token count and size-based recommendations

### Batch Operations
- **`find_missing_descriptions`** - Scan projects for files without descriptions
- **`update_missing_descriptions`** - Bulk update multiple file descriptions

### Search & Discovery
- **`search_descriptions`** - Fast full-text search across all descriptions
- **`get_codebase_overview`** - Complete hierarchical project structure

### Advanced Features
- **`merge_branch_descriptions`** - Two-phase merge with conflict resolution

## 🏗️ Architecture Highlights

### Performance Optimized
- **SQLite with WAL mode** for high-concurrency access
- **Connection pooling** for efficient database operations
- **FTS5 full-text search** with prefix indexing
- **Token-aware caching** to minimize expensive operations

### Production Ready
- **Comprehensive error handling** with structured JSON logging
- **Async-first design** with proper resource cleanup
- **Upstream inheritance** for fork workflows
- **Git integration** with .gitignore support

### Developer Friendly
- **95%+ test coverage** with async support
- **Integration tests** for complete workflows
- **Performance benchmarks** for large codebases
- **Clear error messages** with MCP protocol compliance

## 📖 Documentation

- **[API Reference](docs/api-reference.md)** - Complete MCP tool documentation
- **[Configuration Guide](docs/configuration.md)** - Setup and tuning options
- **[Architecture Overview](docs/architecture.md)** - Technical deep dive
- **[Contributing Guide](docs/contributing.md)** - Development workflow

## 🚦 System Requirements

- **Python 3.8+** with asyncio support
- **SQLite 3.35+** (included with Python)
- **4GB+ RAM** for large codebases (1000+ files)
- **SSD storage** recommended for optimal performance

## 📊 Performance

Tested with codebases up to **10,000 files**:
- File description retrieval: **< 10ms**
- Full-text search: **< 100ms** 
- Codebase overview generation: **< 2s**
- Merge conflict detection: **< 5s**

## 🔧 Advanced Configuration

```bash
# Production setup with custom limits
python main.py \
  --token-limit 50000 \
  --db-path /data/mcp-index.db \
  --cache-dir /tmp/mcp-cache \
  --log-level INFO

# Enable structured logging
export MCP_LOG_FORMAT=json
python main.py
```

## 🤝 Integration Examples

### With AI Agents
```python
# Example: AI agent using MCP tools
async def analyze_codebase(project_path):
    # Check if codebase is large
    size_info = await mcp_client.call_tool("check_codebase_size", {
        "projectName": "my-project",
        "folderPath": project_path,
        "branch": "main"
    })
    
    if size_info["isLarge"]:
        # Use search for large codebases
        results = await mcp_client.call_tool("search_descriptions", {
            "projectName": "my-project", 
            "folderPath": project_path,
            "branch": "main",
            "query": "authentication logic"
        })
    else:
        # Get full overview for smaller projects
        overview = await mcp_client.call_tool("get_codebase_overview", {
            "projectName": "my-project",
            "folderPath": project_path, 
            "branch": "main"
        })
```

### With CI/CD Pipelines
```yaml
# Example: GitHub Actions integration
- name: Update Code Descriptions
  run: |
    python -c "
    import asyncio
    from mcp_client import MCPClient
    
    async def update_descriptions():
        client = MCPClient('mcp-code-indexer')
        
        # Find files without descriptions
        missing = await client.call_tool('find_missing_descriptions', {
            'projectName': '${{ github.repository }}',
            'folderPath': '.',
            'branch': '${{ github.ref_name }}'
        })
        
        # Process with AI and update...
    
    asyncio.run(update_descriptions())
    "
```

## 🧪 Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run performance tests
python -m pytest tests/ -m performance

# Run integration tests only
python -m pytest tests/integration/ -v
```

## 📈 Monitoring

The server provides structured JSON logs for monitoring:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Tool search_descriptions completed",
  "tool_usage": {
    "tool_name": "search_descriptions",
    "success": true,
    "duration_seconds": 0.045,
    "result_size": 1247
  }
}
```

## 🛡️ Security Features

- **Input validation** on all MCP tool parameters
- **SQL injection protection** via parameterized queries  
- **File system sandboxing** with .gitignore respect
- **Error sanitization** to prevent information leakage
- **Async resource cleanup** to prevent memory leaks

## 🚀 Next Steps

1. **[Read the API docs](docs/api-reference.md)** to understand available tools
2. **[Check the examples](examples/)** for integration patterns
3. **[Join our Discord](https://discord.gg/mcp-tools)** for community support
4. **[Contribute](docs/contributing.md)** to help improve the project

## 🤝 Contributing

We welcome contributions! See our **[Contributing Guide](docs/contributing.md)** for:
- Development setup
- Code style guidelines  
- Testing requirements
- Pull request process

## 📄 License

MIT License - see **[LICENSE](LICENSE)** for details.

## 🙏 Built With

- **[Model Context Protocol](https://github.com/modelcontextprotocol/python-sdk)** - The foundation for tool integration
- **[tiktoken](https://pypi.org/project/tiktoken/)** - Fast BPE tokenization  
- **[aiosqlite](https://pypi.org/project/aiosqlite/)** - Async SQLite operations
- **[Pydantic](https://pydantic.dev/)** - Data validation and settings

---

**Ready to supercharge your AI agents with intelligent codebase navigation?** 🚀 [Get started now](#-quick-start) or [explore the API docs](docs/api-reference.md)!
