# MCP Code Indexer 🚀

[![PyPI version](https://badge.fury.io/py/mcp-code-indexer.svg?26)](https://badge.fury.io/py/mcp-code-indexer)
[![Python](https://img.shields.io/pypi/pyversions/mcp-code-indexer.svg?26)](https://pypi.org/project/mcp-code-indexer/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A production-ready **Model Context Protocol (MCP) server** that revolutionizes how AI agents navigate and understand codebases. Built for high-concurrency environments with advanced database resilience, the server provides instant access to intelligent descriptions, semantic search, and context-aware recommendations while maintaining 800+ writes/sec throughput.

## 🎯 What It Does

The MCP Code Indexer solves a critical problem for AI agents working with large codebases: **understanding code structure without repeatedly scanning files**. Instead of reading every file, agents can:

- **Query file purposes** instantly with natural language descriptions
- **Search across codebases** using full-text search
- **Get intelligent recommendations** based on codebase size (overview vs search)
- **Generate condensed overviews** for project understanding

Perfect for AI-powered code review, refactoring tools, documentation generation, and codebase analysis workflows.

## ⚡ Quick Start

### 👨‍💻 For Developers

Get started integrating MCP Code Indexer into your AI agent workflow:

```bash
# Install the package
pip install mcp-code-indexer

# Start the MCP server
mcp-code-indexer

# Connect your MCP client and start using tools
# See API Reference for complete tool documentation
```

### 🔧 For System Administrators

Deploy and configure the server for your team:

```bash
# Production deployment with custom settings
mcp-code-indexer \
  --token-limit 64000 \
  --db-path /data/mcp-index.db \
  --cache-dir /var/cache/mcp \
  --log-level INFO

# Check installation
mcp-code-indexer --version
```

### 🎯 For Everyone

**New to MCP Code Indexer?** Start here:

1. **Install**: `pip install mcp-code-indexer`
2. **Run**: `mcp-code-indexer --token-limit 32000`
3. **Connect**: Use your favorite MCP client
4. **Explore**: Try the `check_codebase_size` tool first

**Development Setup**:

```bash
# Clone and setup for contributing
git clone https://github.com/fluffypony/mcp-code-indexer.git
cd mcp-code-indexer

# Install in development mode (required)
pip install -e .

# Run the server
mcp-code-indexer --token-limit 32000
```

## 🔗 Git Hook Integration

🚀 **NEW Feature**: Automated code indexing with AI-powered analysis! Keep your file descriptions synchronized automatically as your codebase evolves.

### 👤 For Users: Quick Setup

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-v1-your-api-key-here"

# Test git hook functionality
mcp-code-indexer --githook

# Install post-commit hook
cp examples/git-hooks/post-commit .git/hooks/
chmod +x .git/hooks/post-commit
```

### 👨‍💻 For Developers: How It Works

The git hook integration provides intelligent automation:

- **📊 Git Analysis**: Automatically analyzes git diffs after commits/merges
- **🤖 AI Processing**: Uses OpenRouter API with Anthropic's Claude Sonnet 4
- **⚡ Smart Updates**: Only processes files that actually changed  
- **🔄 Overview Maintenance**: Updates project overview when structure changes
- **🛡️ Error Isolation**: Git operations continue even if indexing fails
- **⏱️ Rate Limiting**: Built-in retry logic with exponential backoff

### 🎯 Key Benefits

💡 **Zero Manual Work**: Descriptions stay current without any effort  
⚡ **Performance**: Only analyzes changed files, not entire codebase  
🔒 **Reliability**: Robust error handling ensures git operations never fail  
🎛️ **Configurable**: Support for custom models and timeout settings  

**Learn More**: See [Git Hook Setup Guide](docs/git-hook-setup.md) for complete configuration options and troubleshooting.

## 🔧 Development Setup

### 👨‍💻 For Contributors

Contributing to MCP Code Indexer? Follow these steps for a proper development environment:

```bash
# Setup development environment
git clone https://github.com/fluffypony/mcp-code-indexer.git
cd mcp-code-indexer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode (REQUIRED for development)
pip install -e .

# Install development dependencies
pip install -e .[dev]

# Verify installation
python main.py --help
mcp-code-indexer --version
```

⚠️ **Important**: The editable install (`pip install -e .`) is **required** for development. The project uses proper PyPI package structure with absolute imports like `from mcp_code_indexer.database.database import DatabaseManager`. Without editable installation, you'll get `ModuleNotFoundError` exceptions.

### 🎯 Development Workflow

```bash
# Activate virtual environment
source venv/bin/activate

# Run the server directly
python main.py --token-limit 32000

# Or use the installed CLI command
mcp-code-indexer --token-limit 32000

# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## 🛠️ MCP Tools Available

The server provides **12 powerful MCP tools** for intelligent codebase management. Whether you're an AI agent or human developer, these tools make navigating code effortless.

### 🎯 For Everyone: Start Here
- **`check_codebase_size`** - Get instant recommendations for how to navigate your codebase
- **`search_descriptions`** - Find files by what they do, not just their names
- **`get_codebase_overview`** - Get a high-level understanding of any project

### 👨‍💻 For Developers: Core Operations
- **`get_file_description`** - Retrieve stored file descriptions instantly
- **`update_file_description`** - Store detailed file summaries and metadata  
- **`find_missing_descriptions`** - Scan projects for files without descriptions
- **`update_missing_descriptions`** - Bulk update multiple file descriptions

### 🔍 For Advanced Users: Search & Discovery
- **`get_all_descriptions`** - Complete hierarchical project structure
- **`get_word_frequency`** - Technical vocabulary analysis with stop-word filtering
- **`update_codebase_overview`** - Create comprehensive codebase documentation

### 🏥 For System Monitoring: Health & Performance
- **`check_database_health`** - Real-time database health monitoring and diagnostics

💡 **Pro Tip**: Always start with `check_codebase_size` to get personalized recommendations for navigating your specific codebase.

## 🔗 Git Hook Integration

Keep your codebase documentation automatically synchronized with automated analysis on every commit:

```bash
# Analyze current staged changes
mcp-code-indexer --githook

# Analyze a specific commit
mcp-code-indexer --githook abc123def

# Analyze a commit range (perfect for rebases)
mcp-code-indexer --githook abc123 def456
```

**🎯 Perfect for**:
- **Automated documentation** that never goes stale
- **Rebase-aware analysis** that handles complex git operations
- **Zero-effort maintenance** with background processing

See the **[Git Hook Setup Guide](docs/git-hook-setup.md)** for complete installation instructions including post-commit, post-merge, and post-rewrite hooks.

## 🏗️ Architecture Highlights

### 🚀 Performance Optimized
- **SQLite with WAL mode** for high-concurrency access (800+ writes/sec)
- **Smart connection pooling** with optimized pool size (3 connections default)
- **FTS5 full-text search** with prefix indexing for sub-100ms queries
- **Token-aware caching** to minimize expensive operations
- **Write operation serialization** to eliminate database lock conflicts

### 🛡️ Production Ready
- **Database resilience features** with <2% error rate under high load
- **Exponential backoff retry logic** with intelligent failure recovery
- **Comprehensive health monitoring** with automatic pool refresh
- **Structured JSON logging** with performance metrics tracking
- **Async-first design** with proper resource cleanup
- **MCP protocol compliant** with clean stdio streams
- **Upstream inheritance** for fork workflows
- **Git integration** with .gitignore support

### 👨‍💻 Developer Friendly
- **95%+ test coverage** with async support and concurrent access tests
- **Integration tests** for complete workflows including database stress testing
- **Performance benchmarks** for large codebases with resilience validation
- **Clear error messages** with MCP protocol compliance
- **Comprehensive configuration options** for production tuning

## 📖 Documentation

### 👤 For Users
- **[Git Hook Setup Guide](docs/git-hook-setup.md)** - Automated code indexing setup
- **[Configuration Guide](docs/configuration.md)** - Production deployment and tuning

### 👨‍💻 For Developers  
- **[API Reference](docs/api-reference.md)** - Complete MCP tool documentation with examples
- **[Architecture Overview](docs/architecture.md)** - Technical deep dive into system design
- **[Database Resilience Guide](docs/database-resilience.md)** - Advanced database optimization and monitoring

### 🔧 For System Administrators
- **[Performance Tuning Guide](docs/performance-tuning.md)** - High-concurrency deployment optimization
- **[Monitoring & Diagnostics](docs/monitoring.md)** - Production monitoring setup and troubleshooting

### 🤝 For Contributors
- **[Contributing Guide](docs/contributing.md)** - Development setup and workflow guidelines

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

### 👨‍💻 For Developers: Basic Configuration

```bash
# Production setup with custom limits
mcp-code-indexer \
  --token-limit 50000 \
  --db-path /data/mcp-index.db \
  --cache-dir /tmp/mcp-cache \
  --log-level INFO

# Enable structured logging
export MCP_LOG_FORMAT=json
mcp-code-indexer
```

### 🔧 For System Administrators: Database Resilience Tuning

Configure advanced database resilience features for high-concurrency environments:

```bash
# High-performance production deployment
mcp-code-indexer \
  --token-limit 64000 \
  --db-path /data/mcp-index.db \
  --cache-dir /var/cache/mcp \
  --log-level INFO \
  --db-pool-size 5 \
  --db-retry-count 7 \
  --db-timeout 15.0 \
  --enable-wal-mode \
  --health-check-interval 20.0

# Environment variable configuration
export DB_POOL_SIZE=5
export DB_RETRY_COUNT=7
export DB_TIMEOUT=15.0
export DB_WAL_MODE=true
export DB_HEALTH_CHECK_INTERVAL=20.0
mcp-code-indexer --token-limit 64000
```

#### Configuration Options

| Parameter | Default | Description | Use Case |
|-----------|---------|-------------|----------|
| `--db-pool-size` | 3 | Database connection pool size | Higher for more concurrent clients |
| `--db-retry-count` | 5 | Max retry attempts for failed operations | Increase for unstable environments |
| `--db-timeout` | 10.0 | Transaction timeout (seconds) | Increase for large operations |
| `--enable-wal-mode` | true | Enable WAL mode for concurrency | Always enable for production |
| `--health-check-interval` | 30.0 | Health monitoring interval (seconds) | Lower for faster issue detection |

💡 **Performance Tip**: For environments with 10+ concurrent clients, use `--db-pool-size 5` and `--health-check-interval 15.0` for optimal throughput.

## 🤝 Integration Examples

### With AI Agents
```python
# Example: AI agent using MCP tools
async def analyze_codebase(project_path):
    # Check if codebase is large
    size_info = await mcp_client.call_tool("check_codebase_size", {
        "projectName": "my-project",
        "folderPath": project_path
    })
    
    if size_info["isLarge"]:
        # Use search for large codebases
        results = await mcp_client.call_tool("search_descriptions", {
            "projectName": "my-project", 
            "folderPath": project_path,
            "query": "authentication logic"
        })
    else:
        # Get full overview for smaller projects
        overview = await mcp_client.call_tool("get_codebase_overview", {
            "projectName": "my-project",
            "folderPath": project_path
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
            'folderPath': '.'
        })
        
        # Process with AI and update...
    
    asyncio.run(update_descriptions())
    "
```

## 🧪 Testing

```bash
# Install with test dependencies
pip install mcp-code-indexer[test]

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

## 📋 Command Line Options

### Server Mode (Default)
```bash
mcp-code-indexer [OPTIONS]

Options:
  --token-limit INT     Maximum tokens before recommending search (default: 32000)
  --db-path PATH        SQLite database path (default: ~/.mcp-code-index/tracker.db)
  --cache-dir PATH      Cache directory path (default: ~/.mcp-code-index/cache)
  --log-level LEVEL     Logging level: DEBUG|INFO|WARNING|ERROR|CRITICAL (default: INFO)
```

### Git Hook Mode
```bash
mcp-code-indexer --githook [OPTIONS]

# Automated analysis of git changes using OpenRouter API
# Requires: OPENROUTER_API_KEY environment variable
```

### Utility Commands
```bash
# List all projects
mcp-code-indexer --getprojects

# Execute MCP tool directly
mcp-code-indexer --runcommand '{"method": "tools/call", "params": {...}}'

# Export descriptions for a project
mcp-code-indexer --dumpdescriptions PROJECT_ID
```

## 🛡️ Security Features

- **Input validation** on all MCP tool parameters
- **SQL injection protection** via parameterized queries  
- **File system sandboxing** with .gitignore respect
- **Error sanitization** to prevent information leakage
- **Async resource cleanup** to prevent memory leaks

## 🚀 Next Steps

Ready to supercharge your AI agents with intelligent codebase navigation?

### 👤 Getting Started
1. **[Install and run your first server](#-quick-start)** - Get up and running in 2 minutes
2. **[Set up git hooks](docs/git-hook-setup.md)** - Automate your workflow
3. **[Configure for production](docs/configuration.md)** - Deploy for your team

### 👨‍💻 For Developers
4. **[Explore the API tools](docs/api-reference.md)** - Master all 11 MCP tools
5. **[Understand the architecture](docs/architecture.md)** - Deep dive into the technical design

### 🤝 Join the Community
6. **[Contribute to the project](docs/contributing.md)** - Help make it even better
7. **[Report issues on GitHub](https://github.com/fluffypony/mcp-code-indexer/issues)** - Share feedback and suggestions

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
- **[aiohttp](https://pypi.org/project/aiohttp/)** - Async HTTP client for OpenRouter API
- **[tenacity](https://pypi.org/project/tenacity/)** - Robust retry logic and rate limiting
- **[Pydantic](https://pydantic.dev/)** - Data validation and settings

---

**Transform how your AI agents understand code!** 🚀  

🎯 **New User?** [Get started in 2 minutes](#-quick-start)  
👨‍💻 **Developer?** [Explore the complete API](docs/api-reference.md)  
🔧 **Production?** [Deploy with confidence](docs/configuration.md)
