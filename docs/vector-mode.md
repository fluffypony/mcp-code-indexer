# Vector Mode Documentation

## Overview

Vector Mode extends MCP Code Indexer with semantic search capabilities using AI embeddings. Instead of simple text matching, Vector Mode understands code context and meaning to provide intelligent search results.

## Features

### 🔍 Semantic Search
- Find code by meaning, not just keywords
- Context-aware search results
- Cross-language code discovery
- Similarity-based recommendations

### ⚡ Real-time Indexing  
- Automatic embedding generation on file changes
- Efficient change detection using Merkle trees
- Background daemon processing
- Minimal performance impact

### 🛡️ Security First
- Comprehensive secret redaction (20+ pattern types)
- API keys, tokens, credentials automatically detected
- Configurable confidence thresholds
- No sensitive data sent to external APIs

### 🌐 Multi-language Support
- Python: Full AST parsing with functions, classes, methods
- JavaScript/TypeScript: Modern syntax support 
- Fallback chunking for unsupported languages
- Tree-sitter integration for precise parsing

## Quick Start

### Prerequisites

Vector Mode requires additional dependencies and API keys:

```bash
# Install with vector dependencies
pip install mcp-code-indexer[vector]

# Required API keys
export VOYAGE_API_KEY="pa-your-voyage-api-key"      # For embeddings
export TURBOPUFFER_API_KEY="your-turbopuffer-key"   # For vector storage
```

### Basic Usage

```bash
# Enable vector mode
mcp-code-indexer --vector

# Vector mode with HTTP API
mcp-code-indexer --vector --http --port 8080

# Custom configuration
mcp-code-indexer --vector --vector-config config.yaml
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `VOYAGE_API_KEY` | Voyage AI API key for embeddings | Yes |
| `TURBOPUFFER_API_KEY` | Turbopuffer API key for vector storage | Yes |
| `VECTOR_EMBEDDING_MODEL` | Embedding model (default: voyage-code-2) | No |
| `VECTOR_BATCH_SIZE` | Batch size for embeddings (default: 128) | No |
| `VECTOR_SIMILARITY_THRESHOLD` | Search similarity threshold (default: 0.5) | No |

### Configuration File

Create a YAML configuration file for advanced settings:

```yaml
# vector-config.yaml
embedding_model: "voyage-code-2"
batch_size: 128
max_tokens_per_chunk: 1024
similarity_threshold: 0.5
max_search_results: 20

# Chunking settings
max_chunk_size: 1500
chunk_overlap: 100
prefer_semantic_chunks: true

# File monitoring
watch_debounce_ms: 100
ignore_patterns:
  - "*.log"
  - "*.tmp" 
  - ".git/*"
  - "__pycache__/*"

# Daemon settings
daemon_enabled: true
daemon_poll_interval: 5
max_queue_size: 1000
worker_count: 3

# Security
redact_secrets: true
```

## Architecture

### Components

```
Vector Mode Architecture
├── CLI Integration (--vector flag)
├── Configuration Management (YAML + env vars)
├── Vector Daemon (background processing)
├── External Services
│   ├── Voyage AI (embeddings)
│   └── Turbopuffer (vector storage)
├── Security Layer
│   ├── Secret Detection (20+ patterns)
│   └── Content Redaction
├── File Monitoring
│   ├── Real-time Watching (watchdog)
│   ├── Merkle Trees (change detection)
│   └── Event Processing
├── Code Analysis
│   ├── AST Parsing (Tree-sitter)
│   ├── Language Handlers (Python, JS, TS)
│   └── Chunk Optimization
└── Database Extensions
    ├── Code Chunks (semantic units)
    ├── Merkle Nodes (change tracking)
    └── Index Metadata (progress tracking)
```

### Data Flow

1. **File Change Detection**: Watchdog monitors filesystem
2. **Merkle Tree Update**: Efficient change detection
3. **AST Parsing**: Language-specific semantic analysis
4. **Code Chunking**: Extract functions, classes, methods
5. **Secret Redaction**: Remove sensitive information
6. **Embedding Generation**: Voyage AI creates vectors
7. **Vector Storage**: Turbopuffer stores embeddings
8. **Search Interface**: MCP tools provide search access

## Security

### Secret Redaction

Vector Mode automatically detects and redacts sensitive information:

**API Keys & Tokens**:
- AWS Access Keys (`AKIA...`)
- GitHub Tokens (`gh[pousr]_...`)
- OpenAI API Keys (`sk-...`)
- Google API Keys (`AIza...`)
- Slack Tokens (`xox[baprs]-...`)
- And 15+ more patterns

**Database Credentials**:
- Connection strings (PostgreSQL, MySQL, MongoDB)
- Database URLs with embedded credentials
- Environment variable passwords

**Private Keys**:
- RSA, SSH, EC private keys
- PEM formatted certificates

**Configuration**:
```bash
# Adjust redaction sensitivity
export VECTOR_REDACTION_CONFIDENCE=0.7  # Higher = fewer false positives

# Disable redaction (NOT recommended)
export VECTOR_REDACT_SECRETS=false
```

### Best Practices

- **Never disable redaction** in production
- **Review redaction logs** for false positives
- **Use environment-specific configs** for different sensitivity levels
- **Monitor API usage** to detect potential data leaks

## API Reference

### New MCP Tools

When vector mode is enabled, additional MCP tools become available:

#### `vector_search`
```json
{
  "name": "vector_search",
  "description": "Semantic search across codebase using vector embeddings",
  "parameters": {
    "projectName": "string",
    "folderPath": "string", 
    "query": "string",
    "top_k": "number (optional, default: 10)",
    "similarity_threshold": "number (optional, default: 0.5)",
    "chunk_types": "array (optional, filter by chunk types)"
  }
}
```

#### `similarity_search`
```json
{
  "name": "similarity_search", 
  "description": "Find code similar to a given example",
  "parameters": {
    "projectName": "string",
    "folderPath": "string",
    "reference_code": "string",
    "top_k": "number (optional, default: 5)"
  }
}
```

#### `vector_status`
```json
{
  "name": "vector_status",
  "description": "Get vector indexing status for a project", 
  "parameters": {
    "projectName": "string",
    "folderPath": "string"
  }
}
```

### HTTP API Extensions

Vector mode adds REST endpoints when `--http` is enabled:

```bash
# Vector search
POST /mcp
{
  "method": "tools/call",
  "params": {
    "name": "vector_search",
    "arguments": {
      "projectName": "my-project",
      "folderPath": "/path/to/project",
      "query": "authentication middleware"
    }
  }
}

# Get vector status  
POST /mcp
{
  "method": "tools/call",
  "params": {
    "name": "vector_status", 
    "arguments": {
      "projectName": "my-project",
      "folderPath": "/path/to/project"
    }
  }
}
```

## Troubleshooting

### Common Issues

**1. API Key Errors**
```bash
# Error: Missing API keys for vector mode
export VOYAGE_API_KEY="pa-your-key"
export TURBOPUFFER_API_KEY="your-key"
```

**2. Dependencies Missing**
```bash
# Error: Vector mode requires additional dependencies
pip install mcp-code-indexer[vector]
```

**3. Daemon Not Starting**
```bash
# Check daemon status
mcp-code-indexer --vector --log-level DEBUG

# Manual daemon control
python -m mcp_code_indexer.vector_mode.daemon --help
```

**4. Poor Search Results**
```bash
# Adjust similarity threshold
export VECTOR_SIMILARITY_THRESHOLD=0.3  # Lower = more results

# Force re-indexing
# (Delete vector data and restart - feature coming soon)
```

### Performance Tuning

**Large Codebases**:
- Increase `batch_size` for faster embedding generation
- Reduce `worker_count` to limit resource usage
- Adjust `max_chunk_size` for optimal embedding quality

**Rate Limiting**:
- Voyage AI: Built-in rate limiting and retry logic
- Turbopuffer: Auto-scaling vector storage
- Monitor usage through logs and API dashboards

### Monitoring

**Daemon Statistics**:
```bash
# View processing stats (planned)
curl http://localhost:8080/vector/stats

# Database health
mcp-code-indexer --runcommand '{"method": "tools/call", "params": {"name": "check_database_health", "arguments": {}}}'
```

**Redaction Logs**:
```bash
# Check what was redacted
tail -f ~/.mcp-code-index/cache/server.log | grep redaction
```

## Development Status

### ✅ Implemented (BETA)
- Vector mode infrastructure
- Database schema extensions  
- External service providers (Voyage AI, Turbopuffer)
- Secret redaction engine
- File system monitoring with Merkle trees
- AST-based code chunking
- Configuration management

### 🚧 In Development
- Embedding generation pipeline
- Vector search engine
- MCP tool extensions
- Enhanced project overview
- Complete vector daemon
- HTTP API extensions

### 📋 Planned
- Visual search interface
- Code similarity recommendations
- Integration with IDEs
- Advanced analytics dashboard
- Multi-project search
- Collaborative features

## Contributing

Vector Mode is actively developed. To contribute:

1. **Setup Development Environment**:
```bash
git clone https://github.com/fluffypony/mcp-code-indexer.git
cd mcp-code-indexer
poetry install --extras vector
```

2. **Run Tests**:
```bash
# Vector mode tests (when available)
pytest tests/vector_mode/
```

3. **Code Style**:
- Follow existing patterns
- Use type hints throughout
- Add comprehensive docstrings
- Include error handling

See [Contributing Guide](contributing.md) for detailed guidelines.

## Support

- **Documentation**: [docs/](.)
- **Issues**: [GitHub Issues](https://github.com/fluffypony/mcp-code-indexer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fluffypony/mcp-code-indexer/discussions)

For vector mode specific issues, include:
- Vector mode configuration
- API key status (without revealing keys)
- Relevant log excerpts
- Codebase size and languages
