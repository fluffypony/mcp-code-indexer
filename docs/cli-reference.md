# CLI Reference 📖

---
**Last Updated:** 2025-06-30
**Verified Against:** main.py, src/mcp_code_indexer/main.py
**Test Sources:** Manual verification of argument parser implementation
**Implementation:** Complete argparse configuration with environment variable support
---

Complete command-line interface reference for MCP Code Indexer. This guide covers all commands, options, and configuration parameters for both server operation and utility functions.

**🎯 New to the CLI?** Start with the [Quick Start Guide](../README.md#-quick-start) to get oriented first.

## Quick Reference

| Mode | Command | Purpose |
|------|---------|---------|
| **Server** | `mcp-code-indexer` | Start MCP server (stdio) |
| **HTTP** | `mcp-code-indexer --http` | Start HTTP/REST API server |
| **Vector (BETA)** | `mcp-code-indexer --vector` | Start with semantic search capabilities |
| **Q&A** | `mcp-code-indexer --ask "question" PROJECT` | Simple AI question answering |
| **Enhanced Q&A** | `mcp-code-indexer --deepask "question" PROJECT` | Enhanced AI analysis with file search |
| **Admin** | `mcp-code-indexer --getprojects` | List all tracked projects |
| **Git Hook** | `mcp-code-indexer --githook` | Auto-update descriptions from git changes |

🚀 **[See Examples →](../examples/)**

## Table of Contents

- [Command Syntax](#command-syntax)
- [Server Modes](#server-modes)
- [Q&A Commands](#qa-commands)
- [Administrative Commands](#administrative-commands)
- [Database Configuration](#database-configuration)
- [HTTP Transport Options](#http-transport-options)
- [Global Options](#global-options)
- [Environment Variables](#environment-variables)
- [Configuration Precedence](#configuration-precedence)

## Command Syntax

```bash
mcp-code-indexer [OPTIONS] [PROJECT_NAME]
```

### Basic Examples

```bash
# Start MCP server (stdio transport)
mcp-code-indexer

# Start HTTP server
mcp-code-indexer --http --port 8080

# Ask a question about a project
mcp-code-indexer --ask "What does this project do?" my-app

# Enhanced question with file search
mcp-code-indexer --deepask "How is authentication implemented?" web-app

# List all projects
mcp-code-indexer --getprojects

# Git hook mode for automated updates
mcp-code-indexer --githook
```

## Server Modes

### MCP Server (Default)

Start the MCP server with stdio transport for AI agent integration.

```bash
mcp-code-indexer [DATABASE_OPTIONS] [GLOBAL_OPTIONS]
```

**Use case:** Integration with AI agents and MCP clients

**Example:**
```bash
# Basic server
mcp-code-indexer

# Custom database location
mcp-code-indexer --db-path "/custom/path/tracker.db"

# High-performance configuration
mcp-code-indexer --db-pool-size 10 --enable-wal-mode
```

### HTTP Server

Start the HTTP/REST API server for web-based access.

```bash
mcp-code-indexer --http [HTTP_OPTIONS] [DATABASE_OPTIONS] [GLOBAL_OPTIONS]
```

**Use case:** Web applications, REST API access, browser integration

**Example:**
```bash
# Basic HTTP server
mcp-code-indexer --http

# Production configuration
mcp-code-indexer --http \
  --host 0.0.0.0 \
  --port 443 \
  --auth-token "$MCP_AUTH_TOKEN" \
  --cors-origins "https://myapp.com"
```

### Vector Mode (BETA)

Enable semantic search capabilities with AI embeddings.

```bash
mcp-code-indexer --vector [VECTOR_OPTIONS] [SERVER_OPTIONS]
```

**Use case:** Semantic code search, AI-powered code discovery, context-aware analysis

**Prerequisites:**
- Requires API keys for Voyage AI and TurboPuffer
- Set API keys: `VOYAGE_API_KEY` and `TURBOPUFFER_API_KEY`

**Examples:**
```bash
# Basic vector mode
mcp-code-indexer --vector

# Vector mode with HTTP API
mcp-code-indexer --vector --http --port 8080

# Custom configuration
mcp-code-indexer --vector --vector-config /path/to/config.yaml

# Vector mode with enhanced logging
mcp-code-indexer --vector --log-level DEBUG
```

**Vector Options:**
- `--vector` - Enable vector mode with semantic search
- `--vector-config PATH` - Path to vector mode configuration file

**Required Environment Variables:**
- `VOYAGE_API_KEY` - API key for Voyage AI embedding generation
- `TURBOPUFFER_API_KEY` - API key for Turbopuffer vector storage

**Status:** Currently in BETA. See [Vector Mode Documentation](vector-mode.md) for complete setup guide.

## Q&A Commands

AI-powered question answering using OpenRouter API and Claude.

### Simple Q&A (--ask)

Quick questions using project overview only.

```bash
mcp-code-indexer --ask "QUESTION" PROJECT_NAME [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--json` | Output response in JSON format |
| `--log-level LEVEL` | Set logging verbosity |

**Examples:**
```bash
# Basic question
mcp-code-indexer --ask "What is this project?" web-app

# JSON output
mcp-code-indexer --ask "What technologies are used?" api-service --json

# Debug mode
mcp-code-indexer --ask "How is it structured?" mobile-app --log-level DEBUG
```

### Enhanced Q&A (--deepask)

Detailed analysis with two-stage processing and file search.

```bash
mcp-code-indexer --deepask "QUESTION" PROJECT_NAME [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--json` | Output response in JSON format |
| `--log-level LEVEL` | Set logging verbosity |

**Examples:**
```bash
# Implementation analysis
mcp-code-indexer --deepask "How is user authentication implemented?" web-app

# Security review
mcp-code-indexer --deepask "What security measures are in place?" api-service --json

# Performance analysis
mcp-code-indexer --deepask "What performance optimizations exist?" large-app
```

**Requirements:**
- `OPENROUTER_API_KEY` environment variable
- Project with existing file descriptions and overview

## Administrative Commands

### List Projects (--getprojects)

Display all tracked projects with file counts.

```bash
mcp-code-indexer --getprojects [DATABASE_OPTIONS]
```

**Output:**
```
Projects:
--------------------------------------------------------------------------------
ID: 1
Name: web-app
Files: 45 descriptions
--------------------------------------------------------------------------------
ID: 2
Name: api-service
Files: 28 descriptions
--------------------------------------------------------------------------------
```

### Execute MCP Command (--runcommand)

Execute MCP tool calls using JSON format.

```bash
mcp-code-indexer --runcommand 'JSON_COMMAND' [DATABASE_OPTIONS]
```

**Examples:**
```bash
# Get file description
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "get_file_description",
    "arguments": {
      "projectName": "web-app",
      "folderPath": "/home/user/web-app",
      "filePath": "src/main.ts"
    }
  }
}'

# Search descriptions
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "search_descriptions",
    "arguments": {
      "projectName": "api-service",
      "folderPath": "/home/user/api-service",
      "query": "authentication middleware"
    }
  }
}'
```

### Export Descriptions (--dumpdescriptions)

Export all file descriptions for a project.

```bash
mcp-code-indexer --dumpdescriptions PROJECT_ID [DATABASE_OPTIONS]
```

**Example:**
```bash
mcp-code-indexer --dumpdescriptions 1 > web-app-descriptions.json
```

### Git Hook Mode (--githook)

Auto-update descriptions based on git changes using OpenRouter API.

```bash
mcp-code-indexer --githook [COMMIT_SPEC] [DATABASE_OPTIONS]
```

**Supported commit specifications:**
- No args: Current working tree changes
- `HASH`: Specific commit
- `HASH1 HASH2`: Commit range
- `HEAD`, `HEAD~1`, `HEAD~3`: Relative references
- Branch names and tags

**Examples:**
```bash
# Current changes
mcp-code-indexer --githook

# Specific commit
mcp-code-indexer --githook abc123def

# Commit range
mcp-code-indexer --githook HEAD~3 HEAD

# Branch comparison
mcp-code-indexer --githook main feature-branch
```

**Requirements:**
- `OPENROUTER_API_KEY` environment variable
- Git repository with valid commits

### Generate Project Map (--map)

Create a markdown project map showing file structure and descriptions.

```bash
mcp-code-indexer --map PROJECT_NAME_OR_ID [DATABASE_OPTIONS]
```

**Example:**
```bash
mcp-code-indexer --map web-app > project-map.md
```

### Cleanup Empty Projects (--cleanup)

Remove projects with no descriptions and no overview.

```bash
mcp-code-indexer --cleanup [DATABASE_OPTIONS]
```

### Create Local Database (--makelocal)

Create a local database in a specific folder and migrate project data.

```bash
mcp-code-indexer --makelocal FOLDER_PATH [DATABASE_OPTIONS]
```

**Example:**
```bash
# Create local database for a project
mcp-code-indexer --makelocal /home/user/my-project
```

## Database Configuration

Configure SQLite database behavior and performance characteristics.

### Connection Management

| Option | Default | Description |
|--------|---------|-------------|
| `--db-path PATH` | `~/.mcp-code-index/tracker.db` | SQLite database file path |
| `--db-pool-size N` | `3` | Connection pool size |
| `--db-timeout SECONDS` | `10.0` | Transaction timeout |
| `--enable-wal-mode` | `true` | Enable WAL mode for concurrency |

**Examples:**
```bash
# Custom database location
mcp-code-indexer --db-path "/opt/mcp/tracker.db"

# High-performance configuration
mcp-code-indexer --db-pool-size 10 --db-timeout 30.0

# Disable WAL mode (single-user scenario)
mcp-code-indexer --enable-wal-mode false
```

### Retry Configuration

Configure retry behavior for transient database failures.

| Option | Default | Description |
|--------|---------|-------------|
| `--db-retry-count N` | `5` | Maximum retry attempts |
| `--retry-min-wait SECONDS` | `0.1` | Minimum wait between retries |
| `--retry-max-wait SECONDS` | `2.0` | Maximum wait between retries |
| `--retry-jitter SECONDS` | `0.2` | Random jitter for retry delays |

**Examples:**
```bash
# Aggressive retry configuration
mcp-code-indexer --db-retry-count 10 --retry-max-wait 5.0

# Conservative retry configuration
mcp-code-indexer --db-retry-count 3 --retry-min-wait 0.5
```

### Health Monitoring

| Option | Default | Description |
|--------|---------|-------------|
| `--health-check-interval SECONDS` | `30.0` | Health check frequency |

**Example:**
```bash
# Frequent health checks
mcp-code-indexer --health-check-interval 10.0
```

## HTTP Transport Options

Configure the HTTP/REST API server when using `--http` mode.

### Network Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--http` | False | Enable HTTP transport |
| `--host HOST` | `127.0.0.1` | Host to bind server to |
| `--port PORT` | `7557` | Port to bind server to |

**Examples:**
```bash
# Bind to all interfaces
mcp-code-indexer --http --host 0.0.0.0

# Custom port
mcp-code-indexer --http --port 8080

# IPv6
mcp-code-indexer --http --host "::" --port 443
```

### Security Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--auth-token TOKEN` | None | Bearer token for authentication |
| `--cors-origins ORIGINS...` | `["*"]` | Allowed CORS origins |

**Examples:**
```bash
# Enable authentication
mcp-code-indexer --http --auth-token "your-secret-token"

# Restrict CORS origins
mcp-code-indexer --http --cors-origins "https://localhost:3000" "https://myapp.com"

# Disable CORS (same-origin only)
mcp-code-indexer --http --cors-origins
```

## Global Options

Options that apply to all modes and commands.

### Core Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--version` | - | Show version and exit |
| `--token-limit N` | `32000` | Token limit for overview recommendations |
| `--cache-dir PATH` | `~/.mcp-code-index/cache` | Cache directory for temporary files |
| `--log-level LEVEL` | `INFO` | Logging verbosity level |

**Log levels:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

**Examples:**
```bash
# Check version
mcp-code-indexer --version

# Custom cache location
mcp-code-indexer --cache-dir "/tmp/mcp-cache"

# Debug logging
mcp-code-indexer --log-level DEBUG

# Higher token limit for large projects
mcp-code-indexer --token-limit 64000
```

### Positional Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `PROJECT_NAME` | For `--ask`, `--deepask` | Project name for Q&A commands |

## Environment Variables

Many CLI options can be set via environment variables for easier configuration management.

### Database Variables

| Variable | CLI Equivalent | Default |
|----------|----------------|---------|
| `DB_POOL_SIZE` | `--db-pool-size` | `3` |
| `DB_RETRY_COUNT` | `--db-retry-count` | `5` |
| `DB_TIMEOUT` | `--db-timeout` | `10.0` |
| `DB_WAL_MODE` | `--enable-wal-mode` | `true` |
| `DB_HEALTH_CHECK_INTERVAL` | `--health-check-interval` | `30.0` |
| `DB_RETRY_MIN_WAIT` | `--retry-min-wait` | `0.1` |
| `DB_RETRY_MAX_WAIT` | `--retry-max-wait` | `2.0` |
| `DB_RETRY_JITTER` | `--retry-jitter` | `0.2` |

### Q&A Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key for Claude access | Yes (for Q&A) |
| `MCP_CLAUDE_MODEL` | Claude model to use | No |
| `MCP_CLAUDE_TOKEN_LIMIT` | Token limit per request | No |

### Examples

```bash
# Database configuration
export DB_POOL_SIZE="10"
export DB_WAL_MODE="true"
export DB_RETRY_COUNT="8"

# Q&A configuration
export OPENROUTER_API_KEY="your-api-key"
export MCP_CLAUDE_MODEL="anthropic/claude-3.5-sonnet"
export MCP_CLAUDE_TOKEN_LIMIT="180000"

# Start server with environment configuration
mcp-code-indexer --http
```

## Configuration Precedence

Configuration values are resolved in this order (highest to lowest priority):

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Default values** (lowest priority)

### Examples

```bash
# Environment variable sets pool size to 5
export DB_POOL_SIZE="5"

# CLI argument overrides to 10 (final value: 10)
mcp-code-indexer --db-pool-size 10

# Environment variable with no CLI override (final value: 5)
mcp-code-indexer
```

### Configuration Files

While the CLI doesn't support configuration files directly, you can use environment files:

```bash
# .env file
DB_POOL_SIZE=10
DB_WAL_MODE=true
OPENROUTER_API_KEY=your-api-key
MCP_CLAUDE_MODEL=anthropic/claude-3.5-sonnet

# Load and run
source .env && mcp-code-indexer --http
```

### Docker Environment

```dockerfile
# Dockerfile
ENV DB_POOL_SIZE=10
ENV DB_WAL_MODE=true
ENV OPENROUTER_API_KEY=your-api-key

CMD ["mcp-code-indexer", "--http", "--host", "0.0.0.0"]
```

### Production Scripts

```bash
#!/bin/bash
# production-start.sh

# Database configuration
export DB_POOL_SIZE="15"
export DB_WAL_MODE="true"
export DB_RETRY_COUNT="10"

# HTTP configuration
export MCP_HOST="0.0.0.0"
export MCP_PORT="443"
export MCP_AUTH_TOKEN="$PRODUCTION_TOKEN"

# Start server
exec mcp-code-indexer --http \
  --host "$MCP_HOST" \
  --port "$MCP_PORT" \
  --auth-token "$MCP_AUTH_TOKEN" \
  --cors-origins "https://yourdomain.com" \
  --log-level INFO
```

---

**Next Steps**: Check out the [Administrative Commands Guide](admin-commands.md) for detailed utility workflows, or review the [HTTP API Reference](http-api.md) for REST endpoint details! 🚀
