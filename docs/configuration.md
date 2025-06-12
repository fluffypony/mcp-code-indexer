# Configuration Guide 🔧

Complete guide to configuring and tuning the MCP Code Indexer for optimal performance. Whether you're setting up for development, deploying for a team, or optimizing for production, this guide has you covered.

**🎯 New to configuration?** Start with [Command Line Options](#command-line-options) for basic setup.

## Table of Contents

- [Command Line Options](#command-line-options)
- [Environment Variables](#environment-variables)
- [Database Configuration](#database-configuration)
- [Performance Tuning](#performance-tuning)
- [Logging Configuration](#logging-configuration)
- [Security Settings](#security-settings)
- [Production Deployment](#production-deployment)

## Command Line Options

### 👤 For Users: Basic Setup

The server accepts several command-line arguments for quick configuration:

```bash
mcp-code-indexer [OPTIONS]
```

**Most Important Options**:

| Option | Default | When to Change |
|--------|---------|----------------|
| `--token-limit` | `32000` | **Large codebases**: Use 50000+ for better overview support |
| `--db-path` | `~/.mcp-code-index/tracker.db` | **Custom location**: When you want data elsewhere |
| `--cache-dir` | `~/.mcp-code-index/cache` | **Performance**: Put on SSD for faster access |
| `--log-level` | `INFO` | **Debugging**: Use DEBUG for troubleshooting |

### 🎯 Common Setups

```bash
# Development setup with debug logging
mcp-code-indexer --log-level DEBUG --token-limit 10000

# Production setup with custom paths
mcp-code-indexer \
  --token-limit 50000 \
  --db-path /data/mcp-indexer/database.db \
  --cache-dir /tmp/mcp-cache \
  --log-level WARNING

# High-capacity server for large enterprises
mcp-code-indexer \
  --token-limit 100000 \
  --db-path /mnt/ssd/mcp-database.db \
  --cache-dir /mnt/fast-cache/mcp
```

## Environment Variables

Configure advanced settings through environment variables:

### Core Settings

```bash
# Database configuration
export MCP_DB_POOL_SIZE=10              # Connection pool size (default: 5)
export MCP_DB_TIMEOUT=30000             # Database timeout in milliseconds (default: 30000)
export MCP_DB_WAL_CHECKPOINT=1000       # WAL checkpoint interval (default: 1000)

# Performance settings
export MCP_CACHE_TTL_HOURS=24           # Token cache TTL in hours (default: 24)
export MCP_MAX_FILE_SIZE=10485760       # Max file size to scan in bytes (default: 10MB)
export MCP_SCAN_TIMEOUT=300             # File scan timeout in seconds (default: 300)

# Tiktoken configuration
export TIKTOKEN_CACHE_DIR=./src/tiktoken_cache  # Path to tiktoken cache (auto-set)
```

### Logging Settings

```bash
# Structured logging
export MCP_LOG_FORMAT=json              # Log format: json or text (default: json)
export MCP_LOG_FILE_ENABLED=true        # Enable file logging (default: true)
export MCP_LOG_MAX_SIZE=10MB            # Max log file size (default: 10MB)
export MCP_LOG_BACKUP_COUNT=5           # Number of backup log files (default: 5)

# Debug settings
export MCP_DEBUG_SQL=false              # Log SQL queries (default: false)
export MCP_DEBUG_PERFORMANCE=false      # Log performance metrics (default: false)
```

### Security Settings

```bash
# File system restrictions
export MCP_ALLOWED_PATHS="/home,/opt,/data"  # Comma-separated allowed paths
export MCP_DENY_PATHS="/etc,/var,/sys"       # Comma-separated denied paths
export MCP_MAX_PATH_DEPTH=10                 # Maximum directory depth (default: 10)

# Resource limits
export MCP_MAX_CONCURRENT_TOOLS=20           # Max concurrent tool calls (default: 20)
export MCP_MAX_MEMORY_MB=2048               # Memory limit in MB (default: 2048)
```

## Database Configuration

### SQLite Optimizations

The server automatically applies performance optimizations, but you can tune them:

```bash
# Create a custom configuration file
cat > config/database.conf << EOF
# SQLite PRAGMA settings
pragma.journal_mode=WAL
pragma.synchronous=NORMAL
pragma.cache_size=-64000        # 64MB cache
pragma.temp_store=MEMORY
pragma.mmap_size=268435456      # 256MB memory mapping
pragma.wal_autocheckpoint=1000
pragma.optimize=0x10002         # Enable query planner optimizations
EOF

# Use custom config
export MCP_DB_CONFIG_FILE=config/database.conf
python main.py
```

### Index Optimization

Monitor and optimize database performance:

```bash
# Check database statistics
sqlite3 ~/.mcp-code-index/tracker.db << EOF
.headers on
.mode column

-- Check table sizes
SELECT name, COUNT(*) as rows FROM sqlite_master 
WHERE type='table' AND name NOT LIKE 'sqlite_%'
GROUP BY name;

-- Check index usage
EXPLAIN QUERY PLAN 
SELECT * FROM file_descriptions 
WHERE project_id = 'test' AND branch = 'main';

-- Analyze query performance
.timer on
SELECT COUNT(*) FROM file_descriptions_fts WHERE file_descriptions_fts MATCH 'authentication';
EOF
```

### Backup Configuration

Set up automated database backups:

```bash
#!/bin/bash
# backup-mcp-db.sh

DB_PATH="$1"
BACKUP_DIR="/backups/mcp-indexer"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create backup with SQLite backup API
sqlite3 "$DB_PATH" << EOF
.backup $BACKUP_DIR/mcp-database-$TIMESTAMP.db
.quit
EOF

# Compress backup
gzip "$BACKUP_DIR/mcp-database-$TIMESTAMP.db"

# Clean old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.gz" -mtime +30 -delete

echo "Backup completed: mcp-database-$TIMESTAMP.db.gz"
```

## Performance Tuning

### 🎯 Token Limit Optimization

**What's a token limit?** It determines when the server recommends search vs. full overview. Higher limits = more complete overviews for larger codebases.

| Your Codebase Size | Recommended Limit | Why This Works |
|-------------------|------------------|----------------|
| **Small projects** (< 50 files) | 10,000 - 20,000 | Fast, complete overviews |
| **Medium projects** (50-200 files) | 32,000 (default) | Balanced performance |
| **Large projects** (200+ files) | 50,000 - 100,000 | Handles complex codebases |
| **Enterprise/Monorepos** | 150,000+ | Maximum context for AI agents |

💡 **Tip**: Start with the default (32,000) and increase if you get "use search" recommendations too often.

### Memory Optimization

Configure memory usage based on system resources:

```bash
# For 8GB RAM systems
export MCP_DB_POOL_SIZE=5
export MCP_MAX_MEMORY_MB=1024
python main.py --token-limit 32000

# For 16GB+ RAM systems  
export MCP_DB_POOL_SIZE=10
export MCP_MAX_MEMORY_MB=4096
python main.py --token-limit 50000

# For high-performance servers
export MCP_DB_POOL_SIZE=20
export MCP_MAX_MEMORY_MB=8192
python main.py --token-limit 100000
```

### File System Optimization

Optimize for different storage types:

```bash
# SSD configuration (recommended)
export MCP_DB_WAL_CHECKPOINT=1000
export MCP_CACHE_TTL_HOURS=24

# NFS/Network storage
export MCP_DB_WAL_CHECKPOINT=5000      # Reduce checkpoint frequency
export MCP_CACHE_TTL_HOURS=72          # Longer cache TTL

# Spinning disk optimization
export MCP_DB_WAL_CHECKPOINT=10000     # Minimize disk writes
export MCP_SCAN_TIMEOUT=600            # Longer scan timeout
```

## Logging Configuration

### Structured JSON Logging

Enable detailed structured logging for production monitoring:

```python
# custom_logging.py
import json
from src.logging_config import setup_logging

# Configure for production
logger = setup_logging(
    log_level="INFO",
    log_file=Path("/var/log/mcp-indexer/server.log"),
    enable_file_logging=True,
    max_bytes=50 * 1024 * 1024,  # 50MB
    backup_count=10
)

# Custom log filter for sensitive data
class SensitiveDataFilter:
    def filter(self, record):
        # Remove sensitive information from logs
        if hasattr(record, 'structured_data'):
            data = record.structured_data
            if 'arguments' in data:
                # Redact sensitive paths
                for key, value in data['arguments'].items():
                    if 'path' in key.lower() and '/home/' in str(value):
                        data['arguments'][key] = '[REDACTED_PATH]'
        return True

logger.addFilter(SensitiveDataFilter())
```

### Log Analysis

Monitor server performance with log analysis:

```bash
# Extract performance metrics
jq -r 'select(.tool_usage != null) | [.timestamp, .tool_usage.tool_name, .tool_usage.duration_seconds] | @csv' \
  /var/log/mcp-indexer/server.log > performance_metrics.csv

# Find slow operations
jq 'select(.tool_usage.duration_seconds > 1) | {timestamp, tool: .tool_usage.tool_name, duration: .tool_usage.duration_seconds}' \
  /var/log/mcp-indexer/server.log

# Error rate analysis
jq -r 'select(.level == "ERROR") | .timestamp' /var/log/mcp-indexer/server.log | \
  sort | uniq -c | tail -20
```

## Security Settings

### File System Security

Restrict file system access for security:

```python
# security_config.py
SECURITY_CONFIG = {
    # Allowed base directories
    "allowed_paths": [
        "/home/projects",
        "/opt/workspaces", 
        "/data/repositories"
    ],
    
    # Explicitly denied paths
    "denied_paths": [
        "/etc",
        "/var/log",
        "/sys",
        "/proc",
        "/root"
    ],
    
    # File type restrictions
    "allowed_extensions": [
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".cpp", ".c", ".h", ".hpp",
        ".go", ".rs", ".rb", ".php", ".cs",
        ".md", ".txt", ".json", ".yaml", ".yml",
        ".toml", ".ini", ".cfg", ".conf"
    ],
    
    # Size limits
    "max_file_size_mb": 10,
    "max_files_per_scan": 10000,
    "max_description_length": 2048
}
```

### Input Validation

Configure strict input validation:

```bash
# Validation settings
export MCP_VALIDATE_PATHS=true          # Validate all file paths
export MCP_SANITIZE_INPUTS=true         # Sanitize user inputs
export MCP_MAX_QUERY_LENGTH=500         # Max search query length
export MCP_MAX_DESCRIPTION_LENGTH=2048  # Max description length
export MCP_RATE_LIMIT_ENABLED=true      # Enable rate limiting
export MCP_RATE_LIMIT_PER_MINUTE=100    # Requests per minute limit
```

## Production Deployment

### Systemd Service

Create a systemd service for production deployment:

```ini
# /etc/systemd/system/mcp-code-indexer.service
[Unit]
Description=MCP Code Indexer Server
After=network.target
Requires=network.target

[Service]
Type=simple
User=mcp-indexer
Group=mcp-indexer
WorkingDirectory=/opt/mcp-code-indexer
Environment=PYTHONPATH=/opt/mcp-code-indexer
ExecStart=/opt/mcp-code-indexer/venv/bin/mcp-code-indexer \
  --token-limit 50000 \
  --db-path /data/mcp-indexer/database.db \
  --cache-dir /var/cache/mcp-indexer \
  --log-level INFO

# Resource limits
LimitNOFILE=65536
LimitMEMLOCK=infinity

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/data/mcp-indexer /var/cache/mcp-indexer /var/log/mcp-indexer

# Restart policy
Restart=always
RestartSec=10
StartLimitBurst=3
StartLimitInterval=300

[Install]
WantedBy=multi-user.target
```

### Docker Configuration

Deploy with Docker for isolated environments:

```dockerfile
# Dockerfile.production
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash mcp-indexer

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN chown -R mcp-indexer:mcp-indexer /app

# Switch to application user
USER mcp-indexer

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import asyncio; from src.database.database import DatabaseManager; asyncio.run(DatabaseManager('/data/database.db').initialize())"

# Run the application
CMD ["mcp-code-indexer", "--token-limit", "50000", "--db-path", "/data/database.db"]
```

### Docker Compose

Complete production setup with Docker Compose:

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  mcp-code-indexer:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: mcp-code-indexer
    restart: unless-stopped
    
    environment:
      - MCP_LOG_LEVEL=INFO
      - MCP_DB_POOL_SIZE=10
      - MCP_CACHE_TTL_HOURS=24
      
    volumes:
      - ./data:/data
      - ./logs:/var/log/mcp-indexer
      - ./cache:/var/cache/mcp-indexer
      
    ports:
      - "8000:8000"  # If exposing HTTP interface
      
    # Resource limits
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.25'
    
    # Health check
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Log aggregation (optional)
  fluent-bit:
    image: fluent/fluent-bit:latest
    container_name: mcp-log-collector
    volumes:
      - ./logs:/var/log/mcp-indexer:ro
      - ./fluent-bit.conf:/fluent-bit/etc/fluent-bit.conf
    depends_on:
      - mcp-code-indexer

volumes:
  mcp-data:
    driver: local
  mcp-logs:
    driver: local
```

### Monitoring and Alerting

Set up monitoring with Prometheus metrics:

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
TOOL_CALLS = Counter('mcp_tool_calls_total', 'Total tool calls', ['tool_name', 'status'])
TOOL_DURATION = Histogram('mcp_tool_duration_seconds', 'Tool call duration', ['tool_name'])
ACTIVE_CONNECTIONS = Gauge('mcp_active_connections', 'Active database connections')
CACHE_HITS = Counter('mcp_cache_hits_total', 'Cache hits', ['cache_type'])
ERRORS = Counter('mcp_errors_total', 'Total errors', ['error_type'])

class MetricsMiddleware:
    def __init__(self, port=8080):
        start_http_server(port)
    
    def track_tool_call(self, tool_name, duration, success):
        status = 'success' if success else 'error'
        TOOL_CALLS.labels(tool_name=tool_name, status=status).inc()
        TOOL_DURATION.labels(tool_name=tool_name).observe(duration)
    
    def track_error(self, error_type):
        ERRORS.labels(error_type=error_type).inc()
```

### Performance Monitoring

Monitor key performance indicators:

```bash
# performance_monitor.sh
#!/bin/bash

LOG_FILE="/var/log/mcp-indexer/server.log"
METRICS_FILE="/var/log/mcp-indexer/metrics.txt"

while true; do
    echo "$(date): Performance Check" >> "$METRICS_FILE"
    
    # Database size
    DB_SIZE=$(du -h ~/.mcp-code-index/tracker.db | cut -f1)
    echo "Database Size: $DB_SIZE" >> "$METRICS_FILE"
    
    # Memory usage
    MEMORY=$(ps -o pid,vsz,rss,comm -C python | grep main.py)
    echo "Memory Usage: $MEMORY" >> "$METRICS_FILE"
    
    # Recent error rate
    ERROR_COUNT=$(tail -1000 "$LOG_FILE" | jq -r 'select(.level == "ERROR")' | wc -l)
    echo "Recent Errors: $ERROR_COUNT" >> "$METRICS_FILE"
    
    # Average response time
    AVG_TIME=$(tail -1000 "$LOG_FILE" | jq -r 'select(.tool_usage != null) | .tool_usage.duration_seconds' | \
               awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
    echo "Average Response Time: ${AVG_TIME}s" >> "$METRICS_FILE"
    
    echo "---" >> "$METRICS_FILE"
    sleep 300  # Check every 5 minutes
done
```

---

**Next Steps**: 
- Check out the [Architecture Overview](architecture.md) for technical deep dive
- Review [API Reference](api-reference.md) for tool usage patterns
- Explore [Contributing Guide](contributing.md) for development setup

Need help with configuration? Check the [API Reference](api-reference.md) for tool-specific details or review the [Architecture Overview](architecture.md) for system design! 🚀
