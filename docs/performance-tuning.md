# Performance Tuning Guide ⚡

Comprehensive guide for optimizing MCP Code Indexer performance in high-concurrency production environments. Designed for system administrators and DevOps engineers deploying at scale.

**🎯 Looking for basic setup?** Start with the [Configuration Guide](configuration.md) first.

## Table of Contents

- [Performance Targets](#performance-targets)
- [System Requirements](#system-requirements)
- [Configuration Optimization](#configuration-optimization)
- [Deployment Strategies](#deployment-strategies)
- [Load Testing](#load-testing)
- [Performance Monitoring](#performance-monitoring)
- [Scaling Patterns](#scaling-patterns)
- [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

## Performance Targets

### 🎯 Production Benchmarks

| Metric | Target | Exceptional |
|--------|--------|-------------|
| **Throughput** | 800+ writes/sec | 1200+ writes/sec |
| **Response Time** | <50ms (95th percentile) | <25ms (95th percentile) |
| **Error Rate** | <2% under load | <0.5% under load |
| **Availability** | 99.9% | 99.99% |
| **Recovery Time** | <5 seconds | <2 seconds |

### 📊 Performance Categories

- **Single Client**: 200+ operations/sec
- **Low Concurrency (2-5 clients)**: 500+ operations/sec  
- **Medium Concurrency (6-15 clients)**: 800+ operations/sec
- **High Concurrency (16+ clients)**: Consider distributed deployment

## System Requirements

### 🔧 Hardware Recommendations

#### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4GB available
- **Storage**: SSD with 100MB/s write speed
- **Network**: 100 Mbps

#### Recommended for Production
- **CPU**: 4+ cores, 3.0+ GHz
- **RAM**: 8GB+ available  
- **Storage**: NVMe SSD with 500MB/s+ write speed
- **Network**: 1 Gbps+

#### High-Performance Configuration
- **CPU**: 8+ cores, 3.5+ GHz
- **RAM**: 16GB+ available
- **Storage**: High-end NVMe SSD (1GB/s+ write)
- **Network**: 10 Gbps+

### 💾 Storage Optimization

```bash
# Check current disk performance
dd if=/dev/zero of=testfile bs=1M count=1000 oflag=direct
# Target: >100MB/s for production

# Optimize filesystem for SQLite
mount -o noatime,nodiratime /dev/ssd /data

# Set optimal I/O scheduler for SSDs
echo noop > /sys/block/sda/queue/scheduler
```

## Configuration Optimization

### 🚀 High-Performance Configuration

```bash
# Production-optimized deployment
mcp-code-indexer \
  --token-limit 64000 \
  --db-path /data/mcp-index.db \
  --cache-dir /var/cache/mcp \
  --log-level WARNING \
  --db-pool-size 5 \
  --db-retry-count 3 \
  --db-timeout 30.0 \
  --enable-wal-mode \
  --health-check-interval 15.0
```

### Configuration by Workload

#### CPU-Optimized (Compute-Heavy)
```bash
export DB_POOL_SIZE=3
export DB_RETRY_COUNT=5
export DB_TIMEOUT=20.0
export DB_HEALTH_CHECK_INTERVAL=30.0
```

#### I/O-Optimized (Storage-Heavy)  
```bash
export DB_POOL_SIZE=5
export DB_RETRY_COUNT=3
export DB_TIMEOUT=45.0
export DB_HEALTH_CHECK_INTERVAL=10.0
```

#### Network-Optimized (High Client Count)
```bash
export DB_POOL_SIZE=7
export DB_RETRY_COUNT=2
export DB_TIMEOUT=15.0
export DB_HEALTH_CHECK_INTERVAL=5.0
```

### 📊 Parameter Tuning Matrix

| Workload | Pool Size | Retry Count | Timeout | Health Interval |
|----------|-----------|-------------|---------|-----------------|
| Development | 2 | 5 | 10.0 | 60.0 |
| Staging | 3 | 5 | 15.0 | 30.0 |
| Production | 5 | 3 | 30.0 | 15.0 |
| High-Load | 7 | 2 | 45.0 | 5.0 |

## Deployment Strategies

### 🏗️ Single-Instance Deployment

Best for: Up to 15 concurrent clients

```bash
# Systemd service configuration
[Unit]
Description=MCP Code Indexer
After=network.target

[Service]
Type=simple
User=mcp
Group=mcp
Environment=DB_POOL_SIZE=5
Environment=DB_WAL_MODE=true
ExecStart=/usr/local/bin/mcp-code-indexer \
  --token-limit 64000 \
  --db-path /data/mcp-index.db \
  --log-level INFO
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 🔄 Load-Balanced Deployment

Best for: 16+ concurrent clients

```yaml
# Docker Compose with load balancing
version: '3.8'
services:
  mcp-indexer-1:
    image: mcp-code-indexer:latest
    environment:
      - DB_POOL_SIZE=3
      - DB_PATH=/shared/mcp-index-1.db
    volumes:
      - shared-storage:/shared

  mcp-indexer-2:
    image: mcp-code-indexer:latest
    environment:
      - DB_POOL_SIZE=3
      - DB_PATH=/shared/mcp-index-2.db
    volumes:
      - shared-storage:/shared

  load-balancer:
    image: nginx:alpine
    ports:
      - "8080:80"
    depends_on:
      - mcp-indexer-1
      - mcp-indexer-2

volumes:
  shared-storage:
    driver: local
```

### ☁️ Cloud-Native Deployment

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-code-indexer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-code-indexer
  template:
    metadata:
      labels:
        app: mcp-code-indexer
    spec:
      containers:
      - name: mcp-indexer
        image: mcp-code-indexer:latest
        env:
        - name: DB_POOL_SIZE
          value: "5"
        - name: DB_WAL_MODE
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi" 
            cpu: "2000m"
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: mcp-storage
```

## Load Testing

### 🧪 Performance Testing Setup

```python
# Load testing script
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def test_concurrent_operations(client_count: int, operations_per_client: int):
    """Test concurrent MCP operations."""
    
    async def client_workload(client_id: int):
        operations = 0
        start_time = time.time()
        
        for i in range(operations_per_client):
            # Simulate typical MCP operations
            await call_mcp_tool("get_file_description", {
                "projectName": f"test-project-{client_id}",
                "folderPath": "/test/path",
                "branch": "main",
                "filePath": f"src/file_{i}.py"
            })
            operations += 1
            
        duration = time.time() - start_time
        rate = operations / duration
        print(f"Client {client_id}: {rate:.1f} ops/sec")
        return rate
    
    # Run concurrent clients
    tasks = [client_workload(i) for i in range(client_count)]
    rates = await asyncio.gather(*tasks)
    
    total_rate = sum(rates)
    print(f"Total throughput: {total_rate:.1f} ops/sec")
    return total_rate

# Run load tests
asyncio.run(test_concurrent_operations(10, 100))
```

### 📈 Benchmarking Commands

```bash
# Basic throughput test
time for i in {1..1000}; do
  mcp-code-indexer --runcommand '{"method": "tools/call", "params": {"name": "check_codebase_size", "arguments": {...}}}'
done

# Concurrent client simulation
seq 1 10 | xargs -I {} -P 10 bash -c 'for i in {1..100}; do mcp-client-call; done'

# Memory usage profiling
valgrind --tool=massif mcp-code-indexer --token-limit 32000

# Database performance analysis
sqlite3 ~/.mcp-code-index/tracker.db ".timer on" "SELECT COUNT(*) FROM file_descriptions;"
```

## Performance Monitoring

### 📊 Key Performance Indicators

Monitor these metrics for optimal performance:

```json
{
  "performance_metrics": {
    "throughput": {
      "operations_per_second": 850,
      "target": 800,
      "status": "healthy"
    },
    "latency": {
      "p50_ms": 12,
      "p95_ms": 45,
      "p99_ms": 120,
      "target_p95": 50
    },
    "database": {
      "connection_pool_utilization": 0.75,
      "retry_rate": 0.02,
      "health_check_success_rate": 0.995
    },
    "system": {
      "cpu_usage": 0.45,
      "memory_usage_gb": 2.1,
      "disk_io_ops": 1200
    }
  }
}
```

### 🔍 Monitoring Setup

#### Prometheus Metrics Collection

```yaml
# prometheus.yml configuration
scrape_configs:
  - job_name: 'mcp-code-indexer'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

#### Grafana Dashboard Queries

```promql
# Throughput monitoring
rate(mcp_operations_total[5m])

# Latency percentiles  
histogram_quantile(0.95, rate(mcp_operation_duration_seconds_bucket[5m]))

# Error rate
rate(mcp_errors_total[5m]) / rate(mcp_operations_total[5m])

# Database health
mcp_database_pool_active_connections / mcp_database_pool_size
```

#### Alert Rules

```yaml
# alerting.yml
groups:
- name: mcp-code-indexer
  rules:
  - alert: HighErrorRate
    expr: rate(mcp_errors_total[5m]) > 0.02
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "MCP Code Indexer error rate is high"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(mcp_operation_duration_seconds_bucket[5m])) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "MCP Code Indexer latency is high"
```

## Scaling Patterns

### 🔄 Horizontal Scaling

#### Client-Side Load Balancing

```python
import random
from typing import List

class MCPLoadBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.current = 0
    
    def get_server(self) -> str:
        # Round-robin with health checking
        for _ in range(len(self.servers)):
            server = self.servers[self.current]
            self.current = (self.current + 1) % len(self.servers)
            
            if self.is_healthy(server):
                return server
        
        raise Exception("No healthy servers available")
    
    def is_healthy(self, server: str) -> bool:
        # Implement health check logic
        return True

# Usage
balancer = MCPLoadBalancer([
    "mcp-server-1:8080",
    "mcp-server-2:8080", 
    "mcp-server-3:8080"
])
```

#### Database Sharding Strategy

```python
def get_shard_for_project(project_name: str, shard_count: int) -> int:
    """Determine database shard based on project name."""
    return hash(project_name) % shard_count

def get_database_path(project_name: str) -> str:
    """Get database path for project."""
    shard = get_shard_for_project(project_name, 4)
    return f"/data/mcp-index-shard-{shard}.db"
```

### 📈 Vertical Scaling Limits

| Resource | Single Instance Limit | Scaling Strategy |
|----------|----------------------|------------------|
| CPU | 8 cores | Add more instances |
| Memory | 16GB | Database sharding |
| Storage I/O | 1000 IOPS | Multiple storage devices |
| Network | 1000 concurrent connections | Load balancing |

## Troubleshooting Performance Issues

### 🐛 Common Performance Problems

#### Slow Response Times

```bash
# Check database performance
sqlite3 ~/.mcp-code-index/tracker.db "EXPLAIN QUERY PLAN SELECT * FROM file_descriptions WHERE project_id = 1;"

# Analyze query patterns
sqlite3 ~/.mcp-code-index/tracker.db ".timer on" ".once"

# Monitor system resources
top -p $(pgrep mcp-code-indexer)
iotop -a -o -d 1
```

#### High Memory Usage

```python
# Memory profiling
import tracemalloc

tracemalloc.start()

# Run operations
await db_manager.some_operation()

# Get memory statistics
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

tracemalloc.stop()
```

#### Database Lock Contention

```bash
# Check WAL mode status
sqlite3 ~/.mcp-code-index/tracker.db "PRAGMA journal_mode;"

# Monitor lock wait times
sqlite3 ~/.mcp-code-index/tracker.db "PRAGMA wal_checkpoint(FULL);"

# Check database size
ls -lh ~/.mcp-code-index/tracker.db*
```

### 🔧 Performance Optimization Checklist

#### ✅ System Level
- [ ] Enable WAL mode
- [ ] Use SSD storage
- [ ] Optimize connection pool size
- [ ] Configure appropriate timeouts
- [ ] Enable health monitoring

#### ✅ Application Level
- [ ] Implement proper retry logic
- [ ] Use batch operations where possible
- [ ] Monitor error rates
- [ ] Profile memory usage
- [ ] Optimize query patterns

#### ✅ Infrastructure Level
- [ ] Configure load balancing
- [ ] Set up monitoring and alerting
- [ ] Plan capacity scaling
- [ ] Implement backup strategies
- [ ] Document incident response procedures

### 🚨 Performance Incident Response

#### Step 1: Immediate Assessment
```bash
# Check system health
mcp-code-indexer --runcommand '{"method": "tools/call", "params": {"name": "check_database_health"}}'

# Review recent logs
tail -n 100 /var/log/mcp-code-indexer/server.log | grep ERROR

# Check resource usage
top -p $(pgrep mcp-code-indexer)
```

#### Step 2: Quick Fixes
```bash
# Restart service if needed
systemctl restart mcp-code-indexer

# Clear cache if corrupted
rm -rf ~/.mcp-code-index/cache/*

# Force database optimization
sqlite3 ~/.mcp-code-index/tracker.db "PRAGMA optimize;"
```

#### Step 3: Long-term Resolution
1. **Analyze root cause** from logs and metrics
2. **Adjust configuration** based on findings
3. **Implement monitoring** to prevent recurrence
4. **Update documentation** with lessons learned

---

**🎯 Ready to deploy?** Check the [Monitoring & Diagnostics Guide](monitoring.md) for production observability setup.

**🛡️ Need resilience features?** See the [Database Resilience Guide](database-resilience.md) for advanced error handling.
