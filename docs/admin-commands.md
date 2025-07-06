# Administrative Commands Guide 🛠️

---
**Last Updated:** 2025-06-30
**Verified Against:** main.py, src/mcp_code_indexer/main.py
**Test Sources:** Manual verification of administrative command implementations
**Implementation:** Command handlers with database operations and project management workflows
---

Comprehensive guide to administrative commands and maintenance workflows for MCP Code Indexer. These commands help you manage projects, troubleshoot issues, and maintain your codebase indexing system.

**🎯 New to administration?** Start with the [CLI Reference](cli-reference.md) to understand basic commands first.

## Quick Reference

| Command | Purpose | Use Case |
|---------|---------|----------|
| [`--getprojects`](#list-projects) | List all tracked projects | Project discovery and overview |
| [`--runcommand`](#execute-mcp-tools) | Execute MCP tools directly | Testing and automation |
| [`--dumpdescriptions`](#export-descriptions) | Export project descriptions | Backup and migration |
| [`--makelocal`](#create-local-database) | Create local project database | Project isolation |
| [`--cleanup`](#cleanup-empty-projects) | Remove empty projects | Database maintenance |
| [`--map`](#generate-project-map) | Generate project documentation | Documentation creation |

🔧 **[See Troubleshooting →](../docs/monitoring.md)**

## Table of Contents

- [Project Management](#project-management)
- [Data Operations](#data-operations)
- [Database Maintenance](#database-maintenance)
- [Troubleshooting Workflows](#troubleshooting-workflows)
- [Backup and Migration](#backup-and-migration)
- [Automation Scenarios](#automation-scenarios)
- [Best Practices](#best-practices)

## Project Management

### List Projects

Get an overview of all tracked projects with file counts and metadata.

```bash
mcp-code-indexer --getprojects [DATABASE_OPTIONS]
```

#### Output Format

```
Projects:
--------------------------------------------------------------------------------
ID: 1
Name: web-application
Files: 45 descriptions
--------------------------------------------------------------------------------
ID: 2
Name: api-service
Files: 28 descriptions
--------------------------------------------------------------------------------
ID: 3
Name: mobile-app
Files: 67 descriptions
--------------------------------------------------------------------------------
```

#### Use Cases

**Project Discovery:**
```bash
# Find all tracked projects
mcp-code-indexer --getprojects

# With custom database
mcp-code-indexer --getprojects --db-path "/backup/tracker.db"
```

**Health Assessment:**
```bash
# Identify projects with no descriptions
mcp-code-indexer --getprojects | grep "Files: 0"

# Find large projects that might need optimization
mcp-code-indexer --getprojects | grep -E "Files: [0-9]{3,}"
```

**Inventory Management:**
```bash
# Get project count for capacity planning
mcp-code-indexer --getprojects | grep -c "ID:"

# Export project list for documentation
mcp-code-indexer --getprojects > project-inventory.txt
```

### Generate Project Map

Create comprehensive markdown documentation showing project structure and file descriptions.

```bash
mcp-code-indexer --map PROJECT_NAME_OR_ID [DATABASE_OPTIONS]
```

#### Examples

```bash
# Generate map by project name
mcp-code-indexer --map "web-application" > docs/project-map.md

# Generate map by project ID
mcp-code-indexer --map 1 > web-app-structure.md

# Generate with custom database
mcp-code-indexer --map "api-service" --db-path "/backup/tracker.db" > api-docs.md
```

#### Use Cases

**Documentation Generation:**
```bash
# Create project onboarding docs
mcp-code-indexer --map "new-service" > docs/onboarding/new-service.md

# Generate architecture documentation
mcp-code-indexer --map "core-platform" > architecture/core-platform.md
```

**Code Review Preparation:**
```bash
# Create context for code reviews
mcp-code-indexer --map "feature-branch-project" > reviews/feature-context.md
```

**Knowledge Transfer:**
```bash
# Document legacy systems
mcp-code-indexer --map "legacy-system" > knowledge-transfer/legacy-overview.md
```

## Data Operations

### Execute MCP Tools

Execute MCP tool calls directly using JSON format for testing, automation, and debugging.

```bash
mcp-code-indexer --runcommand 'JSON_COMMAND' [DATABASE_OPTIONS]
```

#### Tool Call Format

```json
{
  "method": "tools/call",
  "params": {
    "name": "TOOL_NAME",
    "arguments": {
      "param1": "value1",
      "param2": "value2"
    }
  }
}
```

#### Examples

**File Description Operations:**
```bash
# Get file description
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "get_file_description",
    "arguments": {
      "projectName": "web-app",
      "folderPath": "/home/user/web-app",
      "filePath": "src/components/UserProfile.tsx"
    }
  }
}'

# Update file description
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "update_file_description",
    "arguments": {
      "projectName": "web-app",
      "folderPath": "/home/user/web-app",
      "filePath": "src/utils/apiClient.ts",
      "description": "HTTP client utility with authentication and retry logic"
    }
  }
}'
```

**Search and Discovery:**
```bash
# Search file descriptions
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "search_descriptions",
    "arguments": {
      "projectName": "api-service",
      "folderPath": "/home/user/api-service",
      "query": "authentication middleware",
      "maxResults": 5
    }
  }
}'

# Find missing descriptions
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "find_missing_descriptions",
    "arguments": {
      "projectName": "mobile-app",
      "folderPath": "/home/user/mobile-app",
      "limit": 10
    }
  }
}'
```

**Health Monitoring:**
```bash
# Check database health
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "check_database_health",
    "arguments": {}
  }
}'

# Check codebase size
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "check_codebase_size",
    "arguments": {
      "projectName": "large-enterprise-app",
      "folderPath": "/home/user/enterprise-app"
    }
  }
}'
```

#### Use Cases

**Testing and Validation:**
```bash
# Test tool functionality
mcp-code-indexer --runcommand '{"method": "tools/call", "params": {"name": "check_database_health", "arguments": {}}}'

# Validate project setup
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "get_codebase_overview",
    "arguments": {
      "projectName": "test-project",
      "folderPath": "/tmp/test-project"
    }
  }
}'
```

**Automation Scripts:**
```bash
#!/bin/bash
# automation/health-check.sh

# Check database health
HEALTH=$(mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {"name": "check_database_health", "arguments": {}}
}')

# Parse results and take action
if echo "$HEALTH" | jq -r '.health_status.overall_health' | grep -q "unhealthy"; then
    echo "Database health issue detected!"
    # Send alert or restart service
fi
```

### Export Descriptions

Export all file descriptions for a project to JSON format for backup, migration, or analysis.

```bash
mcp-code-indexer --dumpdescriptions PROJECT_ID [DATABASE_OPTIONS]
```

#### Examples

```bash
# Export to file
mcp-code-indexer --dumpdescriptions 1 > backups/web-app-descriptions.json

# Export from custom database
mcp-code-indexer --dumpdescriptions 2 --db-path "/backup/tracker.db" > api-service-export.json

# Export multiple projects
for id in 1 2 3; do
  mcp-code-indexer --dumpdescriptions $id > "backup-project-$id.json"
done
```

#### Use Cases

**Backup Operations:**
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y-%m-%d)
BACKUP_DIR="/backups/mcp-descriptions/$DATE"
mkdir -p "$BACKUP_DIR"

# Get all project IDs
PROJECT_IDS=$(mcp-code-indexer --getprojects | grep "^ID:" | cut -d' ' -f2)

# Export each project
for id in $PROJECT_IDS; do
    mcp-code-indexer --dumpdescriptions $id > "$BACKUP_DIR/project-$id.json"
done
```

**Migration Preparation:**
```bash
# Export before major updates
mcp-code-indexer --dumpdescriptions 1 > migration/pre-update-export.json

# Verify export completeness
jq '.descriptions | length' migration/pre-update-export.json
```

**Analysis and Reporting:**
```bash
# Export for analysis
mcp-code-indexer --dumpdescriptions 1 | jq '.descriptions | length'
mcp-code-indexer --dumpdescriptions 1 | jq '.descriptions[].description' | wc -w
```

## Database Maintenance

### Cleanup Empty Projects

Remove projects that have no file descriptions and no project overview to free up database space.

```bash
mcp-code-indexer --cleanup [DATABASE_OPTIONS]
```

#### Examples

```bash
# Basic cleanup
mcp-code-indexer --cleanup

# Cleanup with custom database
mcp-code-indexer --cleanup --db-path "/opt/mcp/tracker.db"

# Dry run simulation (check what would be cleaned)
mcp-code-indexer --getprojects | grep "Files: 0"
```

#### Use Cases

**Regular Maintenance:**
```bash
# Weekly cleanup cron job
0 2 * * 0 /usr/local/bin/mcp-code-indexer --cleanup >> /var/log/mcp-cleanup.log 2>&1
```

**Pre-migration Cleanup:**
```bash
# Clean before major operations
mcp-code-indexer --cleanup
mcp-code-indexer --getprojects  # Verify results
```

### Create Local Database

Create a project-specific database in a folder and migrate relevant data from the global database.

```bash
mcp-code-indexer --makelocal FOLDER_PATH [DATABASE_OPTIONS]
```

#### Examples

```bash
# Create local database for a project
mcp-code-indexer --makelocal /home/user/my-project

# Create with custom source database
mcp-code-indexer --makelocal /opt/projects/api-service --db-path "/backup/tracker.db"
```

#### Use Cases

**Project Isolation:**
```bash
# Isolate project data
cd /home/user/important-project
mcp-code-indexer --makelocal .

# Verify local database creation
ls -la .mcp-code-index/
```

**Development Environment Setup:**
```bash
# Setup development environment
mcp-code-indexer --makelocal /dev/environments/feature-branch
cd /dev/environments/feature-branch
mcp-code-indexer --getprojects  # Should show only relevant projects
```

**Deployment Preparation:**
```bash
# Create deployment-specific database
mcp-code-indexer --makelocal /deploy/staging/app
# Deploy with isolated database
```

## Troubleshooting Workflows

### Database Health Assessment

Comprehensive workflow to assess and resolve database issues.

```bash
# Step 1: Check database health
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {"name": "check_database_health", "arguments": {}}
}' | jq '.health_status'

# Step 2: List projects to check for corruption
mcp-code-indexer --getprojects

# Step 3: Test a specific tool
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "search_descriptions",
    "arguments": {
      "projectName": "test-project",
      "folderPath": "/tmp/test",
      "query": "test"
    }
  }
}'
```

### Performance Investigation

Workflow to investigate and resolve performance issues.

```bash
# Check database configuration
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {"name": "check_database_health", "arguments": {}}
}' | jq '.health_status.database'

# Identify large projects
mcp-code-indexer --getprojects | grep -E "Files: [0-9]{3,}"

# Check codebase sizes
for project in "large-app" "enterprise-service"; do
  echo "=== $project ==="
  mcp-code-indexer --runcommand "{
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"check_codebase_size\",
      \"arguments\": {
        \"projectName\": \"$project\",
        \"folderPath\": \"/home/user/$project\"
      }
    }
  }" | jq '.totalTokens, .isLarge'
done
```

### Data Consistency Verification

Workflow to verify data integrity and consistency.

```bash
# Verify project data integrity
PROJECT_NAME="web-app"
PROJECT_PATH="/home/user/web-app"

# Check project overview exists
mcp-code-indexer --runcommand "{
  \"method\": \"tools/call\",
  \"params\": {
    \"name\": \"get_codebase_overview\",
    \"arguments\": {
      \"projectName\": \"$PROJECT_NAME\",
      \"folderPath\": \"$PROJECT_PATH\"
    }
  }
}" | jq '.overview | length'

# Check for missing descriptions
mcp-code-indexer --runcommand "{
  \"method\": \"tools/call\",
  \"params\": {
    \"name\": \"find_missing_descriptions\",
    \"arguments\": {
      \"projectName\": \"$PROJECT_NAME\",
      \"folderPath\": \"$PROJECT_PATH\"
    }
  }
}" | jq '.totalMissing'

# Verify search functionality
mcp-code-indexer --runcommand "{
  \"method\": \"tools/call\",
  \"params\": {
    \"name\": \"search_descriptions\",
    \"arguments\": {
      \"projectName\": \"$PROJECT_NAME\",
      \"folderPath\": \"$PROJECT_PATH\",
      \"query\": \"component\"
    }
  }
}" | jq '.totalResults'
```

## Backup and Migration

### Complete Backup Strategy

Comprehensive backup workflow for all MCP Code Indexer data.

```bash
#!/bin/bash
# backup-mcp-data.sh

DATE=$(date +%Y-%m-%d-%H%M)
BACKUP_DIR="/backups/mcp-code-indexer/$DATE"
DB_PATH="$HOME/.mcp-code-index/tracker.db"
CACHE_DIR="$HOME/.mcp-code-index/cache"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# 1. Backup database file
cp "$DB_PATH" "$BACKUP_DIR/tracker.db"

# 2. Backup cache directory
tar -czf "$BACKUP_DIR/cache.tar.gz" -C "$CACHE_DIR" .

# 3. Export all project descriptions
PROJECT_IDS=$(mcp-code-indexer --getprojects | grep "^ID:" | cut -d' ' -f2)
mkdir -p "$BACKUP_DIR/descriptions"

for id in $PROJECT_IDS; do
    mcp-code-indexer --dumpdescriptions $id > "$BACKUP_DIR/descriptions/project-$id.json"
done

# 4. Export project list
mcp-code-indexer --getprojects > "$BACKUP_DIR/project-list.txt"

# 5. Create backup manifest
cat > "$BACKUP_DIR/manifest.txt" << EOF
MCP Code Indexer Backup
Date: $(date)
Database: tracker.db
Cache: cache.tar.gz
Projects: $(echo $PROJECT_IDS | wc -w)
Description Files: descriptions/project-*.json
Project List: project-list.txt
EOF

echo "Backup completed: $BACKUP_DIR"
```

### Migration Workflow

Migrate data between different MCP Code Indexer installations.

```bash
#!/bin/bash
# migrate-mcp-data.sh

SOURCE_DB="/backup/old-system/tracker.db"
TARGET_DB="$HOME/.mcp-code-index/tracker.db"
TEMP_DIR="/tmp/mcp-migration"

# Create temporary directory
mkdir -p "$TEMP_DIR"

# 1. Stop current MCP server (if running)
# pkill -f mcp-code-indexer

# 2. Backup current database
cp "$TARGET_DB" "$TARGET_DB.backup.$(date +%Y%m%d)"

# 3. Export all projects from source database
PROJECT_IDS=$(mcp-code-indexer --getprojects --db-path "$SOURCE_DB" | grep "^ID:" | cut -d' ' -f2)

for id in $PROJECT_IDS; do
    mcp-code-indexer --dumpdescriptions $id --db-path "$SOURCE_DB" > "$TEMP_DIR/project-$id.json"
done

# 4. Import data to target system (manual step - use MCP tools to recreate)
echo "Migration data prepared in $TEMP_DIR"
echo "Next steps:"
echo "1. Use MCP tools to recreate projects"
echo "2. Import descriptions using update_file_description tool"
echo "3. Verify data integrity"
```

## Automation Scenarios

### Health Monitoring Script

Automated health monitoring with alerting.

```bash
#!/bin/bash
# monitor-mcp-health.sh

LOG_FILE="/var/log/mcp-health.log"
ALERT_EMAIL="admin@example.com"

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Check database health
HEALTH_OUTPUT=$(mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {"name": "check_database_health", "arguments": {}}
}' 2>/dev/null)

if [ $? -ne 0 ]; then
    log "ERROR: Failed to execute health check"
    echo "MCP Health Check Failed" | mail -s "MCP Alert" "$ALERT_EMAIL"
    exit 1
fi

# Parse health status
OVERALL_HEALTH=$(echo "$HEALTH_OUTPUT" | jq -r '.health_status.overall_health' 2>/dev/null)

case "$OVERALL_HEALTH" in
    "healthy")
        log "INFO: System is healthy"
        ;;
    "degraded")
        log "WARNING: System performance is degraded"
        echo "MCP System Degraded - Check logs" | mail -s "MCP Warning" "$ALERT_EMAIL"
        ;;
    "unhealthy")
        log "ERROR: System is unhealthy"
        echo "MCP System Unhealthy - Immediate attention required" | mail -s "MCP Critical Alert" "$ALERT_EMAIL"
        ;;
    *)
        log "ERROR: Unknown health status: $OVERALL_HEALTH"
        ;;
esac

# Check project count
PROJECT_COUNT=$(mcp-code-indexer --getprojects 2>/dev/null | grep -c "^ID:")
log "INFO: Tracking $PROJECT_COUNT projects"

# Cleanup old logs (keep 30 days)
find /var/log -name "mcp-health.log*" -mtime +30 -delete
```

### Batch Description Updates

Automate description updates for multiple files.

```bash
#!/bin/bash
# batch-update-descriptions.sh

PROJECT_NAME="$1"
PROJECT_PATH="$2"
DESCRIPTION_FILE="$3"

if [ $# -ne 3 ]; then
    echo "Usage: $0 PROJECT_NAME PROJECT_PATH DESCRIPTION_FILE"
    echo "Description file format: filepath:description (one per line)"
    exit 1
fi

# Process description file
while IFS=':' read -r filepath description; do
    # Skip empty lines and comments
    [[ -z "$filepath" || "$filepath" =~ ^#.*$ ]] && continue

    echo "Updating: $filepath"

    # Update file description
    mcp-code-indexer --runcommand "{
      \"method\": \"tools/call\",
      \"params\": {
        \"name\": \"update_file_description\",
        \"arguments\": {
          \"projectName\": \"$PROJECT_NAME\",
          \"folderPath\": \"$PROJECT_PATH\",
          \"filePath\": \"$filepath\",
          \"description\": \"$description\"
        }
      }
    }"

    sleep 0.1  # Rate limiting
done < "$DESCRIPTION_FILE"

echo "Batch update completed"
```

## Best Practices

### Regular Maintenance Schedule

```bash
# Weekly maintenance cron jobs

# Sunday 2 AM: Cleanup empty projects
0 2 * * 0 /usr/local/bin/mcp-code-indexer --cleanup

# Daily 3 AM: Health check
0 3 * * * /opt/scripts/monitor-mcp-health.sh

# Monthly backup (1st day, 4 AM)
0 4 1 * * /opt/scripts/backup-mcp-data.sh
```

### Performance Optimization

```bash
# Identify performance bottlenecks
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {"name": "check_database_health", "arguments": {}}
}' | jq '.health_status.performance'

# Check large projects that might need optimization
mcp-code-indexer --getprojects | awk '$2 == "Files:" && $3 > 100 {print $0}'

# Optimize database (run during maintenance window)
mcp-code-indexer --cleanup
```

### Security Considerations

```bash
# Secure database file permissions
chmod 600 ~/.mcp-code-index/tracker.db

# Secure backup files
chmod 600 /backups/mcp-code-indexer/*

# Rotate logs to prevent disk space issues
logrotate /etc/logrotate.d/mcp-code-indexer
```

### Error Recovery Procedures

```bash
# Database corruption recovery
cp ~/.mcp-code-index/tracker.db ~/.mcp-code-index/tracker.db.corrupted
cp /backup/latest/tracker.db ~/.mcp-code-index/tracker.db

# Verify recovery
mcp-code-indexer --getprojects
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {"name": "check_database_health", "arguments": {}}
}'
```

---

**Next Steps**: Check out the [Monitoring Guide](monitoring.md) for production monitoring strategies, or review the [Database Resilience](database-resilience.md) documentation for advanced reliability features! 🚀
