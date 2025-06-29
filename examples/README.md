# MCP Code Indexer Examples 🚀

---
**✨ What you'll find**: Practical examples, working configurations, and ready-to-use integrations  
**🎯 Best for**: Developers wanting to quickly integrate MCP Code Indexer into their workflow  
**📖 Related**: [Git Hook Setup Guide](../docs/git-hook-setup.md) • [API Reference](../docs/api-reference.md) • [Configuration Guide](../docs/configuration.md)  
---

Welcome to the MCP Code Indexer examples! This directory contains practical, tested examples to help you integrate automated code analysis into your development workflow.

## 🔗 Git Hook Integration Examples

Transform your development workflow with **zero-effort documentation maintenance**. These git hooks automatically update file descriptions whenever you commit changes.

### Available Hooks

| Hook | Purpose | When It Runs | Best For |
|------|---------|--------------|----------|
| `post-commit` | Basic commit analysis | After `git commit` | Simple workflows |
| `post-rewrite` ⭐ | Advanced rebase handling | After `git rebase`, `git commit --amend` | Professional development |

### Quick Setup (2 Minutes)

**Step 1: Set Your API Key**
```bash
export OPENROUTER_API_KEY="your-api-key-here"
# Add to ~/.bashrc or ~/.zshrc for persistence
```

**Step 2: Install the Hook**
```bash
# For most users (recommended)
cp examples/post-rewrite-hook .git/hooks/post-rewrite
chmod +x .git/hooks/post-rewrite

# For simple workflows
cp examples/post-commit-hook .git/hooks/post-commit
chmod +x .git/hooks/post-commit
```

**Step 3: Test It**
```bash
# Test with current changes
mcp-code-indexer --githook

# You should see: "Git hook analysis completed successfully"
```

✅ **Done!** Your repository now automatically maintains file descriptions.

## 🧪 Testing & Verification

### Manual Testing Commands

```bash
# Test current staged changes
mcp-code-indexer --githook

# Test specific commit (replace with actual commit hash)
mcp-code-indexer --githook a1b2c3d

# Test commit range (useful for understanding rebase behavior)
mcp-code-indexer --githook HEAD~3 HEAD

# Test with different branches
git checkout feature/new-api
mcp-code-indexer --githook HEAD~1
```

### What Happens During Analysis

```bash
# The git hook performs these operations:
# 1. Analyzes git diff to identify changed files
# 2. Sends file contents to OpenRouter API (Claude Sonnet 4)
# 3. Updates descriptions in SQLite database
# 4. Logs all activity to ~/.mcp-code-index/githook.log

# Check the logs
tail -f ~/.mcp-code-index/githook.log
```

## 📋 Integration Examples

### CI/CD Pipeline Integration

#### GitHub Actions
```yaml
# .github/workflows/update-descriptions.yml
name: Update Code Descriptions
on:
  push:
    branches: [main, develop]

jobs:
  update-descriptions:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for analysis
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install MCP Code Indexer
        run: pip install mcp-code-indexer
      
      - name: Update descriptions
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          # Analyze commits from this push
          mcp-code-indexer --githook ${{ github.event.before }} ${{ github.sha }}
```

#### GitLab CI
```yaml
# .gitlab-ci.yml
update-descriptions:
  stage: analysis
  image: python:3.11-slim
  before_script:
    - pip install mcp-code-indexer
  script:
    - mcp-code-indexer --githook $CI_COMMIT_BEFORE_SHA $CI_COMMIT_SHA
  variables:
    OPENROUTER_API_KEY: $OPENROUTER_API_KEY
  only:
    - main
    - develop
```

### Development Team Setup

#### Team Configuration Script
```bash
#!/bin/bash
# setup-team-hooks.sh - Run this in each repository

echo "🔧 Setting up MCP Code Indexer for team..."

# Check prerequisites
if ! command -v mcp-code-indexer &> /dev/null; then
    echo "❌ Please install mcp-code-indexer first: pip install mcp-code-indexer"
    exit 1
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Please set OPENROUTER_API_KEY environment variable"
    exit 1
fi

# Install post-rewrite hook (handles rebases)
cp examples/post-rewrite-hook .git/hooks/post-rewrite
chmod +x .git/hooks/post-rewrite

# Install post-commit hook (basic commits)
cp examples/post-commit-hook .git/hooks/post-commit
chmod +x .git/hooks/post-commit

# Test installation
echo "🧪 Testing git hook integration..."
mcp-code-indexer --githook > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Git hooks installed successfully!"
    echo "📝 File descriptions will be updated automatically on commits"
    echo "📊 Check logs: tail -f ~/.mcp-code-index/githook.log"
else
    echo "❌ Installation test failed. Check logs for details."
    exit 1
fi
```

## 🔧 Advanced Configuration Examples

### Custom Hook with Error Handling
```bash
#!/bin/bash
# examples/advanced-post-commit-hook

set -euo pipefail

# Configuration
HOOK_LOG="$HOME/.mcp-code-index/githook.log"
MAX_RETRIES=3
RETRY_DELAY=5

# Ensure log directory exists
mkdir -p "$(dirname "$HOOK_LOG")"

# Function: Log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$HOOK_LOG"
}

# Function: Run with retries
run_with_retries() {
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        log_message "Attempt $attempt: Running MCP Code Indexer"
        
        if mcp-code-indexer --githook >> "$HOOK_LOG" 2>&1; then
            log_message "SUCCESS: Analysis completed on attempt $attempt"
            return 0
        else
            log_message "FAILED: Attempt $attempt failed (exit code: $?)"
            
            if [ $attempt -lt $MAX_RETRIES ]; then
                log_message "Retrying in ${RETRY_DELAY} seconds..."
                sleep $RETRY_DELAY
            fi
        fi
        
        ((attempt++))
    done
    
    log_message "ERROR: All $MAX_RETRIES attempts failed"
    return 1
}

# Main execution
cd "$(git rev-parse --show-toplevel)"
log_message "Starting git hook analysis (commit: $(git rev-parse HEAD))"

# Run in background to avoid blocking git operations
{
    if run_with_retries; then
        log_message "Git hook completed successfully"
    else
        log_message "Git hook failed after all retries"
    fi
} &

# Detach background process
disown
exit 0
```

### Environment-Specific Configuration
```bash
# examples/environment-configs/development.env
export OPENROUTER_API_KEY="your-dev-key"
export MCP_LOG_LEVEL="DEBUG"
export MCP_DB_PATH="$HOME/.mcp-code-index/dev-tracker.db"
export MCP_CACHE_DIR="$HOME/.mcp-code-index/dev-cache"

# examples/environment-configs/staging.env  
export OPENROUTER_API_KEY="your-staging-key"
export MCP_LOG_LEVEL="INFO"
export MCP_DB_PATH="/var/lib/mcp/staging-tracker.db"
export MCP_CACHE_DIR="/var/cache/mcp/staging"

# examples/environment-configs/production.env
export OPENROUTER_API_KEY="your-prod-key"
export MCP_LOG_LEVEL="WARNING"
export MCP_DB_PATH="/var/lib/mcp/prod-tracker.db"
export MCP_CACHE_DIR="/var/cache/mcp/production"
export MCP_DB_POOL_SIZE="10"
export MCP_DB_TIMEOUT="45.0"
```

## 🚨 Troubleshooting Guide

### Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Missing API Key** | `OPENROUTER_API_KEY environment variable is required` | Add to shell profile: `echo 'export OPENROUTER_API_KEY="your-key"' >> ~/.bashrc` |
| **Permission Denied** | `Permission denied: .git/hooks/post-commit` | Make executable: `chmod +x .git/hooks/post-commit` |
| **Network Errors** | `Connection timeout` or `Rate limited` | Check internet connection; upgrade OpenRouter plan if needed |
| **Large Diff Skipped** | `Skipping - diff too large` | Normal for massive changes; run manual analysis if needed |

### Diagnostic Commands

```bash
# Check installation
mcp-code-indexer --version

# Verify API key
echo $OPENROUTER_API_KEY | cut -c1-8  # Shows first 8 chars

# Test basic functionality
mcp-code-indexer --runcommand '{"method": "tools/call", "params": {"name": "check_database_health", "arguments": {}}}'

# Check recent logs
tail -20 ~/.mcp-code-index/githook.log

# Validate git repository
git status && git log --oneline -5
```

## 🎯 Next Steps

### For New Users
1. **[Install MCP Code Indexer](../README.md#-quick-start)** - Get the basic server running
2. **[Set up git hooks](#quick-setup-2-minutes)** - Automate your workflow  
3. **[Explore the API](../docs/api-reference.md)** - Learn about available tools

### For Teams
1. **[Review the Git Hook Setup Guide](../docs/git-hook-setup.md)** - Comprehensive team setup
2. **[Configure production deployment](../docs/configuration.md)** - Optimize for your environment
3. **[Set up monitoring](../docs/monitoring.md)** - Track performance and issues

### For Advanced Users
1. **[Performance tuning](../docs/performance-tuning.md)** - Optimize for high-concurrency
2. **[Database resilience](../docs/database-resilience.md)** - Advanced error handling
3. **[Architecture deep dive](../docs/architecture.md)** - Understand the system design

---

**💡 Need help?** 
- **Quick questions**: Check the [API Reference](../docs/api-reference.md)
- **Setup issues**: See the [Git Hook Setup Guide](../docs/git-hook-setup.md) 
- **Performance problems**: Review the [Performance Tuning Guide](../docs/performance-tuning.md)
- **Bug reports**: [Open an issue on GitHub](https://github.com/fluffypony/mcp-code-indexer/issues)

**🚀 Ready to automate your documentation?** Your next commit will automatically update file descriptions!
