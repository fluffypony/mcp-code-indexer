# Git Hook Integration Setup 🔗

Transform your development workflow with automated code analysis! The MCP Code Indexer can automatically update file descriptions every time you commit, keeping your codebase documentation perfectly synchronized with your code.

**🎯 Perfect for**: Development teams wanting effortless documentation maintenance

## How It Works

The git hook integration provides **zero-effort documentation**:

🔍 **Smart Analysis**: Automatically analyzes git diffs to identify changed files  
🤖 **AI-Powered**: Uses OpenRouter API with Anthropic's Claude Sonnet 4 model  
⚡ **Efficient**: Updates descriptions only for files that actually changed  
📋 **Overview Updates**: Refreshes project overview when structural changes occur  
🔄 **Automated**: Runs automatically on git events (post-commit, post-merge, etc.)

## Prerequisites

1. **OpenRouter API Key**: Required for AI-powered analysis
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-your-api-key-here"
   ```

### Step 2: Install Dependencies

The git hook feature requires additional packages:
```bash
pip install aiohttp>=3.8.0 tenacity>=8.0.0
```

**Already have MCP Code Indexer installed?** These dependencies are included automatically.

## 🚀 Quick Setup Guide

### Test It First

Before setting up automatic hooks, test the functionality manually:

```bash
# Navigate to your git project
cd /path/to/your/project

# Test git hook analysis on recent changes
mcp-code-indexer --githook
```

**What you'll see**: The tool analyzes recent git changes and updates descriptions for modified files.

### Automatic Setup

#### Post-Commit Hook

Create `.git/hooks/post-commit`:

```bash
#!/bin/bash
# MCP Code Indexer - Auto-update descriptions after commit

# Change to repository root
cd "$(git rev-parse --show-toplevel)"

# Run MCP code indexer in git hook mode
# Redirect output to avoid interfering with git operations
mcp-code-indexer --githook >> ~/.mcp-code-index/githook.log 2>&1

# Exit successfully regardless of indexer result
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/post-commit
```

#### Post-Merge Hook

Create `.git/hooks/post-merge`:

```bash
#!/bin/bash
# MCP Code Indexer - Auto-update descriptions after merge

cd "$(git rev-parse --show-toplevel)"
mcp-code-indexer --githook >> ~/.mcp-code-index/githook.log 2>&1
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/post-merge
```

## 🔧 Configuration Options

### Environment Variables

**Required**:
- `OPENROUTER_API_KEY`: Your OpenRouter API key

**Optional**:
- `MCP_GITHOOK_MODEL`: Override the AI model (default: "anthropic/claude-sonnet-4")

### Advanced Options

The git hook mode supports all standard MCP Code Indexer options for fine-tuning:

```bash
# Custom database and cache locations
mcp-code-indexer --githook \
  --db-path ~/.mcp-code-index/tracker.db \
  --cache-dir ~/.mcp-code-index/cache \
  --log-level INFO
```

**💡 Most users can use the defaults** - only customize if you have specific requirements.

## Behavior

### What Gets Updated

1. **File Descriptions**: Updated for files that appear in the git diff
2. **Project Overview**: Updated when significant architectural changes are detected
3. **Database**: All changes are persisted to the SQLite database

### Change Detection

The system analyzes:
- Files added, modified, or deleted in the git diff
- Content changes and their significance
- Structural changes that affect project architecture

### Safety Features

- **Timeout Protection**: 30-second timeout prevents hanging
- **Diff Size Limits**: Skips processing for very large diffs (>100KB)
- **Error Isolation**: Git operations continue even if indexing fails
- **Rate Limiting**: Automatic retries with exponential backoff

## 🔧 Troubleshooting

### Common Issues & Solutions

**🚨 Missing API Key**
```
Error: OPENROUTER_API_KEY environment variable is required
```
**Solution**: Set the environment variable in your shell profile (`~/.bashrc`, `~/.zshrc`)

**🚨 Git Command Failures**
```
Error: Git command not found
```
**Solution**: Ensure git is installed and in your PATH. Try `git --version` to test.

**ℹ️ Large Diff Skipped**
```
Info: Skipping git hook update - diff too large or empty
```
**This is normal** for large changes (>100KB diffs). Run manual analysis if needed with individual file updates.

**⏱️ Rate Limiting**
```
Warning: Rate limited. Retry after 60s
```
**The system handles this automatically** - it will retry with exponential backoff. Consider upgrading your OpenRouter plan for higher limits.

### Logs

Check git hook logs:
```bash
tail -f ~/.mcp-code-index/githook.log
```

Check MCP server logs:
```bash
tail -f ~/.mcp-code-index/cache/server.log
```

### Manual Override

If automatic analysis fails, run manual updates:
```bash
# Update specific file description
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "update_file_description",
    "arguments": {
      "projectName": "your-project",
      "folderPath": "/path/to/project",
      "branch": "main",
      "filePath": "src/example.py",
      "description": "Updated description"
    }
  }
}'
```

## Performance Considerations

- **API Costs**: Each git hook call uses OpenRouter API tokens
- **Latency**: Adds 2-5 seconds to git operations
- **Frequency**: Consider hook placement (post-commit vs post-push)

## Security

- API keys are never logged or stored in git
- All communication uses HTTPS
- Diff content is sent to OpenRouter for analysis (consider for sensitive projects)

## Integration with CI/CD

For automated environments, use environment variables:

```yaml
# GitHub Actions example
env:
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}

steps:
  - uses: actions/checkout@v4
  - name: Update code descriptions
    run: mcp-code-indexer --githook
```

## Best Practices

1. **Selective Usage**: Enable hooks only in development repositories
2. **Log Monitoring**: Regularly check hook logs for errors
3. **API Management**: Monitor OpenRouter usage and costs
4. **Backup**: Regular database backups recommended
5. **Team Coordination**: Ensure all team members have API access

## Migration

To migrate from manual description updates:

1. Set up git hooks as described above
2. Run initial full scan: `mcp-code-indexer --runcommand '{"method": "tools/call", "params": {"name": "check_codebase_size", "arguments": {...}}}'`
3. Verify descriptions are working: `mcp-code-indexer --getprojects`
4. Enable hooks for the team

The git hook integration provides seamless, automated maintenance of your codebase documentation as your project evolves.
