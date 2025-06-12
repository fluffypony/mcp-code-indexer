# Git Hook Examples

This directory contains example git hooks for automated MCP Code Indexer integration.

## Available Hooks

### `post-commit-hook`
Runs after every commit to analyze the latest changes.

**Installation:**
```bash
cp examples/post-commit-hook .git/hooks/post-commit
chmod +x .git/hooks/post-commit
```

### `post-rewrite-hook` (Recommended)
Handles rebases, amends, and other commit-rewriting operations efficiently.

**Installation:**
```bash
cp examples/post-rewrite-hook .git/hooks/post-rewrite
chmod +x .git/hooks/post-rewrite
```

## Requirements

- `mcp-code-indexer` installed and available in PATH
- `OPENROUTER_API_KEY` environment variable set
- Git repository with staged/committed changes

## Usage

Once installed, the hooks run automatically:

- **post-commit**: Triggers on `git commit`
- **post-rewrite**: Triggers on `git rebase`, `git commit --amend`, etc.

All hooks run in the background and log to `~/.mcp-code-index/githook.log`.

## Manual Testing

Test the git hook functionality manually:

```bash
# Test current staged changes
mcp-code-indexer --githook

# Test specific commit
mcp-code-indexer --githook abc123def

# Test commit range (useful for rebases)
mcp-code-indexer --githook abc123 def456
```

## Troubleshooting

Check the log file for any issues:
```bash
tail -f ~/.mcp-code-index/githook.log
```

Common issues:
- Missing `OPENROUTER_API_KEY` environment variable
- Network connectivity problems
- Invalid commit hashes
- Repository not properly initialized

For detailed setup instructions, see [docs/git-hook-setup.md](../docs/git-hook-setup.md).
