#!/bin/bash
# MCP Code Indexer Environment Setup
#
# This script helps set up the environment for git hook integration
# Run: source setup-env.sh

echo "Setting up MCP Code Indexer environment..."

# Check if OpenRouter API key is already set
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "✓ OPENROUTER_API_KEY already set"
else
    echo "⚠ OPENROUTER_API_KEY not found in environment"
    echo "Please obtain an API key from https://openrouter.ai/"
    echo "Then add to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    echo "export OPENROUTER_API_KEY=\"sk-or-v1-your-api-key-here\""
    echo ""
fi

# Check if mcp-code-indexer is installed
if command -v mcp-code-indexer >/dev/null 2>&1; then
    echo "✓ mcp-code-indexer is installed"
    mcp-code-indexer --version
else
    echo "⚠ mcp-code-indexer not found in PATH"
    echo "Install with: poetry install"
    echo ""
fi

# Check if we're in a git repository
if git rev-parse --git-dir >/dev/null 2>&1; then
    echo "✓ Git repository detected"
    echo "Repository: $(basename "$(git rev-parse --show-toplevel)")"
    echo "Current branch: $(git rev-parse --abbrev-ref HEAD)"
else
    echo "⚠ Not in a git repository"
    echo "Navigate to your project directory first"
    echo ""
fi

# Check for required Python packages
echo "Checking Python dependencies..."

check_package() {
    if python -c "import $1" 2>/dev/null; then
        echo "✓ $1 is available"
    else
        echo "⚠ $1 not found - install with: poetry add $2"
    fi
}

check_package "aiohttp" "aiohttp>=3.8.0"
check_package "tenacity" "tenacity>=8.0.0"
check_package "tiktoken" "tiktoken==0.7.0"

# Create MCP directories
MCP_DIR="$HOME/.mcp-code-index"
if [ ! -d "$MCP_DIR" ]; then
    mkdir -p "$MCP_DIR/cache"
    echo "✓ Created MCP directory: $MCP_DIR"
else
    echo "✓ MCP directory exists: $MCP_DIR"
fi

# Suggest next steps
echo ""
echo "Next steps:"
echo "1. Set OPENROUTER_API_KEY if not already set"
echo "2. Install git hooks: cp examples/git-hooks/post-commit .git/hooks/"
echo "3. Make hooks executable: chmod +x .git/hooks/post-commit"
echo "4. Test with: mcp-code-indexer --githook"
echo ""
echo "For detailed setup instructions, see: docs/git-hook-setup.md"
