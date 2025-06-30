# Q&A Command Interface 🤖

---
**Last Updated:** 2025-06-30  
**Verified Against:** src/mcp_code_indexer/ask_handler.py, src/mcp_code_indexer/deepask_handler.py, main.py  
**Test Sources:** Manual verification of CLI interface and handlers  
**Implementation:** OpenRouter API integration with Claude for intelligent codebase analysis  
---

AI-powered question-answering interface for intelligent codebase analysis. Ask natural language questions about your projects and get detailed answers using Claude's advanced code understanding capabilities.

**🎯 New to Q&A mode?** Start with the [Quick Start Guide](../README.md#-quick-start) to configure OpenRouter API access first.

## Quick Reference

| Command | Purpose | Processing | Context |
|---------|---------|------------|---------|
| [`--ask`](#simple-qa-mode) | Simple questions | Single-stage Claude | Project overview only |
| [`--deepask`](#enhanced-qa-mode) | Complex analysis | Two-stage Claude | File search + context |

🔑 **Requires:** `OPENROUTER_API_KEY` environment variable  
🌐 **[OpenRouter Setup →](https://openrouter.ai/)**

## Table of Contents

- [Getting Started](#getting-started)
- [Simple Q&A Mode](#simple-qa-mode)
- [Enhanced Q&A Mode](#enhanced-qa-mode)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Best Practices](#best-practices)
- [Error Handling](#error-handling)
- [Cost Optimization](#cost-optimization)

## Getting Started

### Prerequisites

1. **OpenRouter API Key**: Sign up at [openrouter.ai](https://openrouter.ai/) and get your API key
2. **Project Analysis**: Your project should have file descriptions and overview (use MCP tools first)
3. **Environment Setup**: Set the required environment variable

### Environment Configuration

```bash
# Required: OpenRouter API key
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Optional: Model selection (defaults to Claude)
export MCP_CLAUDE_MODEL="anthropic/claude-3.5-sonnet"

# Optional: Token limit (default: 180000)
export MCP_CLAUDE_TOKEN_LIMIT="180000"
```

### Basic Usage

```bash
# Simple question about project architecture
mcp-code-indexer --ask "What does this project do?" my-project

# Detailed analysis with file search
mcp-code-indexer --deepask "How is authentication implemented?" my-project

# JSON output for programmatic use
mcp-code-indexer --ask "List the main components" my-project --json
```

## Simple Q&A Mode

The `--ask` command provides direct question-answering using the project overview. Best for general architecture questions and high-level understanding.

### Command Syntax

```bash
mcp-code-indexer --ask "QUESTION" PROJECT_NAME [OPTIONS]
```

### How It Works

1. **Context Loading**: Retrieves the project overview from your database
2. **Prompt Building**: Combines overview with your question in a focused prompt
3. **Claude Processing**: Single API call to Claude for analysis
4. **Response**: Returns formatted answer

### Use Cases

- **Architecture Overview**: "What is the overall architecture of this project?"
- **Technology Stack**: "What technologies and frameworks are used?"
- **Component Purpose**: "What are the main components and their roles?"
- **Design Patterns**: "What design patterns are implemented?"
- **High-Level Flow**: "How does data flow through the system?"

### Examples

#### Basic Architecture Question

```bash
mcp-code-indexer --ask "What does this project do?" web-app
```

**Example Output:**
```
This project is a modern web application built with React and Node.js that provides 
user authentication, data processing, and third-party integrations. 

Key components:
- Frontend: React application with TypeScript
- Backend: Express.js API server with JWT authentication
- Database: PostgreSQL with Prisma ORM
- Caching: Redis for session management
- Deployment: Docker containerization

The system follows a microservice architecture pattern with clear separation between 
the authentication service, data processing pipeline, and external API integrations.
```

#### Technology Stack Analysis

```bash
mcp-code-indexer --ask "What technologies are used in this project?" api-service
```

#### JSON Output

```bash
mcp-code-indexer --ask "List the main directories and their purposes" my-app --json
```

**Example JSON Output:**
```json
{
  "answer": "The main directories serve these purposes:\n\n- `/src`: Contains all source code...",
  "metadata": {
    "project_name": "my-app",
    "overview_tokens": 2847,
    "question_tokens": 12,
    "total_tokens": 2859,
    "response_time_seconds": 3.45,
    "timestamp": "2024-01-15T14:30:00Z"
  }
}
```

### Performance Characteristics

- **Speed**: Fast (typically 2-5 seconds)
- **Token Usage**: Low (overview + question only)
- **Context**: Project overview only
- **Best For**: General questions, architecture understanding

## Enhanced Q&A Mode

The `--deepask` command provides comprehensive analysis using two-stage processing with file search. Best for detailed implementation questions and specific feature analysis.

### Command Syntax

```bash
mcp-code-indexer --deepask "QUESTION" PROJECT_NAME [OPTIONS]
```

### How It Works

1. **Stage 1 - Analysis**: Claude extracts search terms and compresses the overview
2. **File Search**: Searches file descriptions using extracted terms
3. **Stage 2 - Enhanced Answer**: Claude provides detailed answer with file context
4. **Response**: Returns comprehensive analysis with specific file references

### Use Cases

- **Implementation Details**: "How is user authentication implemented?"
- **Feature Analysis**: "How does the payment processing work?"
- **Security Review**: "What security measures are in place?"
- **Error Handling**: "How are errors handled throughout the system?"
- **Performance**: "What performance optimizations are implemented?"
- **Testing Strategy**: "How is the codebase tested?"

### Examples

#### Authentication Implementation

```bash
mcp-code-indexer --deepask "How is user authentication implemented?" web-app
```

**Example Output:**
```
User authentication is implemented using a comprehensive JWT-based system with multiple layers:

## Authentication Flow
1. **Login Process**: Users authenticate via POST /api/auth/login endpoint
2. **Token Generation**: JWT tokens created with user ID and role claims
3. **Token Validation**: Middleware validates tokens on protected routes
4. **Session Management**: Redis stores refresh tokens for session persistence

## Key Components

**Authentication Middleware** (`src/middleware/auth.ts`):
- Validates JWT tokens on incoming requests
- Extracts user context and role information
- Handles token expiration and refresh logic

**User Service** (`src/services/userService.ts`):
- Manages user CRUD operations
- Handles password hashing with bcrypt
- Integrates with OAuth providers (Google, GitHub)

**Auth Controllers** (`src/controllers/authController.ts`):
- Login/logout endpoint implementations
- Password reset functionality
- OAuth callback handling

## Security Features
- Password hashing using bcrypt with salt rounds
- JWT tokens with configurable expiration
- Rate limiting on authentication endpoints
- CORS protection for cross-origin requests
- Input validation and sanitization

## Configuration
Authentication settings are managed in `src/config/auth.js` with environment-specific 
configurations for token secrets, expiration times, and OAuth credentials.
```

#### Feature Deep Dive

```bash
mcp-code-indexer --deepask "How does the file upload system work?" document-app
```

#### Security Analysis

```bash
mcp-code-indexer --deepask "What security measures are implemented?" api-service
```

#### Performance Review

```bash
mcp-code-indexer --deepask "What performance optimizations are in place?" large-app --json
```

### Performance Characteristics

- **Speed**: Slower (typically 10-30 seconds)
- **Token Usage**: Higher (two Claude calls + file context)
- **Context**: Project overview + relevant file descriptions
- **Best For**: Detailed analysis, implementation questions

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key for Claude access | None | Yes |
| `MCP_CLAUDE_MODEL` | Claude model to use | `anthropic/claude-3.5-sonnet` | No |
| `MCP_CLAUDE_TOKEN_LIMIT` | Maximum tokens per request | `180000` | No |

### Command Options

| Option | Description | Applies To |
|--------|-------------|------------|
| `--json` | Output response in JSON format | Both |
| `--log-level` | Set logging verbosity | Both |
| `--db-path` | Custom database path | Both |
| `--cache-dir` | Custom cache directory | Both |

### Model Selection

```bash
# Use specific Claude model
export MCP_CLAUDE_MODEL="anthropic/claude-3-opus"
mcp-code-indexer --deepask "Complex question" my-project

# Use different model for cost optimization
export MCP_CLAUDE_MODEL="anthropic/claude-3-haiku"
mcp-code-indexer --ask "Simple question" my-project
```

### Database Configuration

```bash
# Custom database location
mcp-code-indexer --ask "Question" my-project --db-path "/custom/path/tracker.db"

# Custom cache directory
mcp-code-indexer --deepask "Question" my-project --cache-dir "/tmp/mcp-cache"
```

## Output Formats

### Human-Readable Format (Default)

Clean, formatted text output optimized for terminal reading:

```bash
mcp-code-indexer --ask "What is this project?" my-app
```

```
This project is a modern e-commerce platform built with React and Node.js...

## Key Features
- User authentication and authorization
- Product catalog management
- Shopping cart and checkout process
- Payment processing integration
...
```

### JSON Format

Structured output for programmatic processing:

```bash
mcp-code-indexer --ask "What is this project?" my-app --json
```

```json
{
  "answer": "This project is a modern e-commerce platform...",
  "metadata": {
    "project_name": "my-app",
    "question_type": "ask",
    "overview_tokens": 2847,
    "question_tokens": 8,
    "total_tokens": 2855,
    "response_time_seconds": 3.21,
    "timestamp": "2024-01-15T14:30:00Z",
    "model_used": "anthropic/claude-3.5-sonnet"
  }
}
```

#### Enhanced JSON (DeepAsk)

```json
{
  "answer": "User authentication is implemented using...",
  "metadata": {
    "project_name": "web-app",
    "question_type": "deepask",
    "stage1": {
      "search_terms": ["authentication", "jwt", "middleware", "security"],
      "overview_compression": "Reduced overview from 2847 to 1200 tokens",
      "tokens_used": 1250
    },
    "stage2": {
      "files_searched": 15,
      "relevant_files": 6,
      "tokens_used": 3400
    },
    "total_tokens": 4650,
    "response_time_seconds": 18.7,
    "timestamp": "2024-01-15T14:30:00Z",
    "model_used": "anthropic/claude-3.5-sonnet"
  },
  "relevant_files": [
    {
      "path": "src/middleware/auth.ts",
      "relevance_score": 0.95,
      "description": "JWT authentication middleware with role-based access control"
    },
    {
      "path": "src/services/userService.ts",
      "relevance_score": 0.87,
      "description": "User service with authentication and session management"
    }
  ]
}
```

## Best Practices

### Question Types

#### ✅ Good Questions for --ask

- "What is the overall architecture?"
- "What technologies are used?"
- "What are the main components?"
- "How is the project structured?"
- "What design patterns are implemented?"

#### ✅ Good Questions for --deepask

- "How is user authentication implemented?"
- "How does the payment processing work?"
- "What security measures are in place?"
- "How is error handling implemented?"
- "How is the data validation performed?"

#### ❌ Questions to Avoid

- Overly specific line-by-line code questions
- Questions about exact variable names or implementation details
- Questions that require real-time data or external system information
- Questions about undocumented or missing features

### Performance Optimization

#### Use --ask When:
- Getting general project understanding
- Quick architecture overview
- Technology stack information
- High-level component understanding

#### Use --deepask When:
- Analyzing specific implementations
- Understanding complex workflows
- Security or performance reviews
- Detailed feature analysis

### Question Formulation Tips

1. **Be Specific**: "How is user authentication handled?" vs "Tell me about auth"
2. **Ask About Implementation**: "How does X work?" vs "What is X?"
3. **Focus on Architecture**: Ask about patterns, flows, and relationships
4. **Use Domain Terms**: Use technical terms relevant to your project

## Error Handling

### Common Errors

#### Missing API Key
```bash
$ mcp-code-indexer --ask "Question" my-project
Error: OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.
```

**Solution:**
```bash
export OPENROUTER_API_KEY="your-api-key"
```

#### Project Not Found
```bash
$ mcp-code-indexer --ask "Question" nonexistent-project
Error: No existing project found with name: nonexistent-project
```

**Solution:** Verify project name with `--getprojects` command

#### Token Limit Exceeded
```bash
$ mcp-code-indexer --ask "Very long question..." large-project
Error: Question and project context exceed token limit of 180000
```

**Solutions:**
- Use a more specific question
- Use `--deepask` for better context management
- Increase token limit: `export MCP_CLAUDE_TOKEN_LIMIT="200000"`

#### No Project Overview
```bash
$ mcp-code-indexer --ask "Question" my-project
Warning: No project overview found for my-project
Answer: Unable to provide detailed analysis without project overview.
```

**Solution:** Create project overview using MCP tools first

### Debugging

#### Enable Debug Logging

```bash
mcp-code-indexer --ask "Question" my-project --log-level DEBUG
```

#### Check Logs

```bash
# Ask command logs
tail -f ~/.mcp-code-index/cache/ask.log

# DeepAsk command logs  
tail -f ~/.mcp-code-index/cache/deepask.log
```

#### Verbose JSON Output

```bash
mcp-code-indexer --deepask "Question" my-project --json | jq .metadata
```

## Cost Optimization

### Understanding Costs

OpenRouter pricing is based on tokens consumed. Here's how to optimize:

#### Token Usage Patterns

| Command | Typical Tokens | Cost Factor |
|---------|----------------|-------------|
| `--ask` | 1,000 - 5,000 | Low |
| `--deepask` | 5,000 - 20,000 | Medium-High |

#### Cost-Effective Strategies

1. **Start with --ask**: Use for general questions first
2. **Specific --deepask**: Ask focused questions rather than broad ones
3. **Model Selection**: Use cheaper models for simple questions
4. **Batch Questions**: Ask multiple related questions in one session

### Model Recommendations

```bash
# Cost-effective for simple questions
export MCP_CLAUDE_MODEL="anthropic/claude-3-haiku"

# Balanced performance and cost
export MCP_CLAUDE_MODEL="anthropic/claude-3.5-sonnet"

# Maximum capability (higher cost)
export MCP_CLAUDE_MODEL="anthropic/claude-3-opus"
```

### Monitoring Usage

```bash
# Track token usage in JSON output
mcp-code-indexer --ask "Question" my-project --json | jq .metadata.total_tokens

# Monitor costs over time
grep "total_tokens" ~/.mcp-code-index/cache/ask.log
```

---

**Next Steps**: Check out the [CLI Reference](cli-reference.md) for complete command documentation, or review the [Git Hook Setup](git-hook-setup.md) for automated Q&A workflows! 🚀
