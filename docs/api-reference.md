# MCP Tools API Reference 📖

---
**Last Updated:** 2025-01-15
**Verified Against:** src/mcp_code_indexer/server/mcp_server.py
**Test Sources:** tests/integration/test_mcp_tools.py, tests/unit/test_query_preprocessor.py
**Implementation:** All 13 tools verified against actual server code
---

Complete reference for all 13 MCP tools provided by the Code Indexer server. Whether you're building AI agents or integrating MCP tools directly, this guide shows you exactly how to use each tool effectively.

**🎯 New to MCP Code Indexer?** Start with the [Quick Start Guide](../README.md#-quick-start) to set up your server first.

## Quick Reference

| Tool | Purpose | Required Parameters |
|------|---------|-------------------|
| [`check_codebase_size`](#check_codebase_size) ⭐ | Navigation recommendations | `projectName`, `folderPath` |
| [`search_descriptions`](#search_descriptions) | Find files by functionality | `projectName`, `folderPath`, `query` |
| [`get_file_description`](#get_file_description) | Retrieve file summary | `projectName`, `folderPath`, `filePath` |
| [`update_file_description`](#update_file_description) | Store file analysis | `projectName`, `folderPath`, `filePath`, `description` |
| [`get_codebase_overview`](#get_codebase_overview) | Project architecture | `projectName`, `folderPath` |
| [`find_missing_descriptions`](#find_missing_descriptions) | Scan for undocumented files | `projectName`, `folderPath` |
| [`get_all_descriptions`](#get_all_descriptions) | Complete project structure | `projectName`, `folderPath` |
| [`get_word_frequency`](#get_word_frequency) | Technical vocabulary | `projectName`, `folderPath` |
| [`update_codebase_overview`](#update_codebase_overview) | Create project docs | `projectName`, `folderPath`, `overview` |
| [`search_codebase_overview`](#search_codebase_overview) | Search overviews | `projectName`, `folderPath`, `searchWord` |
| [`check_database_health`](#check_database_health) | System monitoring | None |
| [`enabled_vector_mode`](#enabled_vector_mode) | Configure vector search | `projectName`, `folderPath`, `enabled` |
| [`find_similar_code`](#find_similar_code) | Find similar code patterns | `projectName`, `folderPath`, code/file input |

⭐ **Start here** for new projects
📖 **[See Examples →](../examples/)**

## Table of Contents

- [Quick Reference](#quick-reference)
- [Core Operations](#core-operations)
  - [get_file_description](#get_file_description)
  - [update_file_description](#update_file_description)
  - [check_codebase_size](#check_codebase_size)
- [Batch Operations](#batch-operations)
  - [find_missing_descriptions](#find_missing_descriptions)
- [Search & Discovery](#search--discovery)
  - [search_descriptions](#search_descriptions)
  - [get_all_descriptions](#get_all_descriptions)
  - [get_word_frequency](#get_word_frequency)
  - [get_codebase_overview](#get_codebase_overview)
  - [search_codebase_overview](#search_codebase_overview)
- [Advanced Features](#advanced-features)
  - [update_codebase_overview](#update_codebase_overview)
- [System Monitoring](#system-monitoring)
  - [check_database_health](#check_database_health)
- [Configuration Management](#configuration-management)
  - [enabled_vector_mode](#enabled_vector_mode)
  - [find_similar_code](#find_similar_code)
- [Common Parameters](#common-parameters)
- [Error Handling](#error-handling)

## Core Operations

### get_file_description

Retrieves the stored description for a specific file in a codebase. Use this to quickly understand what a file contains without reading its full contents.

#### Parameters

```typescript
interface GetFileDescriptionParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder on disk
  filePath: string;          // Relative path to the file from project root
}
```

#### Response

```typescript
interface GetFileDescriptionResponse {
  exists: boolean;           // Whether the file has a stored description
  description?: string;      // File description if it exists
  lastModified?: string;     // ISO timestamp of last update
  fileHash?: string;         // SHA-256 hash of file contents
  version?: number;          // Version number for optimistic concurrency
  message?: string;          // Error message if file not found
}
```

#### Example

```javascript
const result = await mcp.callTool("get_file_description", {
  projectName: "my-web-app",
  folderPath: "/home/user/projects/my-web-app",
  filePath: "src/components/UserProfile.tsx"
});

// Response:
{
  "exists": true,
  "description": "React component for displaying user profile information with edit capabilities",
  "lastModified": "2024-01-15T10:30:00Z",
  "fileHash": "abc123def456",
  "version": 1
}
```

---

### update_file_description

Creates or updates the description for a file. Use this after analyzing a file's contents to store a detailed summary.

#### Parameters

```typescript
interface UpdateFileDescriptionParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder on disk
  filePath: string;          // Relative path to the file from project root
  description: string;       // Detailed description of the file's contents
  fileHash?: string;         // SHA-256 hash of the file contents (optional)
}
```

#### Response

```typescript
interface UpdateFileDescriptionResponse {
  success: boolean;          // Whether the update succeeded
  message: string;           // Success/failure message
  filePath: string;          // Path that was updated
  lastModified: string;      // ISO timestamp of the update
}
```

#### Example

```javascript
const result = await mcp.callTool("update_file_description", {
  projectName: "my-web-app",
  folderPath: "/home/user/projects/my-web-app",
  filePath: "src/utils/apiClient.ts",
  description: "HTTP client utility with authentication, retry logic, and error handling for API calls",
  fileHash: "def456ghi789"
});

// Response:
{
  "success": true,
  "message": "Description updated for src/utils/apiClient.ts",
  "filePath": "src/utils/apiClient.ts",
  "lastModified": "2024-01-15T10:35:00Z"
}
```

---

### check_codebase_size

Checks the total token count of a codebase's file structure and descriptions. Returns whether the codebase is 'large' and recommends using search instead of the full overview.

#### Parameters

```typescript
interface CheckCodebaseSizeParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder
}
```

#### Response

```typescript
interface CheckCodebaseSizeResponse {
  totalTokens: number;       // Total token count for all descriptions
  isLarge: boolean;          // Whether codebase exceeds token limit
  recommendation: "use_search" | "use_overview";  // Recommended approach
  tokenLimit: number;        // Current configured token limit
  totalFiles: number;        // Number of files with descriptions
}
```

#### Example

```javascript
const result = await mcp.callTool("check_codebase_size", {
  projectName: "large-enterprise-app",
  folderPath: "/home/user/projects/enterprise-app"
});

// Response:
{
  "totalTokens": 45000,
  "isLarge": true,
  "recommendation": "use_search",
  "tokenLimit": 32000,
  "totalFiles": 284
}
```

💡 **Pro Tip**: Always call this first when working with a new codebase to determine the optimal navigation strategy. Large codebases (>32k tokens) work better with `search_descriptions`, while smaller ones can use `get_all_descriptions` for a complete overview.

---

## Batch Operations

### find_missing_descriptions

Scans the project folder to find files that don't have descriptions yet. Use update_file_description to add descriptions for individual files. Respects .gitignore and common ignore patterns.

#### Parameters

```typescript
interface FindMissingDescriptionsParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder
  limit?: number;             // Maximum number of missing files to return (optional)
  randomize?: boolean;        // Randomly shuffle files before applying limit (optional)
}
```

#### Response

```typescript
interface FindMissingDescriptionsResponse {
  missingFiles: string[];     // Array of relative file paths without descriptions
  totalMissing: number;       // Count of missing files
  limit: number;              // Applied limit (from request or default)
  randomized: boolean;        // Whether results were randomized
}
```

#### Example

```javascript
const result = await mcp.callTool("find_missing_descriptions", {
  projectName: "new-project",
  folderPath: "/home/user/projects/new-project",
  limit: 10,
  randomize: false
});

// Response:
{
  "missingFiles": [
    "src/components/NewFeature.tsx",
    "src/hooks/useLocalStorage.ts",
    "tests/integration/api.test.ts"
  ],
  "totalMissing": 3,
  "limit": 10,
  "randomized": false
}
```

---

## Search & Discovery

### search_descriptions

Searches through all file descriptions in a project to find files related to specific functionality using intelligent query preprocessing. Features include:

- **Multi-word search**: `"grpc proto"` finds files containing both terms regardless of order
- **Operator escaping**: FTS5 operators (`AND`, `OR`, `NOT`, `NEAR`) are treated as literal search terms
- **Whole word matching**: Prevents partial matches for more precise results
- **Case insensitive**: Works regardless of case in query or descriptions

Use this for large codebases instead of loading the entire structure. Returns files ranked by relevance using BM25 scoring.

#### Parameters

```typescript
interface SearchDescriptionsParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder
  query: string;             // Search query (e.g., 'authentication middleware', 'database models')
  maxResults?: number;       // Maximum number of results to return (default: 20)
}
```

#### Response

```typescript
interface SearchDescriptionsResponse {
  results: Array<{           // Search results ranked by relevance
    filePath: string;        // Relative path to the matching file
    description: string;     // File description
    relevanceScore: number;  // Search relevance score (higher = more relevant)
  }>;
  totalResults: number;      // Total number of results found
  query: string;             // The search query used
  maxResults: number;        // Maximum results limit applied
}
```

#### Example

```javascript
const result = await mcp.callTool("search_descriptions", {
  projectName: "large-app",
  folderPath: "/home/user/projects/large-app",
  query: "authentication middleware",
  maxResults: 10
});

// Response:
{
  "results": [
    {
      "filePath": "src/middleware/auth.ts",
      "description": "Express middleware for JWT authentication with role-based access control",
      "relevanceScore": 0.95
    },
    {
      "filePath": "src/utils/authHelpers.ts",
      "description": "Helper functions for authentication token validation and user session management",
      "relevanceScore": 0.78
    },
    {
      "filePath": "src/routes/protected.ts",
      "description": "Protected API routes that require authentication middleware",
      "relevanceScore": 0.65
    }
  ],
  "totalResults": 3,
  "query": "authentication middleware",
  "maxResults": 10
}
```

#### Enhanced Search Examples

**Multi-word search (order-agnostic):**
```javascript
// Both queries find the same results
await mcp.callTool("search_descriptions", {
  projectName: "api-service",
  folderPath: "/projects/api-service",
  query: "grpc proto"        // Finds files with both "grpc" AND "proto"
});

await mcp.callTool("search_descriptions", {
  projectName: "api-service",
  folderPath: "/projects/api-service",
  query: "proto grpc"        // Same results as above
});
```

**FTS5 operator escaping:**
```javascript
// Search for files containing literal "AND" as a term
await mcp.callTool("search_descriptions", {
  projectName: "error-handling",
  folderPath: "/projects/error-handling",
  query: "logging AND error"  // Finds files with all three: "logging", "AND", "error"
});
```

**Case insensitive matching:**
```javascript
// All variations return same results
const queries = ["HTTP client", "http CLIENT", "Http Client"];
// Each finds files containing both "http" and "client" regardless of case
```

🔍 **Search Tips**:
- **Use multiple words**: "grpc proto" finds files with both terms
- **Try different orders**: "api client" vs "client api" yield same results
- **Be descriptive**: "authentication logic" vs "auth"
- **Don't worry about operators**: "AND", "OR" are treated as literal search terms
- **Case doesn't matter**: "HTTP", "http", "Http" all work the same
- **Use technical terms**: "middleware", "controller", "utils"
- **Search by purpose**: "error handling", "data validation"

---

### get_all_descriptions

Returns the complete file and folder structure of a codebase with all descriptions. For large codebases (exceeding the configured token limit), this will recommend using search_descriptions instead.

#### Parameters

```typescript
interface GetAllDescriptionsParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder
}
```

#### Response

```typescript
interface GetAllDescriptionsResponse {
  projectName: string;       // Project name
  totalFiles: number;       // Total number of tracked files
  totalTokens: number;      // Total token count for all descriptions
  isLarge: boolean;         // Whether codebase exceeds token limit
  tokenLimit: number;       // Current token limit setting
  structure?: FolderNode;   // Hierarchical folder structure (if not large)
  recommendation?: string;  // Recommendation if codebase is large
  message?: string;         // Additional guidance message
}

interface FolderNode {
  name: string;             // Folder name
  path: string;             // Full path from project root
  files: FileNode[];        // Files in this folder
  folders: FolderNode[];    // Subfolders
}

interface FileNode {
  name: string;             // File name
  path: string;             // Full path from project root
  description: string;      // File description
}
```

#### Example (Small Codebase)

```javascript
const result = await mcp.callTool("get_all_descriptions", {
  projectName: "small-app",
  folderPath: "/home/user/projects/small-app"
});

// Response:
{
  "projectName": "small-app",
  "totalFiles": 8,
  "totalTokens": 1250,
  "isLarge": false,
  "tokenLimit": 32000,
  "structure": {
    "name": "",
    "path": "",
    "files": [
      {
        "name": "package.json",
        "path": "package.json",
        "description": "NPM package configuration with dependencies and scripts"
      }
    ],
    "folders": [
      {
        "name": "src",
        "path": "src",
        "files": [
          {
            "name": "index.ts",
            "path": "src/index.ts",
            "description": "Main application entry point with Express server setup"
          }
        ],
        "folders": []
      }
    ]
  }
}
```

#### Example (Large Codebase)

```javascript
// Response for large codebase:
{
  "projectName": "enterprise-app",
  "totalFiles": 500,
  "totalTokens": 45000,
  "isLarge": true,
  "tokenLimit": 32000,
  "recommendation": "use_search",
  "message": "Codebase has 45000 tokens (limit: 32000). Use search_descriptions instead for better performance."
}
```

---

### get_word_frequency

Analyzes all file descriptions to find the most frequently used technical terms. Filters out common English stop words and symbols, returning the top 200 meaningful terms. Useful for understanding the codebase's domain vocabulary and finding all functions/files related to specific concepts.

#### Parameters

```typescript
interface GetWordFrequencyParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder on disk
  limit?: number;             // Number of top terms to return (default: 200)
}
```

#### Response

```typescript
interface GetWordFrequencyResponse {
  wordFrequency: Array<{     // Array of words and their frequencies
    word: string;            // The technical term
    count: number;           // How many times it appears
    percentage: number;      // Percentage of total meaningful words
  }>;
  totalWords: number;        // Total number of meaningful words analyzed
  projectName: string;       // Project name
  limit: number;             // Applied limit
}
```

#### Example

```javascript
const result = await mcp.callTool("get_word_frequency", {
  projectName: "web-app",
  folderPath: "/home/user/projects/web-app",
  limit: 10
});

// Response:
{
  "wordFrequency": [
    { "word": "component", "count": 45, "percentage": 8.2 },
    { "word": "authentication", "count": 32, "percentage": 5.8 },
    { "word": "api", "count": 28, "percentage": 5.1 },
    { "word": "database", "count": 24, "percentage": 4.4 },
    { "word": "middleware", "count": 19, "percentage": 3.5 }
  ],
  "totalWords": 550,
  "projectName": "web-app",
  "limit": 10
}
```

💡 **Use cases**: Discover domain vocabulary, find related files by topic, understand system concepts, identify technical debt areas by frequency of error-related terms.

---

### get_codebase_overview

Returns a condensed, interpretive overview of the entire codebase. This is a single comprehensive narrative that captures the architecture, key components, relationships, and design patterns. Unlike get_all_descriptions which lists every file, this provides a holistic view suitable for understanding the codebase's structure and purpose. If no overview exists, returns empty string.

#### Parameters

```typescript
interface GetCodebaseOverviewParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder on disk
}
```

#### Response

```typescript
interface GetCodebaseOverviewResponse {
  overview: string;          // Condensed narrative overview of the codebase
  lastModified?: string;     // ISO timestamp of last update
  totalFiles: number;        // Number of files included in overview
  totalTokensInFullDescriptions: number; // Token count of all file descriptions
}
```

#### Example

```javascript
const result = await mcp.callTool("get_codebase_overview", {
  projectName: "microservice-api",
  folderPath: "/home/user/projects/microservice-api"
});

// Response:
{
  "overview": "## System Summary\nMicroservice API built with Node.js and Express providing user authentication, data processing, and third-party integrations.\n\n## Architecture\nRESTful API with JWT authentication, PostgreSQL database, Redis caching, and Docker containerization. Uses dependency injection pattern with service layer separation.\n\n## Key Components\n- Authentication service handles JWT tokens and user sessions\n- Data processing pipeline with async job queues\n- Third-party API integrations for payment processing",
  "lastModified": "2024-01-15T14:30:00Z",
  "totalFiles": 45,
  "totalTokensInFullDescriptions": 12500
}
```

💡 **When to use**: Perfect for getting a high-level understanding of large codebases without loading individual file descriptions. Provides the narrative context needed for AI agents to understand system architecture and design decisions.

---

### search_codebase_overview

Search for a single word in the codebase overview and return 2 sentences before and after where the word is found. Useful for quickly finding specific information in large overviews.

#### Parameters

```typescript
interface SearchCodebaseOverviewParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder on disk
  searchWord: string;         // Single word to search for in the overview
}
```

#### Response

```typescript
interface SearchCodebaseOverviewResponse {
  found: boolean;            // Whether the search word was found
  results: Array<{          // Array of context snippets around the search word
    context: string;         // 2 sentences before + match + 2 sentences after
    position: number;        // Character position in the overview
  }>;
  searchWord: string;        // The word that was searched for
  totalMatches: number;      // Total number of matches found
}
```

#### Example

```javascript
// SOURCE: tests/integration/test_mcp_tools.py:315-324
const result = await mcp.callTool("search_codebase_overview", {
  projectName: "large-project",
  folderPath: "/home/user/projects/large-project",
  searchWord: "authentication"
});

// Response:
{
  "found": true,
  "results": [
    {
      "context": "The application uses JWT tokens for session management. Authentication middleware validates tokens on protected routes. User credentials are stored securely using bcrypt hashing.",
      "position": 1250
    }
  ],
  "searchWord": "authentication",
  "totalMatches": 1
}
```

💡 **When to use**: Perfect for quickly finding specific topics in comprehensive codebase overviews without reading the entire document. Especially useful for large projects where the overview exceeds several thousand words.

---

## Advanced Features

### update_codebase_overview

Updates the condensed codebase overview. Create a comprehensive narrative that would help a new developer understand this codebase.

#### Parameters

```typescript
interface UpdateCodebaseOverviewParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder
  overview: string;          // Comprehensive narrative overview of the codebase
}
```

#### Response

```typescript
interface UpdateCodebaseOverviewResponse {
  success: boolean;          // Whether the update succeeded
  message: string;           // Success/failure message
  projectName: string;       // Project that was updated
  overviewLength: number;    // Length of the overview in characters
}
```

#### Example

```javascript
const result = await mcp.callTool("update_codebase_overview", {
  projectName: "my-app",
  folderPath: "/home/user/projects/my-app",
  overview: `## Architecture Overview
This is a modern web application built with React and Node.js...

## Core Components
- Frontend: React with TypeScript
- Backend: Express.js API server
- Database: PostgreSQL with Prisma ORM
...`
});

// Response:
{
  "success": true,
  "message": "Codebase overview updated successfully",
  "projectName": "my-app",
  "overviewLength": 2847
}
```

---

## System Monitoring

### check_database_health

🏥 **Real-time database health monitoring and diagnostics**

Monitor database performance, connection pool status, and system health in production environments. Essential for maintaining high-availability deployments and troubleshooting performance issues.

#### Parameters

```typescript
interface CheckDatabaseHealthParams {
  // No parameters required
}
```

#### Response

```typescript
interface CheckDatabaseHealthResponse {
  health_status: {
    overall_health: 'healthy' | 'degraded' | 'unhealthy';
    database: {
      pool_healthy: boolean;
      active_connections: number;
      total_connections: number;
      failed_connections: number;
      avg_response_time_ms: number;
      wal_size_mb: number;
    };
    performance: {
      current_throughput: number;
      target_throughput: number;
      p95_latency_ms: number;
      error_rate: number;
      operations_last_minute: number;
    };
    system: {
      memory_usage_mb: number;
      cpu_usage_percent: number;
      disk_usage_percent: number;
      uptime_seconds: number;
    };
  };
  recommendations: string[];
  last_check: string;  // ISO timestamp
}
```

#### Usage Examples

##### 👨‍💻 Basic Health Check

```python
# Check current system health
health_result = await mcp_client.call_tool("check_database_health", {})

if health_result["health_status"]["overall_health"] != "healthy":
    print("⚠️ System health issue detected!")
    for rec in health_result["recommendations"]:
        print(f"💡 {rec}")
```

##### 🔧 Production Monitoring

```python
# Automated health monitoring
async def monitor_health():
    while True:
        health = await mcp_client.call_tool("check_database_health", {})

        # Check critical metrics
        db_status = health["health_status"]["database"]
        if db_status["failed_connections"] > 2:
            send_alert("Database connection failures detected")

        perf_status = health["health_status"]["performance"]
        if perf_status["error_rate"] > 0.05:
            send_alert(f"High error rate: {perf_status['error_rate']:.2%}")

        await asyncio.sleep(30)  # Check every 30 seconds
```

##### 📊 Performance Dashboard

```python
# Gather metrics for dashboard
health_data = await mcp_client.call_tool("check_database_health", {})

metrics = {
    'throughput': health_data["health_status"]["performance"]["current_throughput"],
    'latency_p95': health_data["health_status"]["performance"]["p95_latency_ms"],
    'error_rate': health_data["health_status"]["performance"]["error_rate"],
    'pool_utilization': (
        health_data["health_status"]["database"]["active_connections"] /
        health_data["health_status"]["database"]["total_connections"]
    )
}

# Send to monitoring system
await send_metrics_to_grafana(metrics)
```

#### 🎯 Use Cases

- **Production Monitoring**: Continuous health checks in production deployments
- **Performance Debugging**: Identify bottlenecks and optimization opportunities
- **Capacity Planning**: Monitor resource utilization trends
- **Incident Response**: Quick diagnostics during performance issues
- **Load Testing**: Validate system behavior under stress

#### 🔍 Health Status Indicators

| Status | Database | Performance | System | Action Required |
|--------|----------|-------------|--------|-----------------|
| `healthy` | All connections working | <2% error rate | <80% resource usage | None |
| `degraded` | Some connection issues | 2-5% error rate | 80-90% resource usage | Monitor closely |
| `unhealthy` | Pool failures | >5% error rate | >90% resource usage | Immediate action |

#### 🚨 Common Issues & Recommendations

The tool provides intelligent recommendations based on current system state:

- **"Increase connection pool size"** - When pool utilization >90%
- **"Enable WAL mode for better concurrency"** - When lock contention detected
- **"Consider scaling resources"** - When system resources >85%
- **"Check for database corruption"** - When integrity issues found
- **"Review recent configuration changes"** - When performance regression detected

## Configuration Management

### enabled_vector_mode

Enables or disables vector mode for a project. Vector mode provides semantic search capabilities with embeddings for enhanced code navigation and discovery.

#### Parameters

```typescript
interface EnabledVectorModeParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder on disk
  enabled: boolean;           // Whether to enable (true) or disable (false) vector mode
}
```

#### Response

```typescript
interface EnabledVectorModeResponse {
  success: boolean;           // Whether the operation succeeded
  message: string;            // Success or error message
  project_id: string;         // Project identifier
  vector_mode: boolean | null; // Current vector mode status (null on error)
  error?: string;             // Error details if operation failed
}
```

#### Example

```javascript
// Enable vector mode for a project
const result = await mcp.callTool("enabled_vector_mode", {
  projectName: "my-web-app",
  folderPath: "/home/user/projects/my-web-app",
  enabled: true
});

// Response:
{
  "success": true,
  "message": "Vector mode enabled for project",
  "project_id": "my_web_app_abc123",
  "vector_mode": true
}

// Disable vector mode
const disableResult = await mcp.callTool("enabled_vector_mode", {
  projectName: "my-web-app", 
  folderPath: "/home/user/projects/my-web-app",
  enabled: false
});

// Response:
{
  "success": true,
  "message": "Vector mode disabled for project", 
  "project_id": "my_web_app_abc123",
  "vector_mode": false
}
```

#### Command Line Usage

```bash
# Enable vector mode
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "enabled_vector_mode",
    "arguments": {
      "projectName": "my-web-app",
      "folderPath": "/home/user/projects/my-web-app",
      "enabled": true
    }
  }
}'

# Disable vector mode  
mcp-code-indexer --runcommand '{
  "method": "tools/call",
  "params": {
    "name": "enabled_vector_mode", 
    "arguments": {
      "projectName": "my-web-app",
      "folderPath": "/home/user/projects/my-web-app",
      "enabled": false
    }
  }
}'
```

#### 🎯 Use Cases

- **Enable Semantic Search**: Turn on vector-based code search for large codebases
- **Performance Optimization**: Disable vector mode for smaller projects to reduce overhead
- **Feature Configuration**: Dynamically configure search capabilities per project
- **Testing & Development**: Toggle vector features during development and testing

#### ⚠️ Important Notes

- Vector mode requires additional setup and API keys for embedding providers
- Enabling vector mode may increase processing time and storage requirements
- The tool creates projects automatically if they don't exist
- Changes take effect immediately but indexing may take time for large codebases

#### 🔧 Error Handling

```javascript
// Handle errors gracefully
try {
  const result = await mcp.callTool("enabled_vector_mode", {
    projectName: "my-web-app",
    folderPath: "/invalid/path",
    enabled: true
  });
  
  if (!result.success) {
    console.error("Failed to enable vector mode:", result.error);
  }
} catch (error) {
  console.error("Tool execution failed:", error.message);
}
```

---

### find_similar_code

Find code similar to a given code snippet or file section using vector-based semantic search. This tool uses AI embeddings to understand code context and meaning, providing more intelligent similarity detection than text-based matching.

**⚠️ Vector Mode Required**: This tool only works when vector mode is enabled for the project.

#### Parameters

```typescript
interface FindSimilarCodeParams {
  projectName: string;           // The name of the project
  folderPath: string;            // Absolute path to the project folder on disk
  
  // Input source (mutually exclusive)
  code_snippet?: string;         // Direct code snippet to search for similarities
  file_path?: string;            // Path to file containing code to analyze
  line_start?: number;           // Starting line number for file section (1-indexed)
  line_end?: number;             // Ending line number for file section (1-indexed)
  
  // Search configuration (optional)
  similarity_threshold?: number; // Minimum similarity score (0.0-1.0)
  max_results?: number;          // Maximum number of results to return
}
```

#### Response

```typescript
interface FindSimilarCodeResponse {
  results: Array<{
    file_path: string;           // Path to file containing similar code
    code_section: string;        // The similar code section found
    similarity_score: number;    // Similarity score (0.0-1.0, higher is more similar)
    start_line: number;          // Starting line number of similar section
    end_line: number;            // Ending line number of similar section
    context: string;             // Additional context around the match
  }>;
  search_input: {
    type: "snippet" | "file_section";  // Type of input used
    content: string;             // The code that was searched for
    source?: string;             // Source file path (if using file_path input)
  };
  total_results: number;         // Total number of similar code sections found
  similarity_threshold: number;  // Similarity threshold used
}
```

#### Example Usage

##### Search by Code Snippet

```javascript
const result = await mcp.callTool("find_similar_code", {
  projectName: "my-web-app",
  folderPath: "/home/user/projects/my-web-app",
  code_snippet: `
    function validateEmail(email: string): boolean {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      return emailRegex.test(email);
    }
  `,
  similarity_threshold: 0.7,
  max_results: 5
});

// Response:
{
  "results": [
    {
      "file_path": "src/utils/validators.ts",
      "code_section": "function isValidEmail(emailAddress: string): boolean {\n  const pattern = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n  return pattern.test(emailAddress);\n}",
      "similarity_score": 0.92,
      "start_line": 15,
      "end_line": 18,
      "context": "// Email validation utilities"
    },
    {
      "file_path": "src/auth/validation.ts", 
      "code_section": "const validateUserEmail = (email: string) => {\n  return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email);\n};",
      "similarity_score": 0.85,
      "start_line": 42,
      "end_line": 44,
      "context": "User input validation functions"
    }
  ],
  "search_input": {
    "type": "snippet",
    "content": "function validateEmail(email: string): boolean {\n  const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n  return emailRegex.test(email);\n}"
  },
  "total_results": 2,
  "similarity_threshold": 0.7
}
```

##### Search by File Section

```javascript
const result = await mcp.callTool("find_similar_code", {
  projectName: "large-api",
  folderPath: "/home/user/projects/large-api",
  file_path: "src/controllers/userController.ts",
  line_start: 25,
  line_end: 35,
  similarity_threshold: 0.6,
  max_results: 10
});

// Response:
{
  "results": [
    {
      "file_path": "src/controllers/productController.ts",
      "code_section": "async function createProduct(req: Request, res: Response) {\n  try {\n    const product = await productService.create(req.body);\n    res.status(201).json(product);\n  } catch (error) {\n    res.status(400).json({ error: error.message });\n  }\n}",
      "similarity_score": 0.78,
      "start_line": 18,
      "end_line": 26,
      "context": "Product CRUD operations"
    }
  ],
  "search_input": {
    "type": "file_section",
    "content": "async function createUser(req: Request, res: Response) {\n  try {\n    const user = await userService.create(req.body);\n    res.status(201).json(user);\n  } catch (error) {\n    res.status(400).json({ error: error.message });\n  }\n}",
    "source": "src/controllers/userController.ts"
  },
  "total_results": 1,
  "similarity_threshold": 0.6
}
```

#### 🎯 Use Cases

- **Code Duplication Detection**: Find similar functions or code patterns that could be refactored
- **Code Reuse Discovery**: Locate existing implementations similar to what you're building
- **Pattern Analysis**: Understand common patterns and approaches across your codebase
- **Refactoring Opportunities**: Identify code sections that follow similar patterns
- **Code Review**: Find similar implementations to ensure consistency
- **Learning**: Discover how similar problems were solved elsewhere in the codebase

#### ⚠️ Prerequisites

- **Vector Mode Enabled**: Project must have vector mode activated
- **API Keys Required**: VOYAGE_API_KEY and TURBOPUFFER_API_KEY environment variables
- **Project Indexed**: The project must be indexed with vector embeddings

#### 💡 Tips for Best Results

- **Meaningful Code Sections**: Use code sections with clear functionality (10-50 lines work well)
- **Adjust Similarity Threshold**: Start with 0.7, lower to 0.5 for broader matches
- **Use Representative Code**: Choose code that represents the pattern you're looking for
- **Consider Context**: Similar functionality may be implemented differently but serve the same purpose

## Common Parameters

All tools require these standard parameters for project identification:

| Parameter | Type | Description |
|-----------|------|-------------|
| `projectName` | `string` | The name of the project (required) |
| `folderPath` | `string` | Absolute path to the project folder on disk (required) |

### Project Identification

The server automatically creates unique project identifiers based on:
1. **Project name** + **folder path**
2. **Automatic alias creation** for different path representations
3. **Project deduplication** across sessions

## Error Handling

All tools return consistent error responses following the MCP protocol:

```typescript
interface ErrorResponse {
  error: {
    code: number;            // JSON-RPC error code
    message: string;         // Human-readable error message
    category: string;        // Error category (validation, database, etc.)
    details?: Record<string, any>; // Additional error context
  };
  tool: string;              // Tool name that failed
  timestamp: string;         // ISO timestamp of error
  arguments?: Record<string, any>; // Sanitized input arguments
}
```

### Common Error Codes

| Code | Category | Description |
|------|----------|-------------|
| `-32602` | validation | Invalid or missing parameters |
| `-32603` | database | Database operation failed |
| `-32603` | file_system | File system access error |
| `-32603` | internal | Internal server error |

### Example Error Response

```json
{
  "error": {
    "code": -32602,
    "message": "Missing required fields: branch, filePath",
    "category": "validation",
    "details": {
      "missing_fields": ["branch", "filePath"],
      "provided_fields": ["projectName", "folderPath"]
    }
  },
  "tool": "get_file_description",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Best Practices

### 🎯 For Everyone: Smart Usage
- **Always start** with `check_codebase_size` to get personalized recommendations
- **Use search** for large codebases (>32,000 tokens) instead of getting full overview
- **Be specific** in search queries - "authentication middleware" vs "auth"

### 👨‍💻 For Developers: Integration Tips
- **Batch operations** when updating multiple files for better performance
- **Cache project info** to avoid repeated setup calls
- **Handle validation errors** by checking required parameters first
- **Monitor structured logs** for debugging and performance insights

### 📝 For Content: Description Quality
- **Be descriptive but concise** - focus on what the file does, not how
- **Include key technologies** and frameworks used
- **Mention important dependencies** and relationships to other files
- **Update descriptions** when files change significantly
- **Use consistent terminology** across your project descriptions

### 🔧 For Operations: Error Recovery
- **Retry database operations** on transient failures (network issues, etc.)
- **Provide meaningful resolutions** for merge conflicts that capture both perspectives
- **Set up monitoring** for error rates and performance metrics
- **Regular backups** of your description database

---

**Next Steps**: Check out the [Configuration Guide](configuration.md) for advanced server tuning options, or review the [Architecture Overview](architecture.md) for technical implementation details! 🚀
