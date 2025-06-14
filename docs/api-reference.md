# MCP Tools API Reference 📖

Complete reference for all 10 MCP tools provided by the Code Indexer server. Whether you're building AI agents or integrating MCP tools directly, this guide shows you exactly how to use each tool effectively.

**🎯 New to MCP Code Indexer?** Start with the [Quick Start Guide](../README.md#-quick-start) to set up your server first.

## Table of Contents

- [Core Operations](#core-operations)
  - [get_file_description](#get_file_description)
  - [update_file_description](#update_file_description)
  - [check_codebase_size](#check_codebase_size)
- [Batch Operations](#batch-operations)
  - [find_missing_descriptions](#find_missing_descriptions)
  - [update_missing_descriptions](#update_missing_descriptions)
- [Search & Discovery](#search--discovery)
  - [search_descriptions](#search_descriptions)
  - [get_all_descriptions](#get_all_descriptions)
  - [get_codebase_overview](#get_codebase_overview)
  - [get_word_frequency](#get_word_frequency)
- [Advanced Features](#advanced-features)
  - [update_codebase_overview](#update_codebase_overview)
- [System Monitoring](#system-monitoring)
  - [check_database_health](#check_database_health)
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

Scans the project folder to find files that don't have descriptions yet. This is stage 1 of a two-stage process for updating missing descriptions. Respects .gitignore and common ignore patterns.

#### Parameters

```typescript
interface FindMissingDescriptionsParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder
}
```

#### Response

```typescript
interface FindMissingDescriptionsResponse {
  missingFiles: string[];     // Array of relative file paths without descriptions
  totalMissing: number;       // Count of missing files
  existingDescriptions: number; // Count of files that already have descriptions
  projectStats: {             // Project statistics
    total_files: number;
    trackable_files: number;
    ignored_files: number;
    largest_file_size: number;
    file_extensions: Record<string, number>;
  };
}
```

#### Example

```javascript
const result = await mcp.callTool("find_missing_descriptions", {
  projectName: "new-project",
  folderPath: "/home/user/projects/new-project"
});

// Response:
{
  "missingFiles": [
    "src/components/NewFeature.tsx",
    "src/hooks/useLocalStorage.ts",
    "tests/integration/api.test.ts"
  ],
  "totalMissing": 3,
  "existingDescriptions": 12,
  "projectStats": {
    "total_files": 45,
    "trackable_files": 15,
    "ignored_files": 30,
    "largest_file_size": 15420,
    "file_extensions": {
      ".ts": 8,
      ".tsx": 4,
      ".js": 2,
      ".md": 1
    }
  }
}
```

---

### update_missing_descriptions

Batch updates descriptions for multiple files at once. This is stage 2 after find_missing_descriptions. Provide descriptions for all or some of the missing files identified in stage 1.

#### Parameters

```typescript
interface UpdateMissingDescriptionsParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder
  descriptions: Array<{      // Array of file paths and their descriptions
    filePath: string;        // Relative path to the file
    description: string;     // Detailed description of the file
  }>;
}
```

#### Response

```typescript
interface UpdateMissingDescriptionsResponse {
  success: boolean;          // Whether the batch update succeeded
  updatedFiles: number;      // Number of files successfully updated
  files: string[];           // Array of file paths that were updated
  message: string;           // Success message with summary
}
```

#### Example

```javascript
const result = await mcp.callTool("update_missing_descriptions", {
  projectName: "new-project",
  folderPath: "/home/user/projects/new-project",
  descriptions: [
    {
      filePath: "src/components/NewFeature.tsx",
      description: "React component implementing the new dashboard feature with real-time updates"
    },
    {
      filePath: "src/hooks/useLocalStorage.ts",
      description: "Custom React hook for managing localStorage with TypeScript support and serialization"
    },
    {
      filePath: "tests/integration/api.test.ts", 
      description: "Integration tests for API endpoints covering authentication and data retrieval"
    }
  ]
});

// Response:
{
  "success": true,
  "updatedFiles": 3,
  "files": [
    "src/components/NewFeature.tsx",
    "src/hooks/useLocalStorage.ts", 
    "tests/integration/api.test.ts"
  ],
  "message": "Successfully updated descriptions for 3 files"
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

### get_codebase_overview

Returns the complete file and folder structure of a codebase with all descriptions. For large codebases (exceeding the configured token limit), this will recommend using search_descriptions instead.

#### Parameters

```typescript
interface GetCodebaseOverviewParams {
  projectName: string;        // The name of the project
  folderPath: string;         // Absolute path to the project folder
}
```

#### Response

```typescript
interface GetCodebaseOverviewResponse {
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
const result = await mcp.callTool("get_codebase_overview", {
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
