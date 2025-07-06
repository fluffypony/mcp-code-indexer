# HTTP/REST API Reference 🌐

---
**Last Updated:** 2025-06-30
**Verified Against:** src/mcp_code_indexer/transport/http_transport.py
**Test Sources:** Manual verification of FastAPI implementation
**Implementation:** HTTP transport layer with Server-Sent Events
---

Complete reference for the MCP Code Indexer HTTP/REST API. This HTTP transport provides web-accessible endpoints for all MCP tools with optional authentication and real-time streaming responses.

**🎯 New to HTTP mode?** Start with the [Quick Start Guide](../README.md#-quick-start) to configure your server first.

## Quick Reference

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| [`GET /health`](#health-check) | GET | Server health check | None |
| [`GET /metrics`](#metrics) | GET | Performance metrics | Required* |
| [`GET /tools`](#list-tools) | GET | List available MCP tools | Required* |
| [`POST /mcp`](#mcp-requests) | POST | Execute MCP tool calls | Required* |
| [`GET /events/{id}`](#server-sent-events) | GET | Server-Sent Events stream | Required* |

*Authentication required only if `--auth-token` is provided

🔗 **[See Examples →](../examples/http-examples.md)**

## Table of Contents

- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Core Endpoints](#core-endpoints)
- [MCP Tool Execution](#mcp-tool-execution)
- [Server-Sent Events](#server-sent-events)
- [Error Handling](#error-handling)
- [Client Examples](#client-examples)
- [Security Configuration](#security-configuration)

## Getting Started

### Starting the HTTP Server

```bash
# Basic HTTP server on localhost:7557
mcp-code-indexer --http

# Custom host and port
mcp-code-indexer --http --host 0.0.0.0 --port 8080

# With authentication
mcp-code-indexer --http --auth-token "your-secret-token"

# With CORS configuration
mcp-code-indexer --http --cors-origins "https://localhost:3000" "https://myapp.com"
```

### Base URL

Once started, the API is available at:
```
http://127.0.0.1:7557
```

### Interactive Documentation

FastAPI automatically provides interactive API documentation:
- **Swagger UI**: `http://127.0.0.1:7557/docs`
- **ReDoc**: `http://127.0.0.1:7557/redoc`
- **OpenAPI Schema**: `http://127.0.0.1:7557/openapi.json`

## Authentication

The HTTP API supports optional Bearer token authentication. When enabled, all endpoints except `/health`, `/docs`, `/openapi.json`, and `/metrics` require authentication.

### Enabling Authentication

```bash
mcp-code-indexer --http --auth-token "your-secret-token"
```

### Using Authentication

Include the Bearer token in the Authorization header:

```bash
curl -H "Authorization: Bearer your-secret-token" \
     http://localhost:7557/tools
```

```javascript
// JavaScript fetch
const response = await fetch('http://localhost:7557/tools', {
  headers: {
    'Authorization': 'Bearer your-secret-token'
  }
});
```

```python
# Python requests
import requests

headers = {'Authorization': 'Bearer your-secret-token'}
response = requests.get('http://localhost:7557/tools', headers=headers)
```

## Core Endpoints

### Health Check

Check if the server is running and healthy.

**Endpoint:** `GET /health`
**Authentication:** Not required

#### Response

```typescript
interface HealthResponse {
  status: "healthy";
  transport: "http";
}
```

#### Example

```bash
curl http://localhost:7557/health
```

```json
{
  "status": "healthy",
  "transport": "http"
}
```

---

### Metrics

Get performance metrics and connection statistics.

**Endpoint:** `GET /metrics`
**Authentication:** Required if auth enabled

#### Response

```typescript
interface MetricsResponse {
  http: {
    requests_total: number;
    requests_per_second: number;
    error_rate: number;
    avg_response_time_ms: number;
  };
  connections: {
    active_sse_connections: number;
    connection_ids: string[];
  };
}
```

#### Example

```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:7557/metrics
```

```json
{
  "http": {
    "requests_total": 1247,
    "requests_per_second": 12.3,
    "error_rate": 0.02,
    "avg_response_time_ms": 145.7
  },
  "connections": {
    "active_sse_connections": 3,
    "connection_ids": ["conn-123", "conn-456", "conn-789"]
  }
}
```

---

### List Tools

Get a list of all available MCP tools with their schemas.

**Endpoint:** `GET /tools`
**Authentication:** Required if auth enabled

#### Response

```typescript
interface ToolsResponse {
  tools: Array<{
    name: string;
    description: string;
    inputSchema: object;
  }>;
}
```

#### Example

```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:7557/tools
```

```json
{
  "tools": [
    {
      "name": "get_file_description",
      "description": "Retrieves the stored description for a specific file",
      "inputSchema": {
        "type": "object",
        "properties": {
          "projectName": {"type": "string"},
          "folderPath": {"type": "string"},
          "filePath": {"type": "string"}
        },
        "required": ["projectName", "folderPath", "filePath"]
      }
    }
  ]
}
```

## MCP Tool Execution

Execute MCP tools using JSON-RPC format requests.

**Endpoint:** `POST /mcp`
**Authentication:** Required if auth enabled

### Request Format

```typescript
interface MCPRequest {
  jsonrpc: "2.0";
  method: "tools/call";
  params: {
    name: string;           // Tool name
    arguments: object;      // Tool arguments
  };
  id?: string;              // Optional request ID
}
```

### Response Format

```typescript
interface MCPResponse {
  jsonrpc: "2.0";
  result?: any;             // Tool result (on success)
  error?: {                 // Error details (on failure)
    code: number;
    message: string;
  };
  id?: string;              // Request ID (if provided)
}
```

### Available Tools

All 11 MCP tools are available via HTTP. See the [API Reference](api-reference.md) for complete tool documentation.

| Tool Name | Purpose |
|-----------|---------|
| `get_file_description` | Retrieve file summary |
| `update_file_description` | Store file analysis |
| `check_codebase_size` | Navigation recommendations |
| `search_descriptions` | Find files by functionality |
| `get_codebase_overview` | Project architecture |
| `find_missing_descriptions` | Scan for undocumented files |
| `get_all_descriptions` | Complete project structure |
| `get_word_frequency` | Technical vocabulary |
| `update_codebase_overview` | Create project docs |
| `search_codebase_overview` | Search overviews |
| `check_database_health` | System monitoring |

### Example Tool Calls

#### Get File Description

```bash
curl -X POST -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-token" \
     -d '{
       "jsonrpc": "2.0",
       "method": "tools/call",
       "params": {
         "name": "get_file_description",
         "arguments": {
           "projectName": "my-app",
           "folderPath": "/home/user/my-app",
           "filePath": "src/main.ts"
         }
       },
       "id": "req-123"
     }' \
     http://localhost:7557/mcp
```

```json
{
  "jsonrpc": "2.0",
  "result": {
    "exists": true,
    "description": "Main application entry point with Express server setup",
    "lastModified": "2024-01-15T10:30:00Z",
    "fileHash": "abc123def456"
  },
  "id": "req-123"
}
```

#### Search Descriptions

```bash
curl -X POST -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-token" \
     -d '{
       "jsonrpc": "2.0",
       "method": "tools/call",
       "params": {
         "name": "search_descriptions",
         "arguments": {
           "projectName": "my-app",
           "folderPath": "/home/user/my-app",
           "query": "authentication middleware"
         }
       }
     }' \
     http://localhost:7557/mcp
```

```json
{
  "jsonrpc": "2.0",
  "result": {
    "results": [
      {
        "filePath": "src/middleware/auth.ts",
        "description": "JWT authentication middleware with role-based access control",
        "relevanceScore": 0.95
      }
    ],
    "totalResults": 1,
    "query": "authentication middleware"
  }
}
```

#### Check Database Health

```bash
curl -X POST -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-token" \
     -d '{
       "jsonrpc": "2.0",
       "method": "tools/call",
       "params": {
         "name": "check_database_health",
         "arguments": {}
       }
     }' \
     http://localhost:7557/mcp
```

```json
{
  "jsonrpc": "2.0",
  "result": {
    "health_status": {
      "overall_health": "healthy",
      "database": {
        "pool_healthy": true,
        "active_connections": 2,
        "total_connections": 3,
        "failed_connections": 0,
        "avg_response_time_ms": 15.3,
        "wal_size_mb": 12.4
      },
      "performance": {
        "current_throughput": 145.7,
        "target_throughput": 800,
        "p95_latency_ms": 25.8,
        "error_rate": 0.02,
        "operations_last_minute": 120
      }
    },
    "recommendations": [],
    "last_check": "2024-01-15T14:30:00Z"
  }
}
```

## Server-Sent Events

The HTTP API supports Server-Sent Events (SSE) for real-time streaming of tool responses and system events.

**Endpoint:** `GET /events/{connection_id}`
**Authentication:** Required if auth enabled

### Connecting to Events

```javascript
const connectionId = 'conn-' + Date.now();
const eventSource = new EventSource(
  `http://localhost:7557/events/${connectionId}`,
  {
    headers: {
      'Authorization': 'Bearer your-token'
    }
  }
);

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received event:', data);
};

eventSource.onerror = function(event) {
  console.error('SSE connection error:', event);
};
```

### Event Types

```typescript
interface SSEEvent {
  type: 'tool_result' | 'progress' | 'error' | 'keepalive' | 'disconnect';
  data?: any;
  timestamp: string;
}
```

#### Tool Result Event
```json
{
  "type": "tool_result",
  "data": {
    "tool": "search_descriptions",
    "result": { /* tool response */ }
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

#### Progress Event
```json
{
  "type": "progress",
  "data": {
    "operation": "searching_files",
    "progress": 0.65,
    "message": "Processed 65 of 100 files"
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

#### Keepalive Event
```json
{
  "type": "keepalive",
  "timestamp": "2024-01-15T14:30:00Z"
}
```

## Error Handling

The HTTP API returns standard HTTP status codes and JSON-RPC error responses.

### HTTP Status Codes

| Status | Description |
|--------|-------------|
| `200` | Success |
| `400` | Bad Request (invalid JSON/parameters) |
| `401` | Unauthorized (invalid or missing token) |
| `404` | Not Found (invalid endpoint) |
| `422` | Validation Error (Pydantic validation failed) |
| `500` | Internal Server Error |

### JSON-RPC Error Codes

| Code | Description |
|------|-------------|
| `-32600` | Invalid Request |
| `-32601` | Method Not Found |
| `-32602` | Invalid Params |
| `-32603` | Internal Error |

### Error Response Format

```typescript
interface ErrorResponse {
  jsonrpc: "2.0";
  error: {
    code: number;
    message: string;
  };
  id?: string;
}
```

### Example Error Responses

#### Authentication Error
```bash
# HTTP 401 Unauthorized
curl http://localhost:7557/tools
```

```json
{
  "detail": "Invalid authentication token"
}
```

#### Invalid Tool Name
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32601,
    "message": "Unknown tool: invalid_tool_name"
  },
  "id": "req-123"
}
```

#### Invalid Parameters
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32602,
    "message": "Invalid params: Missing required field 'projectName'"
  },
  "id": "req-123"
}
```

## Client Examples

### JavaScript/TypeScript Client

```typescript
class MCPHttpClient {
  constructor(
    private baseUrl: string,
    private authToken?: string
  ) {}

  private async request(endpoint: string, options: RequestInit = {}) {
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async callTool(toolName: string, arguments: object) {
    return this.request('/mcp', {
      method: 'POST',
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'tools/call',
        params: {
          name: toolName,
          arguments,
        },
        id: Math.random().toString(36).substr(2, 9),
      }),
    });
  }

  async getHealth() {
    return this.request('/health');
  }

  async getTools() {
    return this.request('/tools');
  }

  async getMetrics() {
    return this.request('/metrics');
  }
}

// Usage
const client = new MCPHttpClient('http://localhost:7557', 'your-token');

const result = await client.callTool('get_file_description', {
  projectName: 'my-app',
  folderPath: '/home/user/my-app',
  filePath: 'src/main.ts',
});
```

### Python Client

```python
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional

class MCPHttpClient:
    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_headers(self) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        return headers

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool via HTTP."""
        payload = {
            'jsonrpc': '2.0',
            'method': 'tools/call',
            'params': {
                'name': tool_name,
                'arguments': arguments,
            },
            'id': f'req-{asyncio.current_task().get_name()}'
        }

        async with self.session.post(
            f'{self.base_url}/mcp',
            json=payload,
            headers=self._get_headers()
        ) as response:
            response.raise_for_status()
            result = await response.json()

            if 'error' in result:
                raise Exception(f"Tool error: {result['error']['message']}")

            return result['result']

    async def get_health(self) -> Dict[str, Any]:
        """Check server health."""
        async with self.session.get(
            f'{self.base_url}/health',
            headers=self._get_headers()
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def get_tools(self) -> Dict[str, Any]:
        """Get available tools."""
        async with self.session.get(
            f'{self.base_url}/tools',
            headers=self._get_headers()
        ) as response:
            response.raise_for_status()
            return await response.json()

# Usage
async def main():
    async with MCPHttpClient('http://localhost:7557', 'your-token') as client:
        # Check health
        health = await client.get_health()
        print(f"Server status: {health['status']}")

        # Search for files
        result = await client.call_tool('search_descriptions', {
            'projectName': 'my-app',
            'folderPath': '/home/user/my-app',
            'query': 'authentication middleware'
        })

        print(f"Found {result['totalResults']} files")
        for file_result in result['results']:
            print(f"  {file_result['filePath']}: {file_result['relevanceScore']:.2f}")

if __name__ == '__main__':
    asyncio.run(main())
```

### cURL Examples

```bash
#!/bin/bash

# Set your configuration
BASE_URL="http://localhost:7557"
AUTH_TOKEN="your-secret-token"

# Health check (no auth required)
echo "=== Health Check ==="
curl -s "$BASE_URL/health" | jq

# Get available tools
echo -e "\n=== Available Tools ==="
curl -s -H "Authorization: Bearer $AUTH_TOKEN" \
     "$BASE_URL/tools" | jq '.tools[].name'

# Search for authentication-related files
echo -e "\n=== Search Results ==="
curl -s -X POST \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -d '{
       "jsonrpc": "2.0",
       "method": "tools/call",
       "params": {
         "name": "search_descriptions",
         "arguments": {
           "projectName": "my-app",
           "folderPath": "/home/user/my-app",
           "query": "authentication"
         }
       }
     }' \
     "$BASE_URL/mcp" | jq '.result.results[]'

# Check database health
echo -e "\n=== Database Health ==="
curl -s -X POST \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -d '{
       "jsonrpc": "2.0",
       "method": "tools/call",
       "params": {
         "name": "check_database_health",
         "arguments": {}
       }
     }' \
     "$BASE_URL/mcp" | jq '.result.health_status.overall_health'
```

## Security Configuration

### CORS Configuration

Control cross-origin requests by specifying allowed origins:

```bash
# Allow specific origins
mcp-code-indexer --http --cors-origins "https://localhost:3000" "https://myapp.com"

# Allow all origins (default)
mcp-code-indexer --http --cors-origins "*"

# No CORS (strict same-origin)
mcp-code-indexer --http --cors-origins
```

### Security Headers

The HTTP transport automatically includes security middleware with these headers:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains` (HTTPS only)

### Production Deployment

For production deployments:

1. **Use HTTPS**: Deploy behind a reverse proxy with SSL termination
2. **Enable Authentication**: Always use `--auth-token` in production
3. **Restrict CORS**: Specify exact allowed origins
4. **Firewall**: Restrict access to trusted networks
5. **Monitor**: Use `/metrics` endpoint for monitoring

Example production startup:
```bash
mcp-code-indexer --http \
  --host 127.0.0.1 \
  --port 7557 \
  --auth-token "$MCP_AUTH_TOKEN" \
  --cors-origins "https://yourdomain.com" \
  --log-level INFO
```

---

**Next Steps**: Check out the [Q&A Interface Guide](qa-interface.md) for AI-powered query capabilities, or review the [Configuration Guide](configuration.md) for advanced server tuning! 🚀
