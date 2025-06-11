#!/usr/bin/env python3
"""
Test script to demonstrate correct MCP protocol usage.
This shows the proper initialization sequence and tool calling.
"""

import json
import sys
import subprocess
import asyncio
from asyncio import subprocess as asubprocess


async def test_mcp_server():
    """Test the MCP server with correct protocol sequence."""
    
    # Start the server process
    proc = await asubprocess.create_subprocess_exec(
        "mcp-code-indexer",
        stdin=asubprocess.PIPE,
        stdout=asubprocess.PIPE,
        stderr=asubprocess.PIPE
    )
    
    async def send_message(msg):
        """Send a JSON-RPC message to the server."""
        json_str = json.dumps(msg) + '\n'
        print(f"SENDING: {json_str.strip()}", file=sys.stderr)
        proc.stdin.write(json_str.encode())
        await proc.stdin.drain()
    
    async def read_response():
        """Read a response from the server."""
        line = await proc.stdout.readline()
        if line:
            response = json.loads(line.decode().strip())
            print(f"RECEIVED: {json.dumps(response, indent=2)}", file=sys.stderr)
            return response
        return None
    
    try:
        # Step 1: Initialize
        print("\n=== STEP 1: INITIALIZE ===", file=sys.stderr)
        await send_message({
            "jsonrpc": "2.0",
            "id": "init-001",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "capabilities": {}
            }
        })
        
        response = await read_response()
        
        # Step 2: Initialized notification
        print("\n=== STEP 2: INITIALIZED ===", file=sys.stderr)
        await send_message({
            "jsonrpc": "2.0", 
            "method": "initialized",
            "params": {}
        })
        
        # Step 3: List tools
        print("\n=== STEP 3: LIST TOOLS ===", file=sys.stderr)
        await send_message({
            "jsonrpc": "2.0",
            "id": "req-002", 
            "method": "tools/list",
            "params": {}
        })
        
        response = await read_response()
        
        # Step 4: Call a tool
        print("\n=== STEP 4: CALL TOOL ===", file=sys.stderr)
        await send_message({
            "jsonrpc": "2.0",
            "id": "req-003",
            "method": "tools/call",
            "params": {
                "name": "check_codebase_size",
                "arguments": {
                    "projectName": "test-project",
                    "folderPath": "/tmp",
                    "branch": "main"
                }
            }
        })
        
        response = await read_response()
        
        print("\n=== SUCCESS: Protocol test completed ===", file=sys.stderr)
        
    except Exception as e:
        print(f"\n=== ERROR: {e} ===", file=sys.stderr)
    finally:
        proc.terminate()
        await proc.wait()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
