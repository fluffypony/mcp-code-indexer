#!/usr/bin/env python3
"""
Correct MCP protocol test - shows proper initialization sequence.
Run this to test the server correctly.
"""

import json
import sys
import subprocess
import time


def test_mcp_server():
    """Test MCP server with correct protocol."""
    
    # Start server
    proc = subprocess.Popen(
        ["python", "main.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
    def send_message(msg):
        """Send JSON-RPC message."""
        json_str = json.dumps(msg) + '\n'
        print(f"→ SENDING: {json_str.strip()}", file=sys.stderr)
        proc.stdin.write(json_str)
        proc.stdin.flush()
    
    def read_response():
        """Read response from server."""
        try:
            line = proc.stdout.readline()
            if line.strip():
                response = json.loads(line.strip())
                print(f"← RECEIVED: {json.dumps(response, indent=2)}", file=sys.stderr)
                return response
        except Exception as e:
            print(f"Read error: {e}", file=sys.stderr)
        return None
    
    try:
        # Give server time to start
        time.sleep(1)
        
        print("\n=== Step 1: Initialize ===", file=sys.stderr)
        send_message({
            "jsonrpc": "2.0",
            "id": "init-001",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "capabilities": {}
            }
        })
        
        response = read_response()
        if not response:
            print("❌ No initialize response", file=sys.stderr)
            return
            
        print("\n=== Step 2: Initialized ===", file=sys.stderr)
        send_message({
            "jsonrpc": "2.0",
            "method": "initialized", 
            "params": {}
        })
        
        print("\n=== Step 3: List Tools ===", file=sys.stderr)
        send_message({
            "jsonrpc": "2.0",
            "id": "req-002",
            "method": "tools/list",
            "params": {}
        })
        
        response = read_response()
        if response:
            print("✅ SUCCESS: Got tools list!", file=sys.stderr)
        else:
            print("❌ No tools response", file=sys.stderr)
            
    except Exception as e:
        print(f"❌ Test failed: {e}", file=sys.stderr)
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    test_mcp_server()
