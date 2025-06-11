#!/usr/bin/env python3
"""Simple MCP protocol test with proper timing"""

import json
import subprocess
import time
import sys


def test_mcp():
    """Test MCP with proper protocol sequence."""
    
    proc = subprocess.Popen(
        ["python", "main.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
    def send(msg):
        json_str = json.dumps(msg) + '\n'
        print(f"→ {json_str.strip()}", file=sys.stderr)
        proc.stdin.write(json_str)
        proc.stdin.flush()
    
    def read():
        line = proc.stdout.readline()
        if line.strip():
            resp = json.loads(line.strip())
            print(f"← {json.dumps(resp, indent=2)}", file=sys.stderr)
            return resp
        return None
    
    try:
        time.sleep(0.5)  # Let server start
        
        # Step 1: Initialize
        send({
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize", 
            "params": {"protocolVersion": "1.0", "capabilities": {}}
        })
        
        init_resp = read()
        if not init_resp:
            print("❌ No init response", file=sys.stderr)
            return
            
        # Step 2: Initialized notification
        send({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        })
        
        # Step 3: List tools
        send({
            "jsonrpc": "2.0", 
            "id": "tools",
            "method": "tools/list",
            "params": {}
        })
        
        tools_resp = read()
        if tools_resp and "result" in tools_resp:
            print("✅ SUCCESS! Got tools list", file=sys.stderr)
            print(f"Found {len(tools_resp['result'])} tools", file=sys.stderr)
        else:
            print("❌ No tools response", file=sys.stderr)
            
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    test_mcp()
