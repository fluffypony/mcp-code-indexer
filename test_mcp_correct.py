#!/usr/bin/env python3
"""
Test script for MCP Code Indexer protocol compliance with MCP 1.9.3
"""

import json
import subprocess
import time
import sys
import threading
from io import StringIO


def test_mcp_protocol():
    """Test the complete MCP protocol sequence with correct format."""
    
    # Start the server process
    print("🚀 Starting MCP server...")
    process = subprocess.Popen(
        ["mcp-code-indexer"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
    # Give server a moment to start
    time.sleep(1)
    
    try:
        # Test 1: Initialize request with correct clientInfo
        print("📡 Testing initialize request with clientInfo...")
        initialize_request = {
            "jsonrpc": "2.0",
            "id": "init-001",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "capabilities": {},
                "clientInfo": {
                    "name": "mcp-test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        # Send initialize request
        request_json = json.dumps(initialize_request) + "\n"
        process.stdin.write(request_json)
        process.stdin.flush()
        
        print(f"📤 Sent: {request_json.strip()}")
        
        # Wait for initialize response and read it
        time.sleep(1)
        
        # Read the initialize response to ensure handshake is complete
        stdout_data = ""
        try:
            # Give server time to process and respond
            process.stdout.settimeout(2.0)
            line = process.stdout.readline()
            if line:
                stdout_data += line
                print(f"📥 Initialize response: {line.strip()}")
        except:
            pass
        
        # Test 2: Send initialized notification (correct method name) 
        print("\n📡 Sending initialized notification...")
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        
        notification_json = json.dumps(initialized_notification) + "\n"
        process.stdin.write(notification_json)
        process.stdin.flush()
        
        print(f"📤 Sent: {notification_json.strip()}")
        
        # Wait for notification to be processed
        time.sleep(2)
        
        # Test 3: List tools
        print("\n📡 Testing tools/list request...")
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": "tools-001",
            "method": "tools/list",
            "params": {}
        }
        
        tools_json = json.dumps(list_tools_request) + "\n"
        process.stdin.write(tools_json)
        process.stdin.flush()
        
        print(f"📤 Sent: {tools_json.strip()}")
        time.sleep(2)
        
        # Test 4: Call a tool
        print("\n📡 Testing tool call...")
        tool_call_request = {
            "jsonrpc": "2.0",
            "id": "call-001",
            "method": "tools/call",
            "params": {
                "name": "check_codebase_size",
                "arguments": {
                    "projectName": "test-project",
                    "folderPath": "/tmp/test",
                    "branch": "main"
                }
            }
        }
        
        call_json = json.dumps(tool_call_request) + "\n"
        process.stdin.write(call_json)
        process.stdin.flush()
        
        print(f"📤 Sent: {call_json.strip()}")
        time.sleep(2)
        
        # Check server responses
        print("\n📥 Server responses:")
        stdout_data, stderr_data = process.communicate(timeout=5)
        
        if stdout_data:
            print("STDOUT:")
            for line in stdout_data.split('\n'):
                if line.strip():
                    print(f"  {line}")
        
        if stderr_data:
            print("\nSTDERR:")
            for line in stderr_data.split('\n'):
                if line.strip():
                    print(f"  {line}")
        
        # Check exit code
        exit_code = process.returncode
        print(f"\n🏁 Server exit code: {exit_code}")
        
        if exit_code == 0:
            print("✅ Protocol test completed successfully!")
        else:
            print("❌ Protocol test failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Server response timeout")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        process.kill()
        return False
    finally:
        if process.poll() is None:
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()
    
    return True


if __name__ == "__main__":
    print("🧪 MCP Protocol Test - Version 1.9.3 Compliance")
    print("=" * 50)
    
    success = test_mcp_protocol()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
