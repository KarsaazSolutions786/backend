#!/usr/bin/env python3
"""
Test script to verify PORT environment variable handling
Run this to debug Railway PORT issues
"""

import os

def test_port_handling():
    print("=== PORT Environment Variable Test ===")
    
    # Test 1: Check if PORT exists
    port_env = os.getenv('PORT')
    print(f"1. PORT environment variable: '{port_env}'")
    
    # Test 2: Check all environment variables
    print("\n2. All environment variables:")
    for key, value in os.environ.items():
        print(f"   {key}={value}")
    
    # Test 3: Try to parse PORT
    try:
        if port_env is None:
            port = 8000
            print(f"\n3. PORT not set, using default: {port}")
        else:
            port = int(port_env)
            print(f"\n3. Successfully parsed PORT: {port}")
    except (ValueError, TypeError) as e:
        print(f"\n3. Error parsing PORT '{port_env}': {e}")
        port = 8000
        print(f"   Using default port: {port}")
    
    # Test 4: Validate port range
    if 1 <= port <= 65535:
        print(f"\n4. Port {port} is valid (1-65535)")
    else:
        print(f"\n4. Port {port} is invalid, should be 1-65535")
    
    print(f"\n✅ Final port to use: {port}")
    return port

if __name__ == "__main__":
    test_port_handling() 